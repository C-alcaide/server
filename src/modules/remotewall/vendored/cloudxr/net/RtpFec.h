// RTP-style fragmentation + XOR FEC for minimum-latency, NO-ARQ transport.
//
// Each encoded frame is split into fixed-size chunks (one per packet). For every
// group of G data packets we send one XOR parity packet, which lets the receiver
// recover a single lost packet per group WITHOUT any retransmission round-trip
// (the latency win over ARQ/SRT on a clean link). Losses beyond FEC capacity
// leave the frame incomplete; in the real pipeline rolling intra-refresh heals
// the picture within one refresh cycle — again with no upstream request.
//
// The XOR-protected unit is [u16 chunkLen][chunk padded to mtu], so a recovered
// packet's true length is reconstructed along with its bytes.
#pragma once
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <vector>

namespace rtpfec {

#pragma pack(push, 1)
struct PktHdr {
    uint32_t magic;       // 'R','T','F','1'
    uint16_t tileId;
    uint16_t type;        // 0 = data, 1 = fec
    uint32_t frameIndex;
    uint16_t seq;         // data: chunk index; fec: group start index
    uint16_t dataCount;   // total data chunks in the frame
    uint16_t grp;         // FEC group size G (fec: actual chunks in this group)
    uint16_t mtu;         // fixed chunk size used for this frame
    uint32_t payloadLen;  // bytes following the header
    uint64_t sendTimeNs;  // frame fragmentation timestamp (for latency)
};
#pragma pack(pop)

static const uint32_t kMagic = 0x31465452; // 'RTF1' little-endian

// Serialize one packet (header + payload) into out.
inline void emit(std::vector<std::vector<uint8_t>>& out, const PktHdr& h, const uint8_t* payload) {
    std::vector<uint8_t> pkt(sizeof(PktHdr) + h.payloadLen);
    std::memcpy(pkt.data(), &h, sizeof(PktHdr));
    if (h.payloadLen) std::memcpy(pkt.data() + sizeof(PktHdr), payload, h.payloadLen);
    out.push_back(std::move(pkt));
}

// Fragment one frame into data + FEC packets.
inline std::vector<std::vector<uint8_t>> fragment(uint16_t tileId, uint32_t frameIndex,
                                                  const uint8_t* frame, size_t len,
                                                  uint16_t mtu, uint16_t G, uint64_t sendTimeNs) {
    std::vector<std::vector<uint8_t>> out;
    uint16_t dataCount = (uint16_t)((len + mtu - 1) / mtu);
    if (dataCount == 0) dataCount = 1;

    PktHdr h{}; h.magic = kMagic; h.tileId = tileId; h.frameIndex = frameIndex;
    h.dataCount = dataCount; h.grp = G; h.mtu = mtu; h.sendTimeNs = sendTimeNs;

    // Data packets.
    for (uint16_t i = 0; i < dataCount; ++i) {
        size_t off = (size_t)i * mtu;
        uint16_t clen = (uint16_t)((off + mtu <= len) ? mtu : (len - off));
        h.type = 0; h.seq = i; h.payloadLen = clen;
        emit(out, h, frame + off);
    }

    // FEC parity packets (one per group of G), XOR of protected units.
    const int unit = 2 + mtu; // [u16 len][chunk]
    std::vector<uint8_t> acc(unit);
    for (uint16_t g0 = 0; g0 < dataCount; g0 += G) {
        std::fill(acc.begin(), acc.end(), 0);
        uint16_t actual = 0;
        for (uint16_t i = g0; i < dataCount && i < g0 + G; ++i, ++actual) {
            size_t off = (size_t)i * mtu;
            uint16_t clen = (uint16_t)((off + mtu <= len) ? mtu : (len - off));
            std::vector<uint8_t> u(unit, 0);
            std::memcpy(u.data(), &clen, 2);
            std::memcpy(u.data() + 2, frame + off, clen);
            for (int b = 0; b < unit; ++b) acc[b] ^= u[b];
        }
        h.type = 1; h.seq = g0; h.grp = actual; h.payloadLen = (uint32_t)unit;
        emit(out, h, acc.data());
        h.grp = G;
    }
    return out;
}

// Reassembles frames from packets, recovering single per-group losses via FEC.
class Reassembler {
public:
    // cb(tileId, frameIndex, frameBytes, sendTimeNs, recoveredCount)
    using Callback = std::function<void(uint16_t, uint32_t, const std::vector<uint8_t>&, uint64_t, int)>;
    explicit Reassembler(Callback cb) : m_cb(std::move(cb)) {}

    struct Stats { uint64_t dataRecv = 0, fecRecv = 0, recovered = 0, framesDone = 0, framesIncomplete = 0; };
    const Stats& stats() const { return m_stats; }

    void onPacket(const uint8_t* data, int n) {
        if (n < (int)sizeof(PktHdr)) return;
        PktHdr h; std::memcpy(&h, data, sizeof(PktHdr));
        if (h.magic != kMagic) return;
        const uint8_t* pl = data + sizeof(PktHdr);
        Key k{ h.tileId, h.frameIndex };
        Frame& fr = m_frames[k];
        if (fr.done) return;
        fr.dataCount = h.dataCount; fr.mtu = h.mtu; fr.sendTimeNs = h.sendTimeNs;
        if (h.type == 0) {
            m_stats.dataRecv++;
            if (!fr.chunks.count(h.seq)) fr.chunks[h.seq] = std::vector<uint8_t>(pl, pl + h.payloadLen);
        } else {
            m_stats.fecRecv++;
            Parity p; p.start = h.seq; p.grp = h.grp; p.unit.assign(pl, pl + h.payloadLen);
            fr.parities[h.seq] = std::move(p);
        }
        tryComplete(k, fr);
    }

    // Force-finalize any still-incomplete frames (e.g., at end of stream).
    void flushIncomplete() {
        for (auto& kv : m_frames)
            if (!kv.second.done) m_stats.framesIncomplete++;
    }

private:
    struct Key { uint16_t tile; uint32_t frame; bool operator<(const Key& o) const { return tile != o.tile ? tile < o.tile : frame < o.frame; } };
    struct Parity { uint16_t start, grp; std::vector<uint8_t> unit; };
    struct Frame {
        uint16_t dataCount = 0, mtu = 0; uint64_t sendTimeNs = 0; bool done = false;
        std::map<uint16_t, std::vector<uint8_t>> chunks;
        std::map<uint16_t, Parity> parities;
        int recovered = 0;
    };

    void tryComplete(const Key& k, Frame& fr) {
        if (fr.dataCount == 0) return;
        bool progressed = true;
        while (progressed) {
            progressed = false;
            if ((int)fr.chunks.size() == fr.dataCount) break;
            // Attempt FEC recovery: a group with exactly one missing data chunk.
            for (auto& pkv : fr.parities) {
                uint16_t start = pkv.second.start, grp = pkv.second.grp;
                int missing = -1, missCount = 0;
                for (uint16_t i = start; i < start + grp && i < fr.dataCount; ++i)
                    if (!fr.chunks.count(i)) { missing = i; missCount++; }
                if (missCount == 1) {
                    const int unit = pkv.second.unit.size() ? (int)pkv.second.unit.size() : (2 + fr.mtu);
                    std::vector<uint8_t> acc = pkv.second.unit;
                    for (uint16_t i = start; i < start + grp && i < fr.dataCount; ++i) {
                        if (i == (uint16_t)missing) continue;
                        const auto& c = fr.chunks[i];
                        std::vector<uint8_t> u(unit, 0);
                        uint16_t clen = (uint16_t)c.size(); std::memcpy(u.data(), &clen, 2);
                        std::memcpy(u.data() + 2, c.data(), c.size());
                        for (int b = 0; b < unit; ++b) acc[b] ^= u[b];
                    }
                    uint16_t clen = 0; std::memcpy(&clen, acc.data(), 2);
                    if (clen <= fr.mtu) {
                        fr.chunks[(uint16_t)missing] = std::vector<uint8_t>(acc.begin() + 2, acc.begin() + 2 + clen);
                        fr.recovered++; m_stats.recovered++; progressed = true;
                    }
                }
            }
        }
        if ((int)fr.chunks.size() == fr.dataCount) {
            std::vector<uint8_t> frame;
            for (uint16_t i = 0; i < fr.dataCount; ++i) frame.insert(frame.end(), fr.chunks[i].begin(), fr.chunks[i].end());
            fr.done = true; m_stats.framesDone++;
            m_cb(k.tile, k.frame, frame, fr.sendTimeNs, fr.recovered);
            fr.chunks.clear(); fr.parities.clear();
        }
    }

    Callback m_cb;
    std::map<Key, Frame> m_frames;
    Stats m_stats;
};

} // namespace rtpfec
