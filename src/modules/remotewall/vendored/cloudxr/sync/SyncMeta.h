// Per-frame synchronization metadata carried INSIDE the elementary stream so
// that an independent decoder can align N tile-streams at the output.
//
// Genlock aligns the physical SDI instant; this metadata aligns the logical
// CONTENT: every encoded frame carries the global frame index (plus tile/grid
// identity and an origin timestamp). The multistream receiver buffers decoded
// tiles keyed by globalFrameIndex and only schedules a wall-frame to DeckLink
// once all tiles for that index are present (or a deadline passes).
//
// Transport in-band:
//   - H.264 / HEVC: SEI user_data_unregistered (payloadType 5) = 16-byte UUID
//     + serialized SyncMeta. NVENC inserts it via seiPayloadArray; NVDEC's
//     parser surfaces it on decode.
//   - AV1: an ITU-T T.35 metadata OBU carrying the same UUID + payload.
//
// This header is dependency-free so it can be shared by encoder, decoder and
// the standalone scanner.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

namespace sync {

// Fixed UUID identifying our user_data_unregistered SEI / T.35 payload.
static const uint8_t kUuid[16] = {
    0x9a, 0x7e, 0x53, 0x59, 0x4e, 0x43, 0x4d, 0x45,
    0x54, 0x41, 0x31, 0xc4, 0xd2, 0xe1, 0xf0, 0x5b};

// Full per-frame camera description (all floats, contiguous -> serialized as 62
// little-endian f32). Matrices are 4x4 row-major in the source engine's space.
struct CameraMeta {
    float camToWorld[16] = {0}; // camera transform (this frame)
    float worldToCam[16] = {0}; // view matrix = inverse(camToWorld)
    float proj[16]       = {0}; // projection matrix
    float fx = 0, fy = 0, cx = 0, cy = 0;             // intrinsics (px)
    float fovXDeg = 0, fovYDeg = 0, aspect = 0, focalMm = 0; // optics
    float left = 0, right = 0, bottom = 0, top = 0, nearZ = 0, farZ = 0; // frustum
};
static_assert(sizeof(CameraMeta) == 62 * 4, "CameraMeta must be packed floats");

struct SyncMeta {
    uint16_t version  = 3;
    uint16_t tileId   = 0;
    uint16_t gridCols = 1, gridRows = 1;
    uint16_t tileCol  = 0, tileRow  = 0;
    uint32_t wallW = 0, wallH = 0, tileW = 0, tileH = 0;
    uint32_t fpsNum = 60, fpsDen = 1;
    uint64_t globalFrameIndex = 0;
    uint64_t originTimeNs = 0;      // encode-time stamp, for latency accounting
    // ---- v2 ----
    uint64_t createTimeNsUtc = 0;   // frame creation time (engine), ns UTC
    char     colorSpace[16] = {0};  // free text, e.g. "BT709", "BT2020/PQ"
    CameraMeta cam;                 // full camera (optics + frustum + matrices)
    // ---- v3: SMPTE timecode the SOURCE app generated (Unreal/Resolve/CasparCG
    // timecode provider). Carried alongside globalFrameIndex so the plato has
    // the authored timecode, not a receiver-side count. tcValid=0 => no source
    // timecode on this frame (ignore h/m/s/f). ----
    uint8_t  tcHours = 0, tcMinutes = 0, tcSeconds = 0, tcFrames = 0;
    uint8_t  tcDrop = 0;             // 1 = drop-frame (SMPTE 12M)
    uint8_t  tcValid = 0;            // 1 = h/m/s/f are meaningful
};

// --- little-endian writers/readers ------------------------------------------
namespace detail {
inline void put16(std::vector<uint8_t>& v, uint16_t x) { v.push_back(x & 0xFF); v.push_back((x >> 8) & 0xFF); }
inline void put32(std::vector<uint8_t>& v, uint32_t x) { for (int i = 0; i < 4; ++i) v.push_back((x >> (8 * i)) & 0xFF); }
inline void put64(std::vector<uint8_t>& v, uint64_t x) { for (int i = 0; i < 8; ++i) v.push_back((x >> (8 * i)) & 0xFF); }
inline uint16_t get16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
inline uint32_t get32(const uint8_t* p) { uint32_t x = 0; for (int i = 0; i < 4; ++i) x |= (uint32_t)p[i] << (8 * i); return x; }
inline uint64_t get64(const uint8_t* p) { uint64_t x = 0; for (int i = 0; i < 8; ++i) x |= (uint64_t)p[i] << (8 * i); return x; }
inline void putF32(std::vector<uint8_t>& v, float f) { uint32_t u; std::memcpy(&u, &f, 4); put32(v, u); }
inline float getF32(const uint8_t* p) { uint32_t u = get32(p); float f; std::memcpy(&f, &u, 4); return f; }
} // namespace detail

// 'SYNC' + v1 fields (56) + createTimeNsUtc(8) + colorSpace[16] + camera(62*f32=248)
constexpr int kSerializedSize = 56 + 8 + 16 + 62 * 4; // = 328

inline std::vector<uint8_t> serialize(const SyncMeta& m) {
    using namespace detail;
    std::vector<uint8_t> v;
    v.push_back('S'); v.push_back('Y'); v.push_back('N'); v.push_back('C');
    put16(v, m.version); put16(v, m.tileId);
    put16(v, m.gridCols); put16(v, m.gridRows);
    put16(v, m.tileCol); put16(v, m.tileRow);
    put32(v, m.wallW); put32(v, m.wallH); put32(v, m.tileW); put32(v, m.tileH);
    put32(v, m.fpsNum); put32(v, m.fpsDen);
    put64(v, m.globalFrameIndex); put64(v, m.originTimeNs);
    // ---- v3 timecode (6 bytes) — placed HERE, right after the core, NOT at the
    // tail: some NVENC drivers (582.16) truncate long escaped SEI NALs, and the
    // authored timecode must never fall in the truncated region. ----
    v.push_back(m.tcHours); v.push_back(m.tcMinutes); v.push_back(m.tcSeconds);
    v.push_back(m.tcFrames); v.push_back(m.tcDrop); v.push_back(m.tcValid);
    put64(v, m.createTimeNsUtc);
    v.insert(v.end(), m.colorSpace, m.colorSpace + 16);
    const float* cf = reinterpret_cast<const float*>(&m.cam);
    for (int i = 0; i < 62; ++i) putF32(v, cf[i]);
    return v;
}

inline bool deserialize(const uint8_t* p, size_t len, SyncMeta& m) {
    using namespace detail;
    // Only the core through originTimeNs is required; createTimeNsUtc,
    // colorSpace and the camera block are OPTIONAL tails (zero-filled when
    // absent). Rationale: some NVENC driver builds truncate long escaped SEI
    // NALs (observed: 582.16 emits 328 of 344 declared payload bytes), and
    // older senders emit the 56-byte v1 core — both must keep parsing.
    constexpr size_t kCore = 4 + 6 * 2 + 6 * 4 + 2 * 8;   // 56: through originTimeNs
    if (len < kCore) return false;
    if (p[0] != 'S' || p[1] != 'Y' || p[2] != 'N' || p[3] != 'C') return false;
    size_t o = 4;
    m.version = get16(p + o); o += 2; m.tileId = get16(p + o); o += 2;
    m.gridCols = get16(p + o); o += 2; m.gridRows = get16(p + o); o += 2;
    m.tileCol = get16(p + o); o += 2; m.tileRow = get16(p + o); o += 2;
    m.wallW = get32(p + o); o += 4; m.wallH = get32(p + o); o += 4;
    m.tileW = get32(p + o); o += 4; m.tileH = get32(p + o); o += 4;
    m.fpsNum = get32(p + o); o += 4; m.fpsDen = get32(p + o); o += 4;
    m.globalFrameIndex = get64(p + o); o += 8; m.originTimeNs = get64(p + o); o += 8;
    // ---- v3 timecode (right after the core). ONLY in version >= 3 streams:
    // a v2 sender (older nvenc_tile.lib / UE plugin builds) has createTimeNsUtc
    // at this offset — parsing tc unconditionally read its bytes as a garbage
    // timecode and shifted every later field. Honor the sender's version. ----
    if (m.version >= 3) {
        m.tcHours   = (len >= o + 1) ? p[o] : 0; o += 1;
        m.tcMinutes = (len >= o + 1) ? p[o] : 0; o += 1;
        m.tcSeconds = (len >= o + 1) ? p[o] : 0; o += 1;
        m.tcFrames  = (len >= o + 1) ? p[o] : 0; o += 1;
        m.tcDrop    = (len >= o + 1) ? p[o] : 0; o += 1;
        m.tcValid   = (len >= o + 1) ? p[o] : 0; o += 1;
    } else {
        m.tcHours = m.tcMinutes = m.tcSeconds = m.tcFrames = m.tcDrop = m.tcValid = 0;
    }
    m.createTimeNsUtc = (len >= o + 8) ? get64(p + o) : 0; o += 8;
    if (len >= o + 16) std::memcpy(m.colorSpace, p + o, 16); else std::memset(m.colorSpace, 0, 16); o += 16;
    float* cf = reinterpret_cast<float*>(&m.cam);
    for (int i = 0; i < 62; ++i) { cf[i] = (len >= o + 4) ? getF32(p + o) : 0.0f; o += 4; }
    return true;
}

// SEI user_data_unregistered payload = UUID(16) + serialized meta.
inline std::vector<uint8_t> buildSeiPayload(const SyncMeta& m) {
    std::vector<uint8_t> v(kUuid, kUuid + 16);
    std::vector<uint8_t> body = serialize(m);
    v.insert(v.end(), body.begin(), body.end());
    return v;
}
inline bool parseUserData(const uint8_t* p, size_t len, SyncMeta& m) {
    if (len < 16 || std::memcmp(p, kUuid, 16) != 0) return false;
    return deserialize(p + 16, len - 16, m);
}

// --- Annex-B SEI scanner (for verification / standalone parsing) -------------
// Strips emulation-prevention bytes from a NAL RBSP.
inline std::vector<uint8_t> deEmulate(const uint8_t* p, size_t len) {
    std::vector<uint8_t> out;
    out.reserve(len);
    int zeros = 0;
    for (size_t i = 0; i < len; ++i) {
        if (zeros >= 2 && p[i] == 0x03) { zeros = 0; continue; } // drop emulation byte
        out.push_back(p[i]);
        zeros = (p[i] == 0x00) ? zeros + 1 : 0;
    }
    return out;
}

// Parse all SEI messages in one de-emulated SEI RBSP (after the NAL header),
// invoking cb(SyncMeta) for each user_data_unregistered carrying our UUID.
template <class Cb>
inline void parseSeiRbsp(const uint8_t* p, size_t len, Cb cb) {
    size_t i = 0;
    while (i + 2 <= len) {
        uint32_t type = 0;
        while (i < len && p[i] == 0xFF) { type += 255; ++i; }
        if (i >= len) break; type += p[i++];
        uint32_t size = 0;
        while (i < len && p[i] == 0xFF) { size += 255; ++i; }
        if (i >= len) break; size += p[i++];
        // Clamp instead of rejecting: a truncated last SEI message (NVENC
        // driver quirk with long escaped payloads) still parses its core.
        if (i + size > len) size = (uint32_t)(len - i);
        if (type == 5) { SyncMeta m; if (parseUserData(p + i, size, m)) cb(m); }
        i += size;
        if (i < len && p[i] == 0x80) break; // rbsp_trailing
    }
}

// Scan an Annex-B H.264/HEVC stream and invoke cb for each embedded SyncMeta.
template <class Cb>
inline void scanAnnexB(const uint8_t* data, size_t len, bool hevc, Cb cb) {
    size_t i = 0;
    auto isStart = [&](size_t j, int& sc) {
        if (j + 3 <= len && data[j] == 0 && data[j + 1] == 0 && data[j + 2] == 1) { sc = 3; return true; }
        if (j + 4 <= len && data[j] == 0 && data[j + 1] == 0 && data[j + 2] == 0 && data[j + 3] == 1) { sc = 4; return true; }
        return false;
    };
    int sc = 0;
    while (i < len && !isStart(i, sc)) ++i;
    while (i < len) {
        i += sc;                 // skip this start code
        size_t nalStart = i;
        size_t j = i;
        int sc2 = 0;
        while (j < len && !isStart(j, sc2)) ++j;
        size_t nalLen = j - nalStart;
        if (nalLen >= 2) {
            const uint8_t* nal = data + nalStart;
            int type = hevc ? ((nal[0] >> 1) & 0x3F) : (nal[0] & 0x1F);
            bool isSei = hevc ? (type == 39 || type == 40) : (type == 6);
            if (isSei) {
                size_t hdr = hevc ? 2 : 1;
                auto rbsp = deEmulate(nal + hdr, nalLen - hdr);
                parseSeiRbsp(rbsp.data(), rbsp.size(), cb);
            }
        }
        sc = sc2;
        i = j;
    }
}

} // namespace sync
