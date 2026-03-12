// test_rice_entropy.cpp
// CPU mirror of the ProRes hybrid Rice/Exp-Golomb VLC (cuda_prores_rice.cuh).
//
// ProRes uses a leading-ZEROS unary prefix (not leading ones).  The codeword
// for an unsigned value v with codebook byte cb is:
//
//   Rice path   (v < sw<<ro):  q zeros, one stop-1, ro remainder bits
//   Exp-Golomb  (v >= sw<<ro): nz zeros, then (exp+1)-bit value v'
//
// where sw=(cb&3)+1, ro=cb>>5, eo=(cb>>2)&7, nz=floor(log2(v'))-eo+sw,
// v' = v - sw*2^ro + 2^eo.
//
// Tests validate vlc_count, vlc_encode, and the round-trip decoder against
// hand-computed golden codewords, then run randomised round-trip checks for
// every ProRes codebook constant from FFmpeg proresdata.c.
//
// No GPU is required; the test compiles and runs as a pure C++ executable.
//
// Usage: test_rice_entropy.exe [--verbose]
//
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <stdexcept>
#include <string>
#include <vector>
#include <random>
#include <algorithm>

static bool g_verbose = false;

// ============================================================================
// MSB-first bit writer — matches the GPU bit packer in cuda_prores_rice.cuh
// ============================================================================

struct BitPackerCPU {
    std::vector<uint8_t> buf;
    uint64_t accum      = 0;
    int      accum_bits = 0;

    void flush_full_bytes() {
        while (accum_bits >= 8) {
            accum_bits -= 8;
            buf.push_back((uint8_t)(accum >> accum_bits));
            accum &= (1ULL << accum_bits) - 1;
        }
    }

    void put(uint32_t val, int n) {
        if (n == 0) return;
        accum       = (accum << n) | (uint64_t)(val & ((1u << n) - 1));
        accum_bits += n;
        flush_full_bytes();
    }

    void pad_byte() {
        if (accum_bits > 0)
            put(0u, 8 - accum_bits);
    }
};

// ============================================================================
// ProRes hybrid Rice/Exp-Golomb VLC — mirrors vlc_count / vlc_encode in
// cuda_prores_rice.cuh, which in turn mirrors encode_vlc_codeword() from
// FFmpeg libavcodec/proresenc_kostya.c.
//
// Codebook byte layout (from Apple spec / FFmpeg):
//   bits[7:5]  = rice_order  (ro)
//   bits[4:2]  = exp_order   (eo)
//   bits[1:0]  = switch_bits raw value (sw_raw); encoder uses sw = sw_raw + 1
//
// Rice path   (val < (sw_raw+1) << ro):
//   q  = val >> ro
//   emit: q leading ZEROS, stop-ONE, ro remainder bits
//   total bits: q + 1 + ro
//
// Exp-Golomb  (val >= (sw_raw+1) << ro):
//   v  = val - sw*2^ro + 2^eo       (shifted into [2^eo, ...))
//   exp = floor(log2(v))
//   nz  = exp - eo + sw             (leading zeros count)
//   emit: nz zeros, then (exp+1)-bit value v MSB-first
//   total bits: nz + exp + 1 = 2*exp - eo + sw + 1
// ============================================================================

// Signed → unsigned ProRes zigzag (MAKE_CODE in encode_dcs/encode_acs)
// 0→0  1→2  -1→1  2→4  -2→3  ...
static unsigned make_code(int val) {
    return (unsigned)(val * 2) ^ (unsigned)(val >> 31);
}

// Count bits for one ProRes VLC codeword (matches vlc_count in rice.cuh)
static unsigned vlc_count_cpu(unsigned codebook, unsigned val)
{
    unsigned sw = (codebook & 3u) + 1u;
    unsigned ro = codebook >> 5;
    unsigned eo = (codebook >> 2) & 7u;
    unsigned sv = sw << ro;
    if (val >= sv) {
        unsigned v = val - sv + (1u << eo);
        unsigned exp = 0, t = v;
        while (t > 1u) { exp++; t >>= 1; }
        return exp * 2u - eo + sw + 1u;
    }
    return (val >> ro) + ro + 1u;
}

// Encode one ProRes VLC codeword (matches vlc_encode in rice.cuh)
static void vlc_encode_cpu(BitPackerCPU &bp, unsigned codebook, unsigned val)
{
    unsigned sw = (codebook & 3u) + 1u;
    unsigned ro = codebook >> 5;
    unsigned eo = (codebook >> 2) & 7u;
    unsigned sv = sw << ro;
    if (val >= sv) {
        unsigned v = val - sv + (1u << eo);
        unsigned exp = 0, t = v;
        while (t > 1u) { exp++; t >>= 1; }
        // Write nz = exp - eo + sw leading zeros
        unsigned nz = exp - eo + sw;
        for (unsigned n = nz; n > 0u; ) {
            int chunk = (int)(n >= 30u ? 30u : n);
            bp.put(0u, chunk);
            n -= (unsigned)chunk;
        }
        // Write (exp+1) bits of v MSB-first
        unsigned nb = exp + 1u;
        if (nb > 30u) { bp.put((v >> (nb - 30u)) & 0x3FFFFFFFu, 30); nb -= 30u; }
        bp.put(v & ((1u << nb) - 1u), (int)nb);
    } else {
        // Rice: q zeros, stop 1, ro remainder bits
        unsigned q = val >> ro;
        for (unsigned n = q; n > 0u; ) {
            int chunk = (int)(n >= 30u ? 30u : n);
            bp.put(0u, chunk);
            n -= (unsigned)chunk;
        }
        bp.put(1u, 1);
        if (ro) bp.put(val & ((1u << ro) - 1u), (int)ro);
    }
}

// Decoder mirror of DECODE_CODEWORD macro in FFmpeg proresdec.c.
// NOTE: the decoder uses (codebook & 3) for switch_bits — no +1.
static unsigned vlc_decode_cpu(const std::vector<uint8_t> &data,
                                int &bit_pos, unsigned codebook)
{
    unsigned sw  = codebook & 3u;   // decoder switch_bits (no +1)
    unsigned ro  = codebook >> 5;
    unsigned eo  = (codebook >> 2) & 7u;

    auto read_bit = [&]() -> unsigned {
        int byte_idx = bit_pos / 8;
        int bit_idx  = 7 - (bit_pos % 8);
        bit_pos++;
        if (byte_idx >= (int)data.size()) return 1u; // safe padding
        return (unsigned)((data[byte_idx] >> bit_idx) & 1u);
    };

    // Count leading zeros (q), then consume the stop-1
    unsigned q = 0u;
    while (read_bit() == 0u) q++;

    if (q > sw) {
        // Exp-Golomb: after the q zeros + stop-1, read (eo + q - sw - 1) more bits.
        // Reconstruct: v = (1 << more) | [more bits]; then undo the shift.
        unsigned more = eo + q - sw - 1u;
        unsigned v    = 1u << more;
        for (unsigned b = more; b > 0u; b--)
            v |= (read_bit() << (b - 1u));
        return v - (1u << eo) + ((sw + 1u) << ro);
    }
    // Rice: read ro remainder bits
    unsigned r = 0u;
    for (unsigned b = ro; b > 0u; b--)
        r = (r << 1u) | read_bit();
    return (q << ro) | r;
}

// ============================================================================
// ProRes codebook constants (from FFmpeg libavcodec/proresdata.c)
// ============================================================================

static const uint8_t FIRST_DC_CB = 0xB8u;

static const uint8_t c_dc_codebook[7] = {
    0x04, 0x28, 0x28, 0x4D, 0x4D, 0x70, 0x70
};

static const uint8_t c_run_to_cb[16] = {
    0x06, 0x06, 0x05, 0x05, 0x04, 0x29, 0x29, 0x29,
    0x29, 0x4C, 0x4C, 0x4C, 0x4C, 0x4C, 0x4C, 0x4C
};

static const uint8_t c_level_to_cb[10] = {
    0x04, 0x0A, 0x05, 0x06, 0x04, 0x28, 0x28, 0x28, 0x28, 0x4C
};

// ============================================================================
// Test helpers
// ============================================================================

static int g_pass = 0, g_fail = 0;

static void check(bool cond, const char *label) {
    if (cond) {
        if (g_verbose) printf("  PASS  %s\n", label);
        g_pass++;
    } else {
        printf("  FAIL  %s\n", label);
        g_fail++;
    }
}

// ============================================================================
// Tests
// ============================================================================

static void test_make_code()
{
    // Zigzag mapping: 0→0, 1→2, -1→1, 2→4, -2→3, ...
    check(make_code(0)    == 0u,  "make_code(0)=0");
    check(make_code(1)    == 2u,  "make_code(1)=2");
    check(make_code(-1)   == 1u,  "make_code(-1)=1");
    check(make_code(2)    == 4u,  "make_code(2)=4");
    check(make_code(-2)   == 3u,  "make_code(-2)=3");
    check(make_code(100)  == 200u,"make_code(100)=200");
    // Positive and negative adjacent values must not collide
    check(make_code(1) != make_code(-1), "make_code sign separation");
}

static void test_first_dc_cb_parameters()
{
    // FIRST_DC_CB = 0xB8 = 1011 1000
    //   bits[7:5] = 101 = 5  → rice_order (ro)
    //   bits[4:2] = 110 = 6  → exp_order  (eo)
    //   bits[1:0] = 00  = 0  → switch_bits raw; encoder sw = 0+1 = 1
    // Switch value sv = sw << ro = 1 << 5 = 32
    unsigned sw = (FIRST_DC_CB & 3u) + 1u;
    unsigned ro = FIRST_DC_CB >> 5;
    unsigned eo = (FIRST_DC_CB >> 2) & 7u;
    unsigned sv = sw << ro;
    check(ro == 5u, "FIRST_DC_CB: rice_order = 5");
    check(eo == 6u, "FIRST_DC_CB: exp_order  = 6");
    check(sw == 1u, "FIRST_DC_CB: sw_enc     = 1");
    check(sv == 32u,"FIRST_DC_CB: switch_val = 32");
    // Rice/Exp-Golomb boundary: val=31 Rice, val=32 Exp-Golomb
    check(vlc_count_cpu(FIRST_DC_CB, 31u) < vlc_count_cpu(FIRST_DC_CB, 32u),
          "FIRST_DC_CB: exp-Golomb costs more bits than Rice at boundary");
}

static void test_vlc_count_rice_path()
{
    // FIRST_DC_CB, ro=5: count = (val>>5) + 5 + 1 = q + 6
    // All val in [0,31] have q=0 → 6 bits each.
    check(vlc_count_cpu(FIRST_DC_CB,  0u) == 6u, "rice val=0  → 6 bits");
    check(vlc_count_cpu(FIRST_DC_CB,  1u) == 6u, "rice val=1  → 6 bits");
    check(vlc_count_cpu(FIRST_DC_CB, 31u) == 6u, "rice val=31 → 6 bits");

    // c_dc_codebook[0] = 0x04: ro=0, eo=1, sw=1, sv=1
    //   val=0: Rice q=0, count = 0 + 0 + 1 = 1 bit
    check(vlc_count_cpu(0x04u, 0u) == 1u, "0x04 val=0 → 1 bit");
    // val=1: Exp-Golomb (v=1-1+2=2, exp=1, count=2*1-1+1+1=3)
    check(vlc_count_cpu(0x04u, 1u) == 3u, "0x04 val=1 → 3 bits");
    // val=3: v=3-1+2=4, exp=2, count=2*2-1+1+1=5
    check(vlc_count_cpu(0x04u, 3u) == 5u, "0x04 val=3 → 5 bits");
}

static void test_vlc_count_exp_golomb_path()
{
    // FIRST_DC_CB, eo=6, sw=1: count = 2*exp(v) - 6 + 1 + 1 = 2*exp(v) - 4
    // val=32: v=32-32+64=64=2^6,  exp=6, count=12-4=8
    check(vlc_count_cpu(FIRST_DC_CB, 32u)  ==  8u, "expg val=32  →  8 bits");
    // val=63: v=63-32+64=95,       exp=6, count=8
    check(vlc_count_cpu(FIRST_DC_CB, 63u)  ==  8u, "expg val=63  →  8 bits");
    // val=127: v=127-32+64=159,    exp=7, count=14-4=10
    check(vlc_count_cpu(FIRST_DC_CB, 127u) == 10u, "expg val=127 → 10 bits");
    // val=255: v=255-32+64=287,    exp=8, count=16-4=12
    check(vlc_count_cpu(FIRST_DC_CB, 255u) == 12u, "expg val=255 → 12 bits");
}

static void test_vlc_known_codeword_rice()
{
    // val=0, FIRST_DC_CB (ro=5, q=0):
    //   emit stop-1, then 5 zero remainder bits → "1 00000" (6 bits)
    //   pad to byte → "10000000" = 0x80
    {
        BitPackerCPU bp;
        vlc_encode_cpu(bp, FIRST_DC_CB, 0u);
        bp.pad_byte();
        check(bp.buf.size() == 1,                        "rice val=0: 1 byte");
        check(!bp.buf.empty() && bp.buf[0] == 0x80u,     "rice val=0: byte=0x80");
    }
    // val=1 (q=0, remainder=1): "1 00001" → pad → "10000100" = 0x84
    {
        BitPackerCPU bp;
        vlc_encode_cpu(bp, FIRST_DC_CB, 1u);
        bp.pad_byte();
        check(!bp.buf.empty() && bp.buf[0] == 0x84u,     "rice val=1: byte=0x84");
    }
    // val=31 (q=0, remainder=31=0b11111): "1 11111" → pad → "11111100" = 0xFC
    {
        BitPackerCPU bp;
        vlc_encode_cpu(bp, FIRST_DC_CB, 31u);
        bp.pad_byte();
        check(!bp.buf.empty() && bp.buf[0] == 0xFCu,     "rice val=31: byte=0xFC");
    }
}

static void test_vlc_known_codeword_exp_golomb()
{
    // val=32: v=64=0b1000000, exp=6, nz=1
    //   "0" + 7 bits of 64 → "01000000" = 0x40  (exactly 8 bits)
    {
        BitPackerCPU bp;
        vlc_encode_cpu(bp, FIRST_DC_CB, 32u);
        bp.pad_byte();
        check(bp.buf.size() == 1,                        "expg val=32: 1 byte");
        check(!bp.buf.empty() && bp.buf[0] == 0x40u,     "expg val=32: byte=0x40");
    }
    // val=63: v=95=0b1011111, exp=6, nz=1
    //   "0" + 7 bits of 95 → "01011111" = 0x5F
    {
        BitPackerCPU bp;
        vlc_encode_cpu(bp, FIRST_DC_CB, 63u);
        bp.pad_byte();
        check(!bp.buf.empty() && bp.buf[0] == 0x5Fu,     "expg val=63: byte=0x5F");
    }
    // val=127: v=159=0b10011111, exp=7, nz=7-6+1=2
    //   "00" + 8 bits of 159=0b10011111 → 10 bits: "0010011111"
    //   pad to 16 bits → "0010011111000000"
    //   byte0=0b00100111=0x27, byte1=0b11000000=0xC0
    {
        BitPackerCPU bp;
        vlc_encode_cpu(bp, FIRST_DC_CB, 127u);
        bp.pad_byte();
        bool ok = bp.buf.size() == 2 && bp.buf[0] == 0x27u && bp.buf[1] == 0xC0u;
        check(bp.buf.size() == 2, "expg val=127: 2 bytes");
        check(ok,                 "expg val=127: bytes=0x27,0xC0");
    }
}

static void test_count_equals_encode_length()
{
    // vlc_count must equal the actual number of bits written by vlc_encode,
    // across all ProRes codebook constants and a broad set of coefficient values.
    static const uint8_t codebooks[] = {
        FIRST_DC_CB,
        c_dc_codebook[0], c_dc_codebook[1], c_dc_codebook[3], c_dc_codebook[5],
        c_run_to_cb[0],   c_run_to_cb[4],   c_run_to_cb[8],   c_run_to_cb[15],
        c_level_to_cb[0], c_level_to_cb[5], c_level_to_cb[9]
    };
    static const unsigned vals[] = {
        0, 1, 2, 3, 5, 15, 16, 31, 32, 63, 64, 100, 127, 200, 255, 512, 1000
    };
    bool all_ok = true;
    for (unsigned cb : codebooks) {
        for (unsigned v : vals) {
            unsigned count = vlc_count_cpu(cb, v);
            BitPackerCPU bp;
            vlc_encode_cpu(bp, cb, v);
            unsigned actual = (unsigned)(bp.buf.size() * 8u) + (unsigned)bp.accum_bits;
            if (count != actual) {
                printf("  FAIL  count_eq_encode cb=0x%02X val=%u: count=%u actual=%u\n",
                       cb, v, count, actual);
                all_ok = false;
            }
        }
    }
    check(all_ok, "count_equals_encode: all codebooks x all test values");
}

static void test_vlc_round_trip()
{
    // Encode a batch of random unsigned values with a fixed codebook, then
    // decode — must recover every original value exactly.  Covers both Rice
    // and Exp-Golomb paths for each codebook.
    std::mt19937 rng(0xDEADBEEF);
    std::uniform_int_distribution<unsigned> val_dist(0, 300);

    static const uint8_t codebooks[] = {
        FIRST_DC_CB,
        c_dc_codebook[0], c_dc_codebook[2], c_dc_codebook[4], c_dc_codebook[6],
        c_run_to_cb[0],   c_run_to_cb[7],   c_run_to_cb[15],
        c_level_to_cb[0], c_level_to_cb[9]
    };
    bool all_ok = true;
    for (unsigned cb : codebooks) {
        std::vector<unsigned> orig(64);
        for (auto &v : orig) v = val_dist(rng);

        BitPackerCPU bp;
        unsigned total_bits = 0;
        for (unsigned v : orig) {
            total_bits += vlc_count_cpu(cb, v);
            vlc_encode_cpu(bp, cb, v);
        }
        bp.pad_byte();

        int bit_pos = 0;
        for (unsigned i = 0; i < (unsigned)orig.size(); i++) {
            unsigned dec = vlc_decode_cpu(bp.buf, bit_pos, cb);
            if (dec != orig[i]) {
                printf("  FAIL  vlc_round_trip cb=0x%02X i=%u: enc=%u dec=%u\n",
                       cb, i, orig[i], dec);
                all_ok = false;
                break;
            }
        }
        if (bit_pos != (int)total_bits && all_ok) {
            printf("  NOTE  vlc_round_trip cb=0x%02X: consumed %d bits, expected %u\n",
                   cb, bit_pos, total_bits);
        }
    }
    check(all_ok, "vlc_round_trip: all codebooks recover original unsigned values");
}

static void test_round_trip_make_code()
{
    // Signed coefficients: make_code → vlc_encode → vlc_decode → unmake_code
    // must recover the original signed value exactly.
    auto unmake_code = [](unsigned u) -> int {
        if (u & 1u) return -(int)((u + 1u) / 2u);
        return (int)(u / 2u);
    };

    std::mt19937 rng(0xCAFEBABEu);
    std::uniform_int_distribution<int> dist(-500, 500);
    std::vector<int> orig(128);
    for (auto &v : orig) v = dist(rng);

    BitPackerCPU bp;
    for (int v : orig)
        vlc_encode_cpu(bp, FIRST_DC_CB, make_code(v));
    bp.pad_byte();

    int bit_pos = 0;
    bool all_ok = true;
    for (int i = 0; i < (int)orig.size(); i++) {
        unsigned u   = vlc_decode_cpu(bp.buf, bit_pos, FIRST_DC_CB);
        int      dec = unmake_code(u);
        if (dec != orig[i]) {
            printf("  FAIL  round_trip_make_code i=%d: orig=%d dec=%d\n",
                   i, orig[i], dec);
            all_ok = false;
            break;
        }
    }
    check(all_ok, "round_trip_make_code: signed coefficients survive encode→decode");
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char *argv[])
{
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--verbose") == 0) g_verbose = true;

    printf("test_rice_entropy — CPU reference ProRes VLC validation\n");
    printf("══════════════════════════════════════════════════════════\n");

    test_make_code();
    test_first_dc_cb_parameters();
    test_vlc_count_rice_path();
    test_vlc_count_exp_golomb_path();
    test_vlc_known_codeword_rice();
    test_vlc_known_codeword_exp_golomb();
    test_count_equals_encode_length();
    test_vlc_round_trip();
    test_round_trip_make_code();

    printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
