/*
 * Copyright (c) 2025 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CasparCG is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CasparCG. If not, see <http://www.gnu.org/licenses/>.
 */

// hap_frame_parser.h
// Parse HAP section headers and decompress Snappy-compressed DXT/BC payloads.
//
// HAP frame structure:
//   Section header: 4 bytes (short form) or 8 bytes (long form)
//     - Short: [length_lo:8][length_mid:8][length_hi:8][section_type:8]
//       length = low 24 bits.  If length == 0, use long form.
//     - Long:  [0:8][0:8][0:8][section_type:8][length:32 LE]
//
//   Section types (high nibble = compressor, low nibble = texture format):
//     Compressor:  0x0 = none, 0xB = Snappy, 0xC = chunked
//     Texture fmt: 0x0B = RGB DXT1, 0x0E = RGBA DXT5, 0x0F = YCoCg DXT5,
//                  0x0C = Alpha-only DXT5 (used in HapM second section)
//                  0x01 = BC7 (HAP R)
//     Complex:     0x0D = multi-texture container (HAP Q Alpha / HapM)
//
// Reference: https://github.com/Vidvox/hap/blob/master/documentation/HapVideoDRAFT.md
// ---------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>

#include <snappy.h>

namespace caspar { namespace hap {

// HAP texture format identifiers (low nibble of section type byte).
// Values match kHapFormat* constants in the Vidvox reference library (hap.c).
enum class HapTextureFormat : uint8_t {
    RGB_DXT1       = 0x0B, // GL_COMPRESSED_RGB_S3TC_DXT1_EXT
    RGBA_DXT5      = 0x0E, // GL_COMPRESSED_RGBA_S3TC_DXT5_EXT
    YCoCg_DXT5     = 0x0F, // Scaled YCoCg in DXT5 (HAP Q)
    A_RGTC1        = 0x01, // GL_COMPRESSED_RED_RGTC1 / BC4 (HAP Q Alpha second texture)
    RGBA_BC7       = 0x0C, // GL_COMPRESSED_RGBA_BPTC_UNORM (HAP R)
};

// HAP compressor identifiers (high nibble of section type byte).
// Values match kHapCompressor* constants in the Vidvox reference library (hap.c).
enum class HapCompressor : uint8_t {
    None    = 0x0A,
    Snappy  = 0x0B,
    Complex = 0x0C, // chunked Snappy (decode instructions inside)
};

// High-level HAP variant.
enum class HapVariant {
    Hap,         // Hap1 — RGB DXT1
    HapAlpha,    // Hap5 — RGBA DXT5
    HapQ,        // HapY — Scaled YCoCg DXT5
    HapQAlpha,   // HapM — YCoCg DXT5 + Alpha DXT5
    HapR,        // HapR — BC7
    Unknown,
};

// Parsed result of a single HAP texture section.
struct HapSection {
    HapTextureFormat texture_format = HapTextureFormat::RGB_DXT1;
    const uint8_t*   data           = nullptr; // points into source buffer (if uncompressed/Snappy)
    size_t           data_size      = 0;
    bool             needs_decompress = false;
    HapCompressor    compressor     = HapCompressor::None;
};

// Result of fully parsing (and decompressing) a HAP frame.
struct HapFrameResult {
    HapVariant                variant = HapVariant::Unknown;
    std::vector<uint8_t>      texture_data;       // decompressed DXT data for primary texture
    HapTextureFormat          texture_format = HapTextureFormat::RGB_DXT1;
    std::vector<uint8_t>      alpha_data;          // decompressed alpha DXT data (HapM only)
    HapTextureFormat          alpha_format  = HapTextureFormat::RGBA_DXT5;
};

// ---------------------------------------------------------------------------
// Section header parser
// ---------------------------------------------------------------------------

inline bool parse_section_header(const uint8_t* data, size_t size,
                                 uint8_t& section_type_out,
                                 uint32_t& payload_length_out,
                                 int& header_size_out)
{
    if (size < 4) return false;

    uint32_t length_24 = (uint32_t)data[0]
                       | ((uint32_t)data[1] << 8)
                       | ((uint32_t)data[2] << 16);
    section_type_out = data[3];

    if (length_24 == 0) {
        // Long form: 4-byte header + 4-byte length
        if (size < 8) return false;
        payload_length_out = (uint32_t)data[4]
                           | ((uint32_t)data[5] << 8)
                           | ((uint32_t)data[6] << 16)
                           | ((uint32_t)data[7] << 24);
        header_size_out = 8;
    } else {
        payload_length_out = length_24;
        header_size_out = 4;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Snappy decompression for a single section
// ---------------------------------------------------------------------------

inline bool decompress_section(const uint8_t* compressed, size_t compressed_size,
                               std::vector<uint8_t>& output)
{
    size_t uncompressed_len = 0;
    if (!snappy::GetUncompressedLength(reinterpret_cast<const char*>(compressed),
                                       compressed_size, &uncompressed_len)) {
        return false;
    }
    output.resize(uncompressed_len);
    return snappy::RawUncompress(reinterpret_cast<const char*>(compressed),
                                compressed_size,
                                reinterpret_cast<char*>(output.data()));
}

// ---------------------------------------------------------------------------
// Chunked (complex) Snappy decompression
//
// Per the Vidvox reference (hap.c), the complex section payload contains:
//   [decode_instructions_container (section type 0x01)]
//     Contains sub-sections in any order:
//       [compressor_table (type 0x02): N bytes]  — one byte per chunk
//       [chunk_size_table (type 0x03): N * 4 bytes LE]
//       [chunk_offset_table (type 0x04): N * 4 bytes LE]  — OPTIONAL
//   [chunk_data]
//     Frame data (concatenated compressed chunks) immediately after container
// ---------------------------------------------------------------------------

inline bool decompress_chunked(const uint8_t* data, size_t size,
                               std::vector<uint8_t>& output)
{
    const uint8_t* end = data + size;

    uint8_t  sec_type    = 0;
    uint32_t sec_len     = 0;
    int      sec_hdr_sz  = 0;

    // Read the decode instructions container (must be section type 0x01)
    if (!parse_section_header(data, size, sec_type, sec_len, sec_hdr_sz))
        return false;
    if (sec_type != 0x01) return false; // not a decode instructions container
    if ((size_t)sec_hdr_sz + sec_len > size)
        return false;

    const uint8_t* container_data = data + sec_hdr_sz;
    const size_t   container_len  = sec_len;

    // Frame data follows immediately after the container
    const uint8_t* chunk_base = data + sec_hdr_sz + sec_len;

    // Parse sub-sections inside the container (any order, per spec)
    const uint8_t* compressor_table = nullptr;
    const uint8_t* size_table       = nullptr;
    const uint8_t* offset_table     = nullptr;
    int            num_chunks       = 0;

    const uint8_t* p = container_data;
    size_t remaining = container_len;
    while (remaining > 0) {
        if (!parse_section_header(p, remaining, sec_type, sec_len, sec_hdr_sz))
            return false;
        const uint8_t* sec_data = p + sec_hdr_sz;

        switch (sec_type) {
        case 0x02: // Chunk Second-Stage Compressor Table
            compressor_table = sec_data;
            if (num_chunks == 0) num_chunks = (int)sec_len;
            else if (num_chunks != (int)sec_len) return false;
            break;
        case 0x03: // Chunk Size Table
            size_table = sec_data;
            if (num_chunks == 0) num_chunks = (int)(sec_len / 4);
            else if (num_chunks != (int)(sec_len / 4)) return false;
            break;
        case 0x04: // Chunk Offset Table (optional)
            offset_table = sec_data;
            break;
        default:
            break; // ignore unknown sub-sections
        }

        size_t consumed = (size_t)sec_hdr_sz + sec_len;
        if (consumed > remaining) return false;
        p += consumed;
        remaining -= consumed;
    }

    // Compressor table and size table are required
    if (!compressor_table || !size_table || num_chunks <= 0)
        return false;

    // First pass: compute total uncompressed size
    size_t total_uncompressed = 0;
    std::vector<size_t> chunk_uncomp_sizes(num_chunks);
    size_t running_offset = 0;
    for (int i = 0; i < num_chunks; i++) {
        uint32_t c_size = 0;
        std::memcpy(&c_size, size_table + i * 4, 4);

        uint32_t c_offset;
        if (offset_table) {
            std::memcpy(&c_offset, offset_table + i * 4, 4);
        } else {
            c_offset = (uint32_t)running_offset;
        }
        running_offset += c_size;

        const uint8_t* chunk_ptr = chunk_base + c_offset;
        if (chunk_ptr + c_size > end) return false;

        size_t uncomp_len = 0;
        if (compressor_table[i] == 0x0B) { // Snappy
            if (!snappy::GetUncompressedLength(reinterpret_cast<const char*>(chunk_ptr),
                                               c_size, &uncomp_len))
                return false;
        } else {
            uncomp_len = c_size; // uncompressed chunk (0x0A = None)
        }
        chunk_uncomp_sizes[i] = uncomp_len;
        total_uncompressed += uncomp_len;
    }

    output.resize(total_uncompressed);

    // Second pass: decompress each chunk
    size_t out_offset = 0;
    running_offset = 0;
    for (int i = 0; i < num_chunks; i++) {
        uint32_t c_size = 0;
        std::memcpy(&c_size, size_table + i * 4, 4);

        uint32_t c_offset;
        if (offset_table) {
            std::memcpy(&c_offset, offset_table + i * 4, 4);
        } else {
            c_offset = (uint32_t)running_offset;
        }
        running_offset += c_size;

        const uint8_t* chunk_ptr = chunk_base + c_offset;

        if (compressor_table[i] == 0x0B) { // Snappy
            if (!snappy::RawUncompress(reinterpret_cast<const char*>(chunk_ptr),
                                       c_size,
                                       reinterpret_cast<char*>(output.data() + out_offset)))
                return false;
        } else {
            std::memcpy(output.data() + out_offset, chunk_ptr, c_size);
        }
        out_offset += chunk_uncomp_sizes[i];
    }

    return true;
}

// ---------------------------------------------------------------------------
// Decode a single texture section (not a multi-texture container).
// Returns the raw DXT block data in `output`.
// ---------------------------------------------------------------------------

inline bool decode_texture_section(const uint8_t* data, size_t size,
                                   HapTextureFormat& format_out,
                                   std::vector<uint8_t>& output)
{
    uint8_t  sec_type   = 0;
    uint32_t sec_len    = 0;
    int      sec_hdr_sz = 0;
    if (!parse_section_header(data, size, sec_type, sec_len, sec_hdr_sz))
        return false;
    if ((size_t)sec_hdr_sz + sec_len > size)
        return false;

    // Extract compressor from the high nibble, texture format from the low nibble.
    // Section type byte = (compressor << 4) | texture_format
    uint8_t compressor_id  = (sec_type >> 4) & 0x0F;
    uint8_t texture_fmt_id = sec_type & 0x0F;

    // Map the low nibble to a HapTextureFormat.
    switch (texture_fmt_id) {
    case 0x0B: format_out = HapTextureFormat::RGB_DXT1;   break;
    case 0x0E: format_out = HapTextureFormat::RGBA_DXT5;  break;
    case 0x0F: format_out = HapTextureFormat::YCoCg_DXT5; break;
    case 0x01: format_out = HapTextureFormat::A_RGTC1;    break; // BC4 alpha (HAP Q Alpha)
    case 0x0C: format_out = HapTextureFormat::RGBA_BC7;   break; // BC7 (HAP R)
    default:
        return false;
    }

    const uint8_t* payload = data + sec_hdr_sz;
    const size_t   payload_size = sec_len;

    switch (static_cast<HapCompressor>(compressor_id)) {
    case HapCompressor::None:
        output.assign(payload, payload + payload_size);
        return true;
    case HapCompressor::Snappy:
        return decompress_section(payload, payload_size, output);
    case HapCompressor::Complex:
        return decompress_chunked(payload, payload_size, output);
    default:
        return false;
    }
}

// ---------------------------------------------------------------------------
// Parse a complete HAP frame (may be single-texture or multi-texture).
// ---------------------------------------------------------------------------

inline bool parse_hap_frame(const uint8_t* data, size_t size, HapFrameResult& result)
{
    if (size < 4) return false;

    uint8_t  sec_type   = 0;
    uint32_t sec_len    = 0;
    int      sec_hdr_sz = 0;
    if (!parse_section_header(data, size, sec_type, sec_len, sec_hdr_sz))
        return false;

    if (sec_type == 0x0D) {
        // Multi-texture container (HAP Q Alpha / HapM).
        // Contains two sub-sections: YCoCg DXT5 (color) + Alpha DXT5.
        if ((size_t)sec_hdr_sz + sec_len > size)
            return false;
        const uint8_t* inner     = data + sec_hdr_sz;
        const size_t   inner_len = sec_len;

        // Parse first sub-section (color)
        uint8_t  sub_type   = 0;
        uint32_t sub_len    = 0;
        int      sub_hdr_sz = 0;
        if (!parse_section_header(inner, inner_len, sub_type, sub_len, sub_hdr_sz))
            return false;
        if ((size_t)sub_hdr_sz + sub_len > inner_len)
            return false;

        if (!decode_texture_section(inner, sub_hdr_sz + sub_len,
                                    result.texture_format, result.texture_data))
            return false;

        // Parse second sub-section (alpha)
        const uint8_t* second      = inner + sub_hdr_sz + sub_len;
        const size_t   second_left = inner_len - (sub_hdr_sz + sub_len);

        if (!decode_texture_section(second, second_left,
                                    result.alpha_format, result.alpha_data))
            return false;

        result.variant = HapVariant::HapQAlpha;
        return true;
    }

    // Single-texture section
    if (!decode_texture_section(data, size, result.texture_format, result.texture_data))
        return false;

    switch (result.texture_format) {
    case HapTextureFormat::RGB_DXT1:   result.variant = HapVariant::Hap;      break;
    case HapTextureFormat::RGBA_DXT5:  result.variant = HapVariant::HapAlpha; break;
    case HapTextureFormat::YCoCg_DXT5: result.variant = HapVariant::HapQ;     break;
    case HapTextureFormat::RGBA_BC7:   result.variant = HapVariant::HapR;     break;
    default:                           result.variant = HapVariant::Unknown;   break;
    }
    return true;
}

}} // namespace caspar::hap
