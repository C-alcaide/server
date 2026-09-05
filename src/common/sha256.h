/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

// SHA-256, FIPS 180-4, vendored.
//
// WHY VENDORED. The tree links no crypto library, on either platform: OpenSSL is absent,
// and reaching for Windows' CNG or Linux's libcrypto would put a platform #ifdef and a new
// dependency into `common` for one hash used in one handshake. Boost has no SHA-256 (its
// `uuid::detail::sha1` is SHA-1 and is not a public interface).
//
// WHAT THIS IS AND IS NOT FOR. A challenge/response over a socket, so that a password is
// not sent in the clear on a plain HTTP connection. It is NOT a password store: there is no
// key stretching here, and the config holds the password in plain text, so an attacker with
// the config file has the password whatever this does. Against an attacker on the wire it
// costs a replay window; against an attacker on the disk it costs nothing, and saying so is
// the point of this paragraph.
//
// The constants are the standard's: the first 32 bits of the fractional parts of the square
// roots of the first eight primes (H), and of the cube roots of the first sixty-four (K).
// They are not copied from another implementation -- a wrong one fails the test vectors
// below, which are the standard's own.
//
//   sha256("")    = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
//   sha256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad

#include <array>
#include <cstdint>
#include <cstring>
#include <string>

namespace caspar {

class sha256
{
  public:
    sha256() { reset(); }

    void reset()
    {
        state_ = {0x6a09e667u,
                  0xbb67ae85u,
                  0x3c6ef372u,
                  0xa54ff53au,
                  0x510e527fu,
                  0x9b05688cu,
                  0x1f83d9abu,
                  0x5be0cd19u};
        bitlen_ = 0;
        buflen_ = 0;
    }

    void update(const void* data, std::size_t len)
    {
        const auto* p = static_cast<const std::uint8_t*>(data);
        for (std::size_t i = 0; i < len; ++i) {
            buf_[buflen_++] = p[i];
            if (buflen_ == 64) {
                transform(buf_.data());
                bitlen_ += 512;
                buflen_ = 0;
            }
        }
    }

    void update(const std::string& s) { update(s.data(), s.size()); }

    std::array<std::uint8_t, 32> digest()
    {
        std::array<std::uint8_t, 32> out{};
        auto                         state = state_;
        auto                         buf   = buf_;
        auto                         i     = buflen_;
        auto                         bits  = bitlen_ + static_cast<std::uint64_t>(buflen_) * 8;

        // Pad: one 1 bit, zeros, then the length as 64 big-endian bits. Done on a COPY of
        // the state so `digest()` can be called without ending the object's life -- the
        // handshake hashes the same salt twice and a finalising digest() made that a trap.
        buf[i++] = 0x80;
        if (i > 56) {
            while (i < 64)
                buf[i++] = 0;
            transform_into(state, buf.data());
            i = 0;
        }
        while (i < 56)
            buf[i++] = 0;
        for (int b = 7; b >= 0; --b)
            buf[i++] = static_cast<std::uint8_t>((bits >> (b * 8)) & 0xff);
        transform_into(state, buf.data());

        for (int w = 0; w < 8; ++w)
            for (int b = 0; b < 4; ++b)
                out[static_cast<std::size_t>(w * 4 + b)] =
                    static_cast<std::uint8_t>((state[static_cast<std::size_t>(w)] >> (24 - b * 8)) & 0xff);
        return out;
    }

    static std::string hex(const std::array<std::uint8_t, 32>& d)
    {
        static const char* digits = "0123456789abcdef";
        std::string        out;
        out.reserve(64);
        for (auto b : d) {
            out.push_back(digits[b >> 4]);
            out.push_back(digits[b & 0x0f]);
        }
        return out;
    }

    /// The whole of it, for the common case.
    static std::string hex_of(const std::string& s)
    {
        sha256 h;
        h.update(s);
        return hex(h.digest());
    }

  private:
    static std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void transform(const std::uint8_t* chunk) { transform_into(state_, chunk); }

    static void transform_into(std::array<std::uint32_t, 8>& state, const std::uint8_t* chunk)
    {
        static const std::uint32_t K[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (static_cast<std::uint32_t>(chunk[i * 4]) << 24) |
                   (static_cast<std::uint32_t>(chunk[i * 4 + 1]) << 16) |
                   (static_cast<std::uint32_t>(chunk[i * 4 + 2]) << 8) |
                   static_cast<std::uint32_t>(chunk[i * 4 + 3]);
        for (int i = 16; i < 64; ++i) {
            const auto s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const auto s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i]          = w[i - 16] + s0 + w[i - 7] + s1;
        }

        auto a = state[0], b = state[1], c = state[2], d = state[3];
        auto e = state[4], f = state[5], g = state[6], h = state[7];

        for (int i = 0; i < 64; ++i) {
            const auto S1    = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const auto ch    = (e & f) ^ (~e & g);
            const auto temp1 = h + S1 + ch + K[i] + w[i];
            const auto S0    = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const auto maj   = (a & b) ^ (a & c) ^ (b & c);
            const auto temp2 = S0 + maj;

            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }

    std::array<std::uint32_t, 8> state_{};
    std::array<std::uint8_t, 64> buf_{};
    std::uint64_t                bitlen_ = 0;
    std::size_t                  buflen_ = 0;
};

} // namespace caspar
