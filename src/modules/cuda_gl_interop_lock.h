/*
 * Copyright (c) 2025 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com)
 * and is licensed under the GNU General Public License v3.
 */

// cuda_gl_interop_lock.h
//
// Process-wide mutex that serialises ALL cudaGraphicsGLRegisterImage and
// cudaGraphicsUnregisterResource calls across every CUDA producer (ProRes,
// NotchLC, etc.).
//
// WHY THIS IS NEEDED
// ------------------
// When one producer (e.g. NotchLC) is being destroyed while another (e.g.
// ProRes) is initialising, their read_threads can concurrently call
// cudaGraphicsUnregisterResource and cudaGraphicsGLRegisterImage respectively.
// The NVIDIA driver's CUDA-GL interop layer is not thread-safe across these
// operations, even for distinct GL textures, causing a STATUS_ACCESS_VIOLATION
// (0xC0000005) crash inside nvoglv64.dll.
//
// The mutex is only held for the brief registration/unregistration burst at
// the start and end of each read_thread -- never during normal frame
// decode/map/unmap -- so there is zero performance impact at steady state.
//
// USAGE
// -----
// #include "../../cuda_gl_interop_lock.h"   (adjust relative path as needed)
//
// In read_loop() startup:
//   {
//       std::lock_guard<std::mutex> gl_lk(caspar::cuda_gl_interop_mutex());
//       for (int i = 0; i < num_slots_; i++)
//           cgt_[i] = std::make_shared<CudaGLTexture>(gl_tex_[i]);
//   }
//
// In read_loop() cleanup:
//   {
//       std::lock_guard<std::mutex> gl_lk(caspar::cuda_gl_interop_mutex());
//       for (int i = 0; i < num_slots_; i++) cgt_[i].reset();
//   }

#pragma once

#include <mutex>

namespace caspar {

// Returns the one process-wide mutex.  Using an inline function with a
// function-local static guarantees exactly one instance across all translation
// units that include this header (C++17 §9.4.2, guaranteed for inline
// functions with internal linkage via static local).
inline std::mutex& cuda_gl_interop_mutex()
{
    static std::mutex mtx;
    return mtx;
}

} // namespace caspar
