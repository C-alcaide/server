// prores_producer.h
// CasparCG frame_producer for CUDA-accelerated ProRes decode.
//
// AMCP command:
//   PLAY 1-10 CUDA_PRORES [FILE <path>] [DEVICE <cuda_device_index>]
//
// This producer:
//  - Demuxes raw icpf packets from MOV/MXF/MKV using libavformat (no avcodec).
//  - Decodes each frame entirely on the GPU via CUDA kernels.
//  - Writes the decoded BGRA16 output directly to an OpenGL texture via
//    CUDA-GL interop (zero host transfers for pixel data).
//  - Returns a core::const_frame with the GL texture pre-populated, so the
//    OGL image mixer can composite it without a PBO upload path.
// ---------------------------------------------------------------------------
#pragma once

#include <common/memory.h>
#include <core/fwd.h>
#include <core/module_dependencies.h>
#include <core/producer/frame_producer.h>

#include <string>
#include <vector>

namespace caspar { namespace cuda_prores {

// Factory function — registered with the producer registry at module init.
spl::shared_ptr<core::frame_producer>
create_prores_producer(const core::frame_producer_dependencies& dependencies,
                       const std::vector<std::wstring>&         params);

// Module-level registration helper (called from cuda_prores.cpp::init()).
void register_prores_producer(const core::module_dependencies& module_deps);

}} // namespace caspar::cuda_prores
