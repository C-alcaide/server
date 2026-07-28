/*
 * Copyright (c) 2026 CasparCG Contributors
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify it under the terms of the GNU
 * General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 */

#include "isf_image_load.h"

#include <modules/image/util/image_converter.h>
#include <modules/image/util/image_loader.h>

#include <cstring>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
}

namespace caspar { namespace isf {

bool load_rgba_image(const std::wstring& path, std::vector<unsigned char>& rgba, int& width, int& height)
{
    try {
        auto frame = image::load_image(path);
        frame      = image::convert_image_frame(frame, AV_PIX_FMT_RGBA);
        width      = frame->width;
        height     = frame->height;
        if (width <= 0 || height <= 0 || !frame->data[0])
            return false;

        const int dst_stride = width * 4;
        const int src_stride = frame->linesize[0];
        rgba.resize(static_cast<std::size_t>(dst_stride) * height);
        for (int y = 0; y < height; ++y)
            std::memcpy(rgba.data() + static_cast<std::size_t>(y) * dst_stride,
                        frame->data[0] + static_cast<std::size_t>(y) * src_stride,
                        static_cast<std::size_t>(dst_stride));
        return true;
    } catch (...) {
        return false;
    }
}

}} // namespace caspar::isf
