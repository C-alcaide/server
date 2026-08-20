/*
 * Copyright (c) 2011 Sveriges Television AB <info@casparcg.com>
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
 *
 * Author: Robert Nagy, ronag89@gmail.com
 */

#include "StdAfx.h"

#include "ffmpeg.h"

#include "consumer/ffmpeg_consumer.h"
#include "producer/ffmpeg_producer.h"

#include <common/log.h>

#include <core/module_dependencies.h>

#include <mutex>

extern "C" {
#include <libavdevice/avdevice.h>
#include <libavfilter/avfilter.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
}

namespace caspar { namespace ffmpeg {
static void sanitize(uint8_t* line)
{
    while (*line != 0u) {
        if (*line < 0x08 || (*line > 0x0D && *line < 0x20))
            *line = '?';
        line++;
    }
}

void log_callback(void* ptr, int level, const char* fmt, va_list vl)
{
    static thread_local bool print_prefix_tss = true;

    char     line[1024];
    AVClass* avc = ptr != nullptr ? *static_cast<AVClass**>(ptr) : nullptr;
    if (level > AV_LOG_DEBUG)
        return;
    line[0] = 0;

    // `AVClass::item_name` IS ALLOWED TO BE NULL, and calling it anyway is a jump to
    // address 0. FFmpeg's own formatter says so plainly -- `libavutil/log.c`:
    //
    //     return (cls->item_name ? cls->item_name : av_default_item_name)(obj);
    //
    // so a class that omits it is not malformed, it is opting into the default. Nearly every
    // class in FFmpeg sets `av_default_item_name` explicitly, which is why this went
    // unnoticed for years; `FFVulkanContext`'s class (`libavutil/vulkan.c`) does not:
    //
    //     static const AVClass vulkan_context_class = {
    //         .class_name = "vk",
    //         .version    = LIBAVUTIL_VERSION_INT,
    //         .parent_log_context_offset = offsetof(FFVulkanContext, log_parent),
    //     };
    //
    // So the FIRST log line FFmpeg emits from a Vulkan compute decoder killed the decode
    // thread. Measured 2026-08-21: `prores_vulkan` reported "Vulkan decoder initialization
    // successful" -- logged against the AVCodecContext, whose class is fine -- and then took
    // an access violation at address 0 on the next line, 13,675 times in five seconds. It
    // read as an FFmpeg or driver fault for hours, because everything about it pointed at
    // Vulkan: it survived FFmpeg creating its own device, every extension and feature the
    // decoder wanted, the queue families, the frame pool and the thread count, and the same
    // decode of the same file worked in ffmpeg.exe -- which uses FFmpeg's own null-safe
    // callback. Attributing it took a vectored exception handler to catch the first-chance
    // fault (/EHa turns it into a C++ exception that `catch (...)` swallows) plus DbgHelp to
    // resolve the return address, which named this function.
    const auto item_name_of = [](AVClass* cls, void* obj) -> const char* {
        return (cls->item_name != nullptr ? cls->item_name : av_default_item_name)(obj);
    };

#undef fprintf
    if (print_prefix_tss && (avc != nullptr)) {
        if (avc->parent_log_context_offset != 0) {
            AVClass** parent =
                *reinterpret_cast<AVClass***>(static_cast<uint8_t*>(ptr) + avc->parent_log_context_offset);
            if ((parent != nullptr) && (*parent != nullptr))
                std::snprintf(line, sizeof(line), "[%s @ %p] ", item_name_of(*parent, parent), parent);
        }
        std::snprintf(
            line + strlen(line), sizeof(line) - strlen(line), "[%s @ %p] ", item_name_of(avc, ptr), ptr);
    }

    std::vsnprintf(line + strlen(line), sizeof(line) - strlen(line), fmt, vl);

    print_prefix_tss = (strlen(line) != 0u) && line[strlen(line) - 1] == '\n';

    sanitize(reinterpret_cast<uint8_t*>(line));

    if (strstr(line, "Assuming an incorrectly") != nullptr) {
        return;
    }

    try {
        if (level == AV_LOG_VERBOSE)
            CASPAR_LOG(trace) << L"[ffmpeg] " << line;
        else if (level == AV_LOG_DEBUG)
            CASPAR_LOG(trace) << L"[ffmpeg] " << line;
        else if (level == AV_LOG_INFO)
            CASPAR_LOG(info) << L"[ffmpeg] " << line;
        else if (level == AV_LOG_WARNING)
            CASPAR_LOG(warning) << L"[ffmpeg] " << line;
        else if (level == AV_LOG_ERROR)
            CASPAR_LOG(error) << L"[ffmpeg] " << line;
        else if (level == AV_LOG_FATAL)
            CASPAR_LOG(fatal) << L"[ffmpeg] " << line;
        else
            CASPAR_LOG(trace) << L"[ffmpeg] " << line;
    } catch (...) {
    }
}

void log_for_thread(void* ptr, int level, const char* fmt, va_list vl) { log_callback(ptr, level, fmt, vl); }

void init(const core::module_dependencies& dependencies)
{
    av_log_set_callback(log_for_thread);

    avformat_network_init();
    avdevice_register_all();

    dependencies.consumer_registry->register_consumer_factory(L"FFmpeg Consumer", create_consumer);
    dependencies.consumer_registry->register_preconfigured_consumer_factory(L"ffmpeg", create_preconfigured_consumer);

    dependencies.producer_registry->register_producer_factory(L"FFmpeg Producer", create_producer);
}

void uninit()
{
    // avfilter_uninit();
    avformat_network_deinit();
}
}} // namespace caspar::ffmpeg
