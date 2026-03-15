/*
* Copyright (c) 2026 Open Media Transport Contributors
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

#pragma once

#include <stdint.h>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <core/video_format.h>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
// Use WinAPI for file IO
#define VMX_IO_WINAPI
#endif

#ifndef VMX_IO_WINAPI

#include "common/utf.h"

#define _FILE_OFFSET_BITS  64

#ifdef _WIN32
#ifndef fopen64
#define fopen64 fopen
#endif
#ifndef fseek64
#define fseek64 _fseeki64
#endif
#ifndef ftell64
#define ftell64 _ftelli64
#endif
#endif

#ifdef __x86_64
#ifndef fseek64
#define fseek64 fseek
#endif
#ifndef ftell64
#define ftell64 ftell
#endif
#endif

#ifndef FILE_CURRENT
#define FILE_CURRENT SEEK_CUR
#endif
#ifndef FILE_BEGIN
#define FILE_BEGIN SEEK_SET
#endif
#ifndef GENERIC_READ
#define GENERIC_READ 0x80000000
#endif
#ifndef GENERIC_WRITE
#define GENERIC_WRITE 0x40000000
#endif
#ifndef FILE_SHARE_READ
#define FILE_SHARE_READ 0x00000001
#endif
#ifndef FILE_SHARE_WRITE
#define FILE_SHARE_WRITE 0x00000002
#endif
#endif

#ifdef VMX_IO_WINAPI
typedef HANDLE                          vmx_file_handle;
#define VMX_INVALID_HANDLE              INVALID_HANDLE_VALUE
#define VMX_IO_WINAPI_FUNC(x)           ::x
#else
typedef FILE*                           vmx_file_handle;
#define VMX_INVALID_HANDLE              NULL
#define VMX_IO_WINAPI_FUNC(x)           x
#endif

namespace caspar { namespace vmx {

    struct vmx_file_header {
        char                            magick[4]; // = 'OMAV'
        uint8_t                         version; // = 1 
        uint32_t                        width;
        uint32_t                        height;
        double                          fps;
        uint8_t                         field_mode; // 
        // boost::posix_time::ptime is not POD, careful with serialization. Replay uses raw bytes?
    };

    // simplified structure for POD serialization
    struct vmx_file_header_pod {
        char                            magick[4]; // = 'OMAV'
        uint8_t                         version; // = 1
        uint32_t                        width;
        uint32_t                        height;
        double                          fps;
        uint8_t                         field_mode; 
        int64_t                         time_ticks; // storage for ptime
    };

    struct vmx_file_header_ex {
        char                            video_fourcc[4]; // = 'VMX '
        char                            audio_fourcc[4]; // = 'in32'

        int                             audio_channels;
    };

    vmx_file_handle safe_fopen(const wchar_t* filename, uint32_t mode, uint32_t shareFlags);
    void safe_fclose(vmx_file_handle file_handle);
    
    void write_index_header(vmx_file_handle outfile_idx, const core::video_format_desc* format_desc, boost::posix_time::ptime start_timecode, int audio_channels);
    void write_index(vmx_file_handle outfile_idx, long long offset);
    
    // Reads from file
    long long read_index(vmx_file_handle infile_idx);
    long long length_index(vmx_file_handle infile_idx);
    int seek_index(vmx_file_handle infile_idx, long long frame, uint32_t origin);
    
    int read_index_header(vmx_file_handle infile_idx, vmx_file_header_pod* header);
    int read_index_header_ex(vmx_file_handle infile_idx, vmx_file_header_ex* header);

}}
