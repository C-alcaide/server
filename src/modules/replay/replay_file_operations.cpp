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
 *
 * This module uses libvmx (https://github.com/openmediatransport/libvmx),
 * licensed under MIT, which is compatible with GPL-3.
 *
 * Derived from the CasparCG replay module
 * (https://github.com/krzyc/CasparCG-Server/tree/master/src/modules/replay).
 * Copyright (c) 2011 Sveriges Television AB <info@casparcg.com>
 * Copyright (c) 2013 Technical University of Lodz Multimedia Centre <office@cm.p.lodz.pl>
 * Authors: Jan Starzak <jan@ministryofgoodsteps.com>,
 *          Krzysztof Pyrkosz <pyrkosz@o2.pl>
 */

#include "replay_file_operations.h"
#include <cstdio>
#include <cwchar>

namespace caspar { namespace replay {

    replay_file_handle safe_fopen(const wchar_t* filename, uint32_t mode, uint32_t shareFlags)
    {
#ifdef REPLAY_IO_WINAPI
        replay_file_handle handle;
        // GENERIC_WRITE mapped to CREATE_ALWAYS, READ to OPEN_EXISTING
        DWORD creation = (mode & GENERIC_WRITE) ? CREATE_ALWAYS : OPEN_EXISTING;
        
        handle = CreateFileW(filename, mode, shareFlags, NULL, creation, 0, NULL);
        if (handle == INVALID_HANDLE_VALUE) return NULL;
        return handle;
#else
        // POSIX implementation simplified
        return NULL; 
#endif
    }

    void safe_fclose(replay_file_handle file_handle)
    {
#ifdef REPLAY_IO_WINAPI
        if(file_handle) CloseHandle(file_handle);
#else
        if(file_handle) fclose(file_handle);
#endif
    }

    void write_index_header(replay_file_handle outfile_idx, const core::video_format_desc* format_desc, boost::posix_time::ptime start_timecode, int audio_channels)
    {
        replay_file_header_pod header;
        memcpy(header.magick, "OMAV", 4);
        header.version = 2;
        header.width = format_desc->width;
        header.height = format_desc->height;
        header.fps = format_desc->fps;
        header.field_mode = 3; // Progressive
        
        // ptime serializing
        boost::posix_time::time_duration td = start_timecode - boost::posix_time::from_time_t(0);
        header.time_ticks = td.ticks();

        replay_file_header_ex header_ex;
        memcpy(header_ex.video_fourcc, "VMX ", 4);
        memcpy(header_ex.audio_fourcc, "in32", 4);
        header_ex.audio_channels = audio_channels;

        DWORD written;
#ifdef REPLAY_IO_WINAPI
        WriteFile(outfile_idx, &header, sizeof(header), &written, NULL);
        WriteFile(outfile_idx, &header_ex, sizeof(header_ex), &written, NULL);
#endif
    }

    void write_index(replay_file_handle outfile_idx, long long offset)
    {
        DWORD written;
#ifdef REPLAY_IO_WINAPI
        WriteFile(outfile_idx, &offset, sizeof(offset), &written, NULL);
#endif
    }

    long long read_index(replay_file_handle infile_idx)
    {
        long long offset = 0;
        DWORD read;
#ifdef REPLAY_IO_WINAPI
        if (ReadFile(infile_idx, &offset, sizeof(offset), &read, NULL) && read == sizeof(offset)) {
            return offset;
        }
#endif
        return -1;
    }

    long long length_index(replay_file_handle infile_idx)
    {
#ifdef REPLAY_IO_WINAPI
        LARGE_INTEGER size;
        if(GetFileSizeEx(infile_idx, &size)) {
             long long header_size = sizeof(replay_file_header_pod) + sizeof(replay_file_header_ex);
             if (size.QuadPart < header_size) return 0;
             return (size.QuadPart - header_size) / sizeof(long long);
        }
#endif
        return 0;
    }
    
    int seek_index(replay_file_handle infile_idx, long long frame, uint32_t origin)
    {
#ifdef REPLAY_IO_WINAPI
        long long header_size = sizeof(replay_file_header_pod) + sizeof(replay_file_header_ex);
        LARGE_INTEGER pos;
        pos.QuadPart = frame * sizeof(long long) + header_size; // Seek to frame offset
        
        // origin handling: typicaly we seek from BEGIN
        return SetFilePointerEx(infile_idx, pos, NULL, FILE_BEGIN) ? 0 : 1;
#endif
        return 1;
    }

    int read_index_header(replay_file_handle infile_idx, replay_file_header_pod* header)
    {
        DWORD read;
#ifdef REPLAY_IO_WINAPI
        SetFilePointer(infile_idx, 0, NULL, FILE_BEGIN);
        if(ReadFile(infile_idx, header, sizeof(replay_file_header_pod), &read, NULL) && read == sizeof(replay_file_header_pod))
             return 0;
#endif
        return 1;
    }

    int read_index_header_ex(replay_file_handle infile_idx, replay_file_header_ex* header)
    {
          DWORD read;
#ifdef REPLAY_IO_WINAPI
        SetFilePointer(infile_idx, sizeof(replay_file_header_pod), NULL, FILE_BEGIN);
        if(ReadFile(infile_idx, header, sizeof(replay_file_header_ex), &read, NULL) && read == sizeof(replay_file_header_ex))
             return 0;
#endif
        return 1;
    }

}}
