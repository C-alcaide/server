#pragma once
#include <cstdint>

namespace caspar { namespace vmx {

    // Version 2 index entry including timestamp
    struct vmx_index_entry_v2 {
        int64_t file_offset;    
        uint64_t timestamp; 
    };

    // Header for each segment file to identify it and its range
    struct vmx_segment_header {
        uint32_t segment_index;
        uint64_t start_timestamp;
        uint64_t end_timestamp;
        uint32_t width;
        uint32_t height;
        double   fps;
        uint32_t segment_duration_sec; // Was reserved
    };

}}
