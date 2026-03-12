// async_file_writer.h
// High-throughput unbuffered async file writer using IOCP and a pre-allocated
// sector-aligned write buffer ring.
//
// Design goals
// ─────────────────────────────────────────────────────────────────────────────
//  • FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_WRITE_THROUGH
//    bypasses the OS page cache for direct NVMe/RAID throughput.
//  • K=8 sector-aligned pinned write buffers — each WriteFile call is backed by
//    one buffer slot.  When all K slots are in-flight, write() blocks until one
//    completes (backpressure rather than unbounded queue growth).
//  • Caller must ensure each write() size fits in one slot (i.e. ≤ slot_bytes).
//    For ProRes HQ 4K: ≈ 14 MB/frame; a 16 MB slot is safe.
//  • close() drains in-flight writes, appends any caller-supplied tail (e.g. moov
//    atom from MovMuxer) via a separate buffered handle, then closes the file.
//
// Thread safety: single-writer thread only.
#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

class AsyncFileWriter {
public:
    static constexpr int kRingSize = 8; // max simultaneous in-flight writes

    // slot_bytes must be a multiple of sector_size and large enough for the
    // largest single write() call (typ. one encoded ProRes frame ≈ 14 MB).
    // A value of 16 MiB covers HQ 4K comfortably.
    static constexpr size_t kDefaultSlotBytes   = 16 * 1024 * 1024; // 16 MB
    static constexpr size_t kDefaultSectorSize  = 4096;

    AsyncFileWriter() = default;
    ~AsyncFileWriter();

    // Open output file.  sector_size should be the physical sector size of the
    // target volume (query with GetDiskFreeSpace or IOCTL_STORAGE_QUERY_PROPERTY).
    bool open(const wchar_t *path,
              size_t slot_bytes  = kDefaultSlotBytes,
              size_t sector_size = kDefaultSectorSize);

    // Write `size` bytes from `data` to the file at the current position.
    // If all ring slots are busy, blocks until one completes.
    // Returns false on I/O error.
    bool write(const void *data, size_t size);

    // Drain all in-flight writes.  Must be called before close() or before
    // accessing current_offset().
    bool flush();

    // Returns the current logical write offset (= total bytes submitted, NOT
    // yet necessarily flushed).  Useful for building chunk offset tables.
    uint64_t current_offset() const { return write_offset_; }

    // Drain in-flight writes, optionally append a tail buffer (e.g. moov atom)
    // via a separate buffered handle so size need not be sector-aligned, then close.
    bool close(const void *tail = nullptr, size_t tail_size = 0);

private:
    struct Slot {
        uint8_t  *buf       = nullptr; // VirtualAlloc'd, sector-aligned
        OVERLAPPED ov       = {};
        bool       in_flight = false;
        size_t     size      = 0;      // bytes for the current write (≤ slot_bytes)
    };

    bool wait_for_slot(int idx);    // wait for in-flight write on slot idx to complete
    bool drain_all();               // wait for ALL in-flight slots
    int  acquire_free_slot();       // returns slot index, blocks if needed

    HANDLE   file_      = INVALID_HANDLE_VALUE;
    HANDLE   iocp_      = nullptr;
    size_t   slot_bytes_   = 0;
    size_t   sector_size_  = 0;
    uint64_t write_offset_ = 0;

    Slot     slots_[kRingSize] = {};
    int      next_slot_        = 0;       // round-robin slot selector
    int      in_flight_count_  = 0;

    std::wstring path_; // stored for tail write in close()
};
