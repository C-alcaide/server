// async_file_writer.cpp
#include "async_file_writer.h"

#include <cassert>
#include <cstring>
#include <cstdio>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
AsyncFileWriter::~AsyncFileWriter()
{
    if (file_ != INVALID_HANDLE_VALUE) {
        drain_all();
        if (iocp_) { CloseHandle(iocp_); iocp_ = nullptr; }
        CloseHandle(file_); file_ = INVALID_HANDLE_VALUE;
    }
    for (int i = 0; i < kRingSize; i++) {
        if (slots_[i].buf) {
            VirtualFree(slots_[i].buf, 0, MEM_RELEASE);
            slots_[i].buf = nullptr;
        }
    }
}

// ---------------------------------------------------------------------------
// open()
// ---------------------------------------------------------------------------
bool AsyncFileWriter::open(const wchar_t *path,
                           size_t         slot_bytes,
                           size_t         sector_size)
{
    assert(!path_[0]); // must not be open already
    slot_bytes_  = slot_bytes  ? slot_bytes  : kDefaultSlotBytes;
    sector_size_ = sector_size ? sector_size : kDefaultSectorSize;

    // Round slot_bytes up to sector boundary
    if (slot_bytes_ % sector_size_)
        slot_bytes_ = ((slot_bytes_ / sector_size_) + 1) * sector_size_;

    file_ = CreateFileW(
        path,
        GENERIC_WRITE,
        0, nullptr,
        CREATE_ALWAYS,
        FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_WRITE_THROUGH,
        nullptr);

    if (file_ == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "[AsyncFileWriter] CreateFile failed: error %lu\n", GetLastError());
        return false;
    }

    iocp_ = CreateIoCompletionPort(file_, nullptr, reinterpret_cast<ULONG_PTR>(this), 0);
    if (!iocp_) {
        CloseHandle(file_); file_ = INVALID_HANDLE_VALUE;
        fprintf(stderr, "[AsyncFileWriter] CreateIoCompletionPort failed: error %lu\n", GetLastError());
        return false;
    }

    // Allocate write buffer ring using VirtualAlloc (sector-aligned, committed)
    for (int i = 0; i < kRingSize; i++) {
        slots_[i].buf = static_cast<uint8_t*>(
            VirtualAlloc(nullptr, slot_bytes_, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
        if (!slots_[i].buf) {
            fprintf(stderr, "[AsyncFileWriter] VirtualAlloc(%zu) failed for slot %d\n", slot_bytes_, i);
            // Cleanup already allocated
            for (int j = 0; j < i; j++) {
                VirtualFree(slots_[j].buf, 0, MEM_RELEASE);
                slots_[j].buf = nullptr;
            }
            CloseHandle(iocp_); iocp_ = nullptr;
            CloseHandle(file_); file_ = INVALID_HANDLE_VALUE;
            return false;
        }
        slots_[i].in_flight = false;
    }

    path_ = path;
    write_offset_    = 0;
    next_slot_       = 0;
    in_flight_count_ = 0;
    return true;
}

// ---------------------------------------------------------------------------
// wait_for_slot() — wait for a specific in-flight slot to complete
// ---------------------------------------------------------------------------
bool AsyncFileWriter::wait_for_slot(int idx)
{
    if (!slots_[idx].in_flight) return true;

    DWORD      bytes_transferred = 0;
    ULONG_PTR  key               = 0;
    OVERLAPPED *pov              = nullptr;

    // We wait until we dequeue a completion that belongs to slot[idx]
    while (slots_[idx].in_flight) {
        if (!GetQueuedCompletionStatus(iocp_, &bytes_transferred, &key, &pov, INFINITE)) {
            fprintf(stderr, "[AsyncFileWriter] IOCP wait failed: error %lu\n", GetLastError());
            return false;
        }
        // Identify which slot completed by pointer arithmetic on the OVERLAPPED
        for (int i = 0; i < kRingSize; i++) {
            if (pov == &slots_[i].ov) {
                slots_[i].in_flight = false;
                --in_flight_count_;
                break;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// drain_all() — wait for all in-flight writes
// ---------------------------------------------------------------------------
bool AsyncFileWriter::drain_all()
{
    bool ok = true;
    while (in_flight_count_ > 0) {
        DWORD      bytes_transferred = 0;
        ULONG_PTR  key               = 0;
        OVERLAPPED *pov              = nullptr;

        if (!GetQueuedCompletionStatus(iocp_, &bytes_transferred, &key, &pov, INFINITE)) {
            fprintf(stderr, "[AsyncFileWriter] drain_all IOCP error: %lu\n", GetLastError());
            ok = false;
            // Keep draining to avoid leaving slots in_flight = true
        }

        for (int i = 0; i < kRingSize; i++) {
            if (pov == &slots_[i].ov && slots_[i].in_flight) {
                slots_[i].in_flight = false;
                --in_flight_count_;
                break;
            }
        }
    }
    return ok;
}

// ---------------------------------------------------------------------------
// acquire_free_slot() — returns index of a free slot, blocking if all busy
// ---------------------------------------------------------------------------
int AsyncFileWriter::acquire_free_slot()
{
    // Fast path: check round-robin next slot
    for (int n = 0; n < kRingSize; n++) {
        int i = (next_slot_ + n) % kRingSize;
        if (!slots_[i].in_flight) {
            next_slot_ = (i + 1) % kRingSize;
            return i;
        }
    }

    // All slots busy: wait for the oldest in-flight write (next_slot_)
    int wait_idx = next_slot_ % kRingSize;
    if (!wait_for_slot(wait_idx)) return -1;
    next_slot_ = (wait_idx + 1) % kRingSize;
    return wait_idx;
}

// ---------------------------------------------------------------------------
// write()
// ---------------------------------------------------------------------------
bool AsyncFileWriter::write(const void *data, size_t size)
{
    if (!size) return true;
    if (size > slot_bytes_) {
        fprintf(stderr, "[AsyncFileWriter] write(%zu) exceeds slot size (%zu)\n",
                size, slot_bytes_);
        return false;
    }

    int idx = acquire_free_slot();
    if (idx < 0) return false;

    Slot &s = slots_[idx];

    // Copy data into sector-aligned buffer, zero-pad the last sector
    size_t aligned = (size + sector_size_ - 1) & ~(sector_size_ - 1);
    memcpy(s.buf, data, size);
    if (aligned > size)
        memset(s.buf + size, 0, aligned - size);

    // Submit async write
    memset(&s.ov, 0, sizeof(OVERLAPPED));
    s.ov.Offset     = static_cast<DWORD>( write_offset_        & 0xFFFFFFFFULL);
    s.ov.OffsetHigh = static_cast<DWORD>((write_offset_ >> 32) & 0xFFFFFFFFULL);

    s.in_flight = true;
    ++in_flight_count_;
    write_offset_ += aligned;

    BOOL ok = WriteFile(file_, s.buf, static_cast<DWORD>(aligned), nullptr, &s.ov);
    if (!ok) {
        DWORD err = GetLastError();
        if (err != ERROR_IO_PENDING) {
            s.in_flight = false;
            --in_flight_count_;
            fprintf(stderr, "[AsyncFileWriter] WriteFile failed: error %lu\n", err);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// flush()
// ---------------------------------------------------------------------------
bool AsyncFileWriter::flush()
{
    return drain_all();
}

// ---------------------------------------------------------------------------
// close()
// ---------------------------------------------------------------------------
bool AsyncFileWriter::close(const void *tail, size_t tail_size)
{
    bool ok = drain_all();

    if (iocp_)  { CloseHandle(iocp_);  iocp_  = nullptr; }
    if (file_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_); file_ = INVALID_HANDLE_VALUE;
    }

    // Append tail (e.g. moov atom) via a regular buffered handle so it doesn't
    // need to be sector-aligned.
    if (tail && tail_size > 0 && !path_.empty()) {
        HANDLE bf = CreateFileW(
            path_.c_str(),
            GENERIC_WRITE, 0, nullptr, OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL, nullptr);
        if (bf != INVALID_HANDLE_VALUE) {
            // Seek to logical end of last sector-aligned write
            LARGE_INTEGER li;
            li.QuadPart = static_cast<LONGLONG>(write_offset_);
            SetFilePointerEx(bf, li, nullptr, FILE_BEGIN);
            DWORD written = 0;
            if (!WriteFile(bf, tail, static_cast<DWORD>(tail_size), &written, nullptr))
                ok = false;
            CloseHandle(bf);
        } else {
            fprintf(stderr, "[AsyncFileWriter] close: could not reopen for tail write (error %lu)\n",
                    GetLastError());
            ok = false;
        }
    }

    // Free slot buffers
    for (int i = 0; i < kRingSize; i++) {
        if (slots_[i].buf) {
            VirtualFree(slots_[i].buf, 0, MEM_RELEASE);
            slots_[i].buf = nullptr;
        }
    }

    path_.clear();
    return ok;
}
