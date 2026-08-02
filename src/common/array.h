#pragma once

#include <any>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

namespace caspar {

template <typename T>
class array final
{
    template <typename>
    friend class array;

  public:
    using iterator       = T*;
    using const_iterator = const T*;

    array() = default;

    /// Allocates `size` ELEMENTS, zero-filled.
    ///
    /// size_ is an element count everywhere else in this class -- size() returns it
    /// and end() is ptr_ + size_ -- but this constructor used to malloc/memset the
    /// argument as if it were a byte count. For T wider than a byte that made the
    /// two conventions disagree, and either reading of the argument was then wrong:
    /// pass bytes and the array reports sizeof(T)x more elements than it holds (which
    /// is what AudioResampler did, and audio_mixer duly read past the allocation);
    /// pass elements and the allocation is sizeof(T)x too small.
    ///
    /// Scaling by sizeof(T) here makes the argument mean elements consistently. It is
    /// a no-op for every existing caller: after the AudioResampler fix they are all
    /// byte-typed (device::create_array in both accelerators returns array<uint8_t>,
    /// and av_producer's zero-length image plane is array<const uint8_t>).
    explicit array(std::size_t size)
        : size_(size)
    {
        if (size_ > 0) {
            const std::size_t bytes = size_ * sizeof(T);
            auto              storage = std::shared_ptr<void>(std::malloc(bytes), std::free);
            if (!storage)
                throw std::bad_alloc();
            ptr_ = reinterpret_cast<T*>(storage.get());
            std::memset(ptr_, 0, bytes);
            storage_ = std::make_shared<std::any>(std::move(storage));
        }
    }

    array(std::vector<T> other)
    {
        auto storage = std::make_shared<std::vector<T>>(std::move(other));
        ptr_         = storage->data();
        size_        = storage->size();
        storage_     = std::make_shared<std::any>(std::move(storage));
    }

    template <typename S>
    explicit array(T* ptr, std::size_t size, S&& storage)
        : ptr_(ptr)
        , size_(size)
        , storage_(std::make_shared<std::any>(std::forward<S>(storage)))
    {
    }

    array(const array<T>&) = delete;

    array(array&& other)
        : ptr_(other.ptr_)
        , size_(other.size_)
        , storage_(std::move(other.storage_))
    {
        other.ptr_  = nullptr;
        other.size_ = 0;
    }

    array& operator=(const array<T>&) = delete;

    array& operator=(array&& other)
    {
        ptr_     = std::move(other.ptr_);
        size_    = std::move(other.size_);
        storage_ = std::move(other.storage_);

        return *this;
    }

    /// A second handle on the same memory, sharing the same owner.
    ///
    /// The copy constructor is deleted so that a writable buffer has exactly one
    /// owner, and that is still the rule -- this is a named exception for the one
    /// case that needs it: a pixel_format_desc plane that deliberately aliases an
    /// earlier plane's bytes, where the same buffer is handed to the GPU twice under
    /// two different interpretations (UYVY exposes one row of bytes as both a
    /// full-rate luma view and a half-rate chroma view). Writing through either
    /// handle writes the one buffer; there is no copy-on-write.
    array alias() const
    {
        array a;
        a.ptr_     = ptr_;
        a.size_    = size_;
        a.storage_ = storage_;
        return a;
    }

    T*          begin() const { return ptr_; }
    T*          data() const { return ptr_; }
    T*          end() const { return ptr_ + size_; }
    std::size_t size() const { return size_; }

    explicit operator bool() const { return size_ > 0; };

    template <typename S>
    S* storage() const
    {
        return std::any_cast<S>(storage_.get());
    }

  private:
    T*                        ptr_  = nullptr;
    std::size_t               size_ = 0;
    std::shared_ptr<std::any> storage_;
};

template <typename T>
class array<const T> final
{
  public:
    using iterator       = const T*;
    using const_iterator = const T*;

    array() = default;

    array(std::size_t size)
        : size_(size)
    {
        if (size_ > 0) {
            auto storage = std::shared_ptr<void>(std::malloc(size), std::free);
            std::memset(storage.get(), 0, size);
            ptr_         = reinterpret_cast<T*>(storage.get());
            storage_ = std::make_shared<std::any>(storage);
        }
    }

    array(const std::vector<T>& other)
    {
        auto storage = std::make_shared<std::vector<T>>(std::move(other));
        ptr_         = storage->data();
        size_        = storage->size();
        storage_     = std::make_shared<std::any>(std::move(storage));
    }

    template <typename S>
    explicit array(const T* ptr, std::size_t size, S&& storage)
        : ptr_(ptr)
        , size_(size)
        , storage_(std::make_shared<std::any>(std::forward<S>(storage)))
    {
    }

    array(const array& other)
        : ptr_(other.ptr_)
        , size_(other.size_)
        , storage_(other.storage_)
    {
    }

    array(array<T>&& other)
        : ptr_(other.ptr_)
        , size_(other.size_)
        , storage_(other.storage_)
    {
        other.ptr_     = nullptr;
        other.size_    = 0;
        other.storage_ = nullptr;
    }

    array& operator=(const array& other)
    {
        ptr_     = other.ptr_;
        size_    = other.size_;
        storage_ = other.storage_;
        return *this;
    }

    const T*    begin() const { return ptr_; }
    const T*    data() const { return ptr_; }
    const T*    end() const { return ptr_ + size_; }
    std::size_t size() const { return size_; }

    explicit operator bool() const { return size_ > 0; }

    template <typename S>
    S* storage() const
    {
        return std::any_cast<S>(storage_.get());
    }

  private:
    const T*                  ptr_  = nullptr;
    std::size_t               size_ = 0;
    std::shared_ptr<std::any> storage_;
};

} // namespace caspar

namespace std {

template <typename T>
void swap(caspar::array<const T>& lhs, caspar::array<const T>& rhs)
{
    lhs.swap(rhs);
}

} // namespace std
