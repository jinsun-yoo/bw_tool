#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>

// Lock-free single-producer single-consumer ring buffer.
// SIZE must be a power of 2.
template <typename T, size_t SIZE>
class RingBuffer {
    static_assert((SIZE & (SIZE - 1)) == 0, "SIZE must be a power of 2");

public:
    RingBuffer() : head_(0), tail_(0) {}

    // Called by producer. Returns false if buffer is full (sample dropped).
    bool push(const T& item) {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t next = (head + 1) & MASK;
        if (next == tail_.load(std::memory_order_acquire))
            return false; // full
        data_[head] = item;
        head_.store(next, std::memory_order_release);
        return true;
    }

    // Called by consumer. Returns false if buffer is empty.
    bool pop(T& item) {
        const size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire))
            return false; // empty
        item = data_[tail];
        tail_.store((tail + 1) & MASK, std::memory_order_release);
        return true;
    }

private:
    static constexpr size_t MASK = SIZE - 1;
    T data_[SIZE];
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
};
