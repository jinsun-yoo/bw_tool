#pragma once
#include <atomic>
#include <cstdint>
#include "ring_buffer.h"

struct Sample {
    uint64_t timestamp_ns;
    uint64_t xmit_data;
    uint64_t rcv_data;
    uint64_t xmit_pkts;
    uint64_t rcv_pkts;
};

// 4096 entries ≈ 4s of backlog at 1ms sampling rate
using SampleBuffer = RingBuffer<Sample, 4096>;

void sampler_thread(SampleBuffer* buf, std::atomic<bool>* stop);
