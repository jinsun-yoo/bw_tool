#include "sampler.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <x86intrin.h>   // __rdtsc

// ---------------------------------------------------------------------------
// TSC calibration: measure TSC ticks per nanosecond.
// ---------------------------------------------------------------------------
static double calibrate_tsc_ghz() {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    uint64_t c0 = __rdtsc();

    // Spin for ~100ms
    struct timespec sleep_req = {0, 100'000'000L};
    nanosleep(&sleep_req, nullptr);

    uint64_t c1 = __rdtsc();
    clock_gettime(CLOCK_MONOTONIC, &t1);

    uint64_t ns_elapsed = (uint64_t)(t1.tv_sec - t0.tv_sec) * 1'000'000'000ULL
                        + (uint64_t)(t1.tv_nsec - t0.tv_nsec);
    return (double)(c1 - c0) / (double)ns_elapsed; // ticks/ns  (≈ GHz)
}

// ---------------------------------------------------------------------------
// Read the IB TX counter from sysfs.
// Returns 0 on error (caller should treat same as last known value).
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Read a single IB counter from sysfs.
// Returns 0 on error.
// ---------------------------------------------------------------------------
static uint64_t read_counter(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return 0;
    uint64_t val = 0;
    fscanf(f, "%lu", &val);
    fclose(f);
    return val;
}

// ---------------------------------------------------------------------------
// Current wall-clock time in nanoseconds (CLOCK_REALTIME for CSV timestamps).
// ---------------------------------------------------------------------------
static uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1'000'000'000ULL + (uint64_t)ts.tv_nsec;
}

// ---------------------------------------------------------------------------
// Sampler thread entry point.
// ---------------------------------------------------------------------------
void sampler_thread(SampleBuffer* buf, std::atomic<bool>* stop) {
    const double tsc_ghz = calibrate_tsc_ghz(); // ticks per ns
    const uint64_t interval_ticks = (uint64_t)(1'000'000.0 * tsc_ghz); // 1ms in ticks

    uint64_t next_tick = __rdtsc() + interval_ticks;

    while (!stop->load(std::memory_order_relaxed)) {
        // Busy-wait until the next scheduled tick
        while (__rdtsc() < next_tick) {
            _mm_pause(); // hint to CPU: we're in a spin loop
        }
        next_tick += interval_ticks;

        Sample s;
        s.timestamp_ns = now_ns();
        s.xmit_data    = read_counter("/sys/class/infiniband/mlx5_0/ports/1/counters/port_xmit_data");
        s.rcv_data     = read_counter("/sys/class/infiniband/mlx5_0/ports/1/counters/port_rcv_data");
        s.xmit_pkts    = read_counter("/sys/class/infiniband/mlx5_0/ports/1/counters/port_xmit_packets");
        s.rcv_pkts     = read_counter("/sys/class/infiniband/mlx5_0/ports/1/counters/port_rcv_packets");
        buf->push(s); // non-blocking; silently drops if writer is too slow
    }
}
