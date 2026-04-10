#include "writer.h"
#include <cstdio>
#include <cstring>
#include <time.h>

// port_xmit_data / port_rcv_data are in 4-byte (dword) units.
// bandwidth_gbps = delta_raw * 4 * 8 / delta_ns / 1e9 = delta_raw * 32 / delta_ns
static inline double compute_gbps(uint64_t delta_raw, uint64_t delta_ns) {
    if (delta_ns == 0) return 0.0;
    return (double)delta_raw * 32.0 / (double)delta_ns;
}

// port_xmit_packets / port_rcv_packets are in packet units.
// rate_mpps = delta_pkts / delta_ns * 1000  (million packets per second)
static inline double compute_mpps(uint64_t delta_pkts, uint64_t delta_ns) {
    if (delta_ns == 0) return 0.0;
    return (double)delta_pkts * 1000.0 / (double)delta_ns;
}

void writer_thread(SampleBuffer* buf, std::atomic<bool>* stop, const char* csv_path) {
    FILE* f = fopen(csv_path, "w");
    if (!f) {
        perror("writer: fopen");
        return;
    }

    // Large write buffer: reduces fwrite syscall frequency
    char io_buf[256 * 1024];
    setvbuf(f, io_buf, _IOFBF, sizeof(io_buf));

    fprintf(f, "timestamp_ns,elapsed_ns,xmit_data_raw,rcv_data_raw,xmit_pkts_raw,rcv_pkts_raw,xmit_gbps,rcv_gbps,xmit_pps,rcv_pps\n");

    uint64_t first_ts      = 0;
    uint64_t prev_ts       = 0;
    uint64_t prev_xmit_data = 0;
    uint64_t prev_rcv_data  = 0;
    uint64_t prev_xmit_pkts = 0;
    uint64_t prev_rcv_pkts  = 0;
    bool     has_prev       = false;

    while (!stop->load(std::memory_order_relaxed) || true) {
        Sample s;
        if (!buf->pop(s)) {
            // Buffer empty: yield briefly to avoid spinning when idle
            struct timespec t = {0, 100'000L}; // 100µs
            nanosleep(&t, nullptr);

            // Exit only after stop is set AND buffer is drained
            if (stop->load(std::memory_order_relaxed))
                break;
            continue;
        }

        if (first_ts == 0)
            first_ts = s.timestamp_ns;

        uint64_t elapsed_ns    = s.timestamp_ns - first_ts;
        double   xmit_gbps     = 0.0;
        double   rcv_gbps      = 0.0;
        double   xmit_pps      = 0.0;
        double   rcv_pps       = 0.0;

        if (has_prev) {
            uint64_t delta_ns = s.timestamp_ns - prev_ts;
            auto delta = [](uint64_t cur, uint64_t prev) -> uint64_t {
                return (cur >= prev) ? cur - prev : 0;
            };
            xmit_gbps = compute_gbps(delta(s.xmit_data, prev_xmit_data), delta_ns);
            rcv_gbps  = compute_gbps(delta(s.rcv_data,  prev_rcv_data),  delta_ns);
            xmit_pps  = compute_mpps(delta(s.xmit_pkts, prev_xmit_pkts), delta_ns);
            rcv_pps   = compute_mpps(delta(s.rcv_pkts,  prev_rcv_pkts),  delta_ns);
        }

        fprintf(f, "%lu,%lu,%lu,%lu,%lu,%lu,%.6f,%.6f,%.6f,%.6f\n",
                s.timestamp_ns, elapsed_ns,
                s.xmit_data, s.rcv_data, s.xmit_pkts, s.rcv_pkts,
                xmit_gbps, rcv_gbps, xmit_pps, rcv_pps);

        prev_ts        = s.timestamp_ns;
        prev_xmit_data = s.xmit_data;
        prev_rcv_data  = s.rcv_data;
        prev_xmit_pkts = s.xmit_pkts;
        prev_rcv_pkts  = s.rcv_pkts;
        has_prev       = true;
    }

    fflush(f);
    fclose(f);
}
