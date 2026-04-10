#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <csignal>
#include <thread>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "sampler.h"
#include "writer.h"

#define PIDFILE "/tmp/bw_monitor.pid"

static std::atomic<bool> g_stop{false};

static void handle_sigterm(int) {
    g_stop.store(true, std::memory_order_relaxed);
}

// Double-fork daemonize. Returns in the grandchild (daemon) process.
static void daemonize() {
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); exit(1); }
    if (pid > 0) exit(0); // parent exits

    if (setsid() < 0) { perror("setsid"); exit(1); }

    pid = fork();
    if (pid < 0) { perror("fork2"); exit(1); }
    if (pid > 0) exit(0); // first child exits

    // Redirect stdin/stdout/stderr to /dev/null
    int devnull = open("/dev/null", O_RDWR);
    if (devnull >= 0) {
        dup2(devnull, STDIN_FILENO);
        dup2(devnull, STDOUT_FILENO);
        dup2(devnull, STDERR_FILENO);
        if (devnull > STDERR_FILENO) close(devnull);
    }
}

static void write_pidfile() {
    FILE* f = fopen(PIDFILE, "w");
    if (!f) { perror("write_pidfile"); exit(1); }
    fprintf(f, "%d\n", (int)getpid());
    fclose(f);
}

// Build CSV path: <output_dir>/bwmonitor-MMDD_HHMMSS.csv
static void build_csv_path(const char* output_dir, char* out, size_t len) {
    time_t now = time(nullptr);
    struct tm* tm = localtime(&now);
    char ts[32];
    strftime(ts, sizeof(ts), "%m%d_%H%M%S", tm);
    snprintf(out, len, "%s/bwmonitor-%s.csv", output_dir, ts);
}

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s --start <output_dir>\n", prog);
}

int main(int argc, char* argv[]) {
    if (argc < 3 || strcmp(argv[1], "--start") != 0) {
        print_usage(argv[0]);
        return 1;
    }

    const char* output_dir = argv[2];

    char csv_path[4096];
    build_csv_path(output_dir, csv_path, sizeof(csv_path));

    daemonize();
    write_pidfile();

    // Install signal handler for graceful shutdown
    struct sigaction sa{};
    sa.sa_handler = handle_sigterm;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGINT,  &sa, nullptr);

    SampleBuffer buf;

    std::thread sampler(sampler_thread, &buf, &g_stop);
    std::thread writer(writer_thread,  &buf, &g_stop, csv_path);

    sampler.join();
    writer.join();

    remove(PIDFILE);
    return 0;
}
