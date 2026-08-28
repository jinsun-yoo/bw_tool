#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <csignal>
#include <sched.h>
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

// fork()/exec() inherit the caller's CPU affinity mask. If bw_monitor is
// launched under a job launcher (e.g. `mpirun bw-start ...`) that pins its
// child to a single core, the daemonized process -- including the sampler
// thread's busy-spin loop in sampler.cpp -- stays confined to that same
// core for its entire lifetime. That core is often the very one a
// co-located compute rank (e.g. an NCCL proxy/progress thread) is bound to,
// so the sampler thread steals cycles from it and skews measured bandwidth.
// Reset the affinity mask to all online CPUs so the daemon (and its
// threads) are free to be scheduled anywhere, regardless of how it was
// launched.
static void reset_cpu_affinity() {
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    if (nprocs <= 0) return;

    cpu_set_t* set = CPU_ALLOC((size_t)nprocs);
    if (!set) return;
    size_t set_size = CPU_ALLOC_SIZE((size_t)nprocs);
    CPU_ZERO_S(set_size, set);
    for (long i = 0; i < nprocs; ++i) CPU_SET_S((size_t)i, set_size, set);

    if (sched_setaffinity(0, set_size, set) != 0) {
        perror("sched_setaffinity");
    }
    CPU_FREE(set);
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

    reset_cpu_affinity();

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

static bool in_mpi_context() {
    // Detect common MPI launcher env vars across OpenMPI/PMI/PMIx stacks.
    return getenv("OMPI_COMM_WORLD_RANK") != nullptr ||
           getenv("PMI_RANK") != nullptr ||
           getenv("PMIX_RANK") != nullptr ||
           getenv("MPI_LOCALRANKID") != nullptr;
}

static bool get_local_hostname(char* out, size_t len) {
    if (!out || len == 0) return false;
    if (gethostname(out, len) != 0) return false;
    out[len - 1] = '\0';
    return out[0] != '\0';
}

// Build CSV path:
// - <output_dir>/bwmonitor-<SLURM_JOB_ID>.csv (if SLURM_JOB_ID is set)
// - <output_dir>/bwmonitor-MMDD_HHMMSS.csv (fallback)
// If running under MPI context, append -<hostname> before .csv.
static void build_csv_path(const char* output_dir, char* out, size_t len) {
    const char* slurm_job_id = getenv("SLURM_JOB_ID");
    const bool mpi = in_mpi_context();

    char hostname[256] = {0};
    const bool have_hostname = mpi && get_local_hostname(hostname, sizeof(hostname));

    time_t now = time(nullptr);
    struct tm* tm = localtime(&now);
    char ts[32];
    if (!tm || strftime(ts, sizeof(ts), "%m%d_%H%M%S", tm) == 0) {
        snprintf(ts, sizeof(ts), "unknown_time");
    }

    if (slurm_job_id && slurm_job_id[0] != '\0') {
        if (have_hostname) {
            snprintf(out, len, "%s/bwmonitor-%s-%s-%s.csv", output_dir, slurm_job_id, hostname, ts);
        } else {
            snprintf(out, len, "%s/bwmonitor-%s-%s.csv", output_dir, slurm_job_id, ts);
        }
        return;
    }

    if (have_hostname) {
        snprintf(out, len, "%s/bwmonitor-%s-%s.csv", output_dir, ts, hostname);
    } else {
        snprintf(out, len, "%s/bwmonitor-%s.csv", output_dir, ts);
    }
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
    printf("CSV path is %s\n", csv_path);

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
