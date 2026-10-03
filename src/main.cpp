#include <atomic>
#include <cerrno>
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

#define DEFAULT_DEVICE "mlx5_0"
#define PIDFILE_DIR "/tmp"
#define CPU_LOCK_PREFIX "/tmp/bw_monitor_cpu"

static std::atomic<bool> g_stop{false};

static char g_pidfile[512];
static char g_cpu_lockfile[512];

static void handle_sigterm(int) {
    g_stop.store(true, std::memory_order_relaxed);
}

// A lock file is stale if it holds no live PID.
static bool cpu_lock_is_stale(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return true;
    int pid = 0;
    const bool parsed = (fscanf(f, "%d", &pid) == 1);
    fclose(f);
    if (!parsed || pid <= 0) return true;
    return kill(pid, 0) != 0 && errno == ESRCH;
}

// fork()/exec() inherit the caller's CPU affinity mask. If bw_monitor is
// launched under a job launcher (e.g. `mpirun bw-start ...`) that pins its
// child to a single core, the daemonized process -- including the sampler
// thread's busy-spin loop in sampler.cpp -- stays confined to that same
// core for its entire lifetime. That core is often the very one a
// co-located compute rank (e.g. an NCCL proxy/progress thread) is bound to,
// so the sampler thread steals cycles from it and skews measured bandwidth.
//
// When several bw_monitor daemons run concurrently (one per NIC) their
// busy-spinning sampler threads must also not share a core with each other.
// Claim one core exclusively via a lock file, scanning from the highest
// online CPU downwards; fall back to "all CPUs" if none can be claimed.
static void claim_exclusive_cpu() {
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    if (nprocs <= 0) return;

    size_t set_size = CPU_ALLOC_SIZE((size_t)nprocs);
    cpu_set_t* set = CPU_ALLOC((size_t)nprocs);
    if (!set) return;

    for (long cpu = nprocs - 1; cpu >= 0; --cpu) {
        char path[512];
        snprintf(path, sizeof(path), "%s%ld.lock", CPU_LOCK_PREFIX, cpu);

        int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0644);
        if (fd < 0) {
            if (errno != EEXIST) continue;
            if (!cpu_lock_is_stale(path)) continue;
            fd = open(path, O_WRONLY | O_TRUNC, 0644);
            if (fd < 0) continue;
        }

        dprintf(fd, "%d\n", (int)getpid());
        close(fd);

        CPU_ZERO_S(set_size, set);
        CPU_SET_S((size_t)cpu, set_size, set);
        if (sched_setaffinity(0, set_size, set) == 0) {
            snprintf(g_cpu_lockfile, sizeof(g_cpu_lockfile), "%s", path);
            CPU_FREE(set);
            return;
        }
        unlink(path);
    }

    // No core could be claimed: spread over every online CPU instead of
    // inheriting the launcher's (possibly single-core) mask.
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

    claim_exclusive_cpu();

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
    FILE* f = fopen(g_pidfile, "w");
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
// If an explicit device was requested, append -<device> before .csv.
static void build_csv_path(const char* output_dir, const char* device_suffix,
                           char* out, size_t len) {
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

    char dev[128] = {0};
    if (device_suffix && device_suffix[0] != '\0') {
        snprintf(dev, sizeof(dev), "-%s", device_suffix);
    }

    if (slurm_job_id && slurm_job_id[0] != '\0') {
        if (have_hostname) {
            snprintf(out, len, "%s/bwmonitor-%s-%s-%s%s.csv", output_dir, slurm_job_id, hostname, ts, dev);
        } else {
            snprintf(out, len, "%s/bwmonitor-%s-%s%s.csv", output_dir, slurm_job_id, ts, dev);
        }
        return;
    }

    if (have_hostname) {
        snprintf(out, len, "%s/bwmonitor-%s-%s%s.csv", output_dir, ts, hostname, dev);
    } else {
        snprintf(out, len, "%s/bwmonitor-%s%s.csv", output_dir, ts, dev);
    }
}

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s --start <output_dir> [--device <ib_device>]\n", prog);
}

int main(int argc, char* argv[]) {
    if (argc < 3 || strcmp(argv[1], "--start") != 0) {
        print_usage(argv[0]);
        return 1;
    }

    const char* output_dir = argv[2];
    const char* device = DEFAULT_DEVICE;
    bool device_explicit = false;

    for (int i = 3; i < argc; ++i) {
        if ((strcmp(argv[i], "--device") == 0 || strcmp(argv[i], "-d") == 0) && i + 1 < argc) {
            device = argv[++i];
            device_explicit = true;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (strchr(device, '/') != nullptr) {
        fprintf(stderr, "Error: invalid device name '%s'\n", device);
        return 1;
    }

    if (device_explicit) {
        snprintf(g_pidfile, sizeof(g_pidfile), "%s/bw_monitor-%s.pid", PIDFILE_DIR, device);
    } else {
        snprintf(g_pidfile, sizeof(g_pidfile), "%s/bw_monitor.pid", PIDFILE_DIR);
    }

    char csv_path[4096];
    build_csv_path(output_dir, device_explicit ? device : nullptr, csv_path, sizeof(csv_path));
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

    std::thread sampler(sampler_thread, &buf, &g_stop, device);
    std::thread writer(writer_thread,  &buf, &g_stop, csv_path);

    sampler.join();
    writer.join();

    remove(g_pidfile);
    if (g_cpu_lockfile[0] != '\0') remove(g_cpu_lockfile);
    return 0;
}
