# bw_monitor bandwidth degradation — root cause & fix

`bw-start`/`bw-stop` launched via `mpirun` inherit OpenMPI's default core binding; `fork()` in `bw_monitor`'s daemonize path propagates that same single-core affinity to the daemon and its busy-spinning sampler thread, which then contends with NCCL's rank/proxy thread pinned to the same core, degrading measured bandwidth.

**Fix**: `bw_tool/src/main.cpp` now resets the daemon's CPU affinity to all online cores via `sched_setaffinity()` right after daemonizing. No slurm script changes needed — rebuild/reinstall (`make install`) is sufficient.
