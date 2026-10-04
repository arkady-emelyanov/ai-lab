/*
 * GPU occupancy shared by every process on a tray: which processes use which
 * GPU and how much memory, and how long each GPU has been busy. libcuda
 * writes it, NVML (and so nvidia-smi and exporters) reads it.
 *
 * A file (state_path in fakegpu.conf, default /dev/shm/fakegpu) mapped by
 * all processes; on the lab trays it lives on a volume the switch tray reads
 * (NVLink traffic and utilisation for fabric telemetry).
 * An all-zero file is a valid empty state, so whoever opens it first
 * creates it; updates use atomics, no locks. Slots of processes that died
 * without cleaning up are reclaimed by readers.
 */
#ifndef FAKEGPU_OCCUPANCY_H
#define FAKEGPU_OCCUPANCY_H

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include "fakegpu.h"

#define FG_MAX_PROCS 128

struct fg_proc {
    int32_t pid;          /* 0 = free slot */
    int32_t gpu;          /* physical GPU index */
    uint64_t mem;         /* bytes allocated by this process on this GPU */
};

/* Layout is read by fakenmxc (switch-side telemetry): keep in sync with
 * fakenmxc/telemetry.go. All fields little-endian, naturally aligned. */
struct fg_gpu_state {
    uint64_t busy_ns;      /* total time the GPU spent executing work */
    uint64_t sample_ns;    /* last utilisation sample (NVML) */
    uint64_t sample_busy;  /* busy_ns at that sample */
    uint32_t util;         /* utilisation % over the last window */
    uint32_t temp_mc;      /* smoothed temperature, milli-degrees C */
    uint64_t temp_ns;      /* last temperature update */
    uint64_t nvlink_tx;    /* bytes sent over NVLink (collectives, peer copies) */
    uint64_t nvlink_rx;    /* bytes received over NVLink */
};

struct fg_occupancy {
    struct fg_gpu_state gpu[FG_MAX_GPUS];
    struct fg_proc proc[FG_MAX_PROCS];
};

static inline uint64_t fg_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static struct fg_occupancy *fg_occ;
static pthread_once_t fg_occ_once = PTHREAD_ONCE_INIT;

static void fg_occ_open(void)
{
    fg_init();
    const char *path = fg_cfg.state_path;
    mode_t old = umask(0);
    int fd = open(path, O_RDWR | O_CREAT | O_CLOEXEC, 0666);
    umask(old);
    if (fd < 0) return; /* occupancy is best effort */
    struct stat st;
    if (fstat(fd, &st) == 0 && (size_t)st.st_size < sizeof(struct fg_occupancy))
        if (ftruncate(fd, sizeof(struct fg_occupancy)) != 0) { close(fd); return; }
    void *p = mmap(NULL, sizeof(struct fg_occupancy), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p != MAP_FAILED) fg_occ = p;
}

static inline struct fg_occupancy *fg_occupancy(void)
{
    pthread_once(&fg_occ_once, fg_occ_open);
    return fg_occ;
}

static inline int fg_pid_alive(int32_t pid)
{
    return pid > 0 && (kill(pid, 0) == 0 || errno == EPERM);
}

/* Slot for (this process, gpu), created on first use. */
static inline struct fg_proc *fg_proc_slot(int gpu)
{
    struct fg_occupancy *o = fg_occupancy();
    if (!o) return NULL;
    int32_t me = getpid();
    for (int i = 0; i < FG_MAX_PROCS; i++)
        if (__atomic_load_n(&o->proc[i].pid, __ATOMIC_ACQUIRE) == me && o->proc[i].gpu == gpu)
            return &o->proc[i];
    for (int i = 0; i < FG_MAX_PROCS; i++) {
        int32_t cur = __atomic_load_n(&o->proc[i].pid, __ATOMIC_ACQUIRE);
        if (cur != 0 && fg_pid_alive(cur)) continue;
        /* Claim the slot, then reset its fields before publishing gpu. */
        if (__atomic_compare_exchange_n(&o->proc[i].pid, &cur, me, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            __atomic_store_n(&o->proc[i].mem, 0, __ATOMIC_RELEASE);
            __atomic_store_n(&o->proc[i].gpu, gpu, __ATOMIC_RELEASE);
            return &o->proc[i];
        }
    }
    return NULL;
}

/* Drop this process's slots (atexit) and any left by dead processes. */
static inline void fg_proc_cleanup(int self)
{
    struct fg_occupancy *o = fg_occupancy();
    if (!o) return;
    int32_t me = getpid();
    for (int i = 0; i < FG_MAX_PROCS; i++) {
        int32_t pid = __atomic_load_n(&o->proc[i].pid, __ATOMIC_ACQUIRE);
        if (pid != 0 && ((self && pid == me) || !fg_pid_alive(pid)))
            __atomic_compare_exchange_n(&o->proc[i].pid, &pid, 0, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
    }
}

static inline uint64_t fg_gpu_mem_used(int gpu)
{
    struct fg_occupancy *o = fg_occupancy();
    uint64_t sum = 0;
    if (!o) return 0;
    for (int i = 0; i < FG_MAX_PROCS; i++)
        if (__atomic_load_n(&o->proc[i].pid, __ATOMIC_ACQUIRE) && o->proc[i].gpu == gpu)
            sum += __atomic_load_n(&o->proc[i].mem, __ATOMIC_ACQUIRE);
    return sum;
}

#endif
