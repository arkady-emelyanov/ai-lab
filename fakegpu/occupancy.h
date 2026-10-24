/*
 * GPU occupancy shared by every process on a tray: which processes use which
 * GPU and how much memory, and how long each GPU has been busy. libcuda
 * writes it, NVML (and so nvidia-smi and exporters) reads it.
 *
 * A file (state_path in fakegpu.conf, default /dev/shm/fakegpu) mapped by
 * all processes; on the lab trays it lives on a volume the switch tray reads
 * (NVLink traffic and utilisation for fabric telemetry).
 * An all-zero file is a valid empty state, so whoever opens it first
 * creates it; updates use atomics.
 *
 * Processes run in different PID namespaces (the tray, Kubernetes pods), so
 * a slot's PID only means something in its own namespace. A slot records the
 * namespace, and its owner holds an open-file-description lock on the slot's
 * first byte: the kernel drops the lock when the process exits, which tells
 * every reader, in any namespace, whether the slot is still live. Slots of
 * processes that died without cleaning up are reclaimed by readers.
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
    int32_t pid;          /* 0 = free slot; PID in the owner's namespace */
    int32_t gpu;          /* physical GPU index */
    uint64_t mem;         /* bytes allocated by this process on this GPU and MIG device (proc_mig[i]) */
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

/* GPU reset state (nvidia-smi --gpu-reset). As on GB200, a GPU takes its
 * NVLink partition's clique at a reset (or node reboot): until then NVML
 * reports the previous clique and recovery action GPU_RESET. Read by
 * fakebmc and fakenmxc: keep in sync with their copies of the layout. */
struct fg_gpu_reset {
    uint32_t clique;      /* clique taken at the last reset, plus one; 0 = not yet set */
    uint32_t count;       /* resets so far */
    uint64_t reset_ns;    /* CLOCK_MONOTONIC time of the last reset */
};

/* InfiniBand traffic of the GPU's NIC (one per GPU, mlx5_<gpu>): NCCL
 * traffic between ranks in different NVLink partitions. Read by the trays'
 * InfiniBand port counters (roles/fakeib). */
struct fg_gpu_net {
    uint64_t ib_tx;       /* bytes sent */
    uint64_t ib_rx;       /* bytes received */
};

/* MIG state of a GPU (mig.h): the mode, and GPU instances by GI id (the
 * slot of their placement), each with its compute instances by CI id. */
#define FG_MIG_SLOTS 15
#define FG_MIG_CIS 8

struct fg_mig_ci {
    uint8_t used, profile; /* profile: NVML compute instance profile index */
    uint8_t start, size;   /* placement in the GI's compute slices */
};

struct fg_mig_gi {
    uint8_t used, profile; /* profile: entry in mig_profiles (mig.h) */
    uint8_t start, size;   /* placement in the GPU's memory slices */
    uint32_t serial;       /* when it was created (MIG UUIDs) */
    struct fg_mig_ci ci[FG_MIG_CIS];
};

struct fg_gpu_mig {
    uint32_t current, pending; /* MIG mode: 1 enabled */
    uint32_t serial, pad;
    struct fg_mig_gi gi[FG_MIG_SLOTS];
};

/* The MIG device a process slot is on (the slot is per process, GPU and
 * MIG device): GI and CI id plus one, 0 for the whole GPU. */
struct fg_proc_mig {
    uint8_t gi, ci;
    uint8_t pad[6];
};

/* Fields added later go after everything older versions map, so processes
 * still running an older library never read or clear them. */
struct fg_occupancy {
    struct fg_gpu_state gpu[FG_MAX_GPUS];
    struct fg_proc proc[FG_MAX_PROCS];
    uint64_t pidns[FG_MAX_PROCS]; /* slot owner's PID namespace (inode of /proc/self/ns/pid) */
    struct fg_gpu_reset reset[FG_MAX_GPUS];
    struct fg_gpu_net net[FG_MAX_GPUS];
    struct fg_gpu_mig mig[FG_MAX_GPUS];
    struct fg_proc_mig proc_mig[FG_MAX_PROCS];
};

static inline uint64_t fg_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static struct fg_occupancy *fg_occ;
static int fg_occ_fd = -1; /* kept open: slot locks live on it */
static pthread_once_t fg_occ_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t fg_occ_claim = PTHREAD_MUTEX_INITIALIZER;

/* A forked child shares the parent's open file description, and with it the
 * parent's slot locks, which it would then take for its own. Give it its
 * own description (the mapping is inherited). */
static void fg_occ_atfork_child(void)
{
    if (fg_occ_fd < 0) return;
    int fd = open(fg_cfg.state_path, O_RDWR | O_CLOEXEC);
    close(fg_occ_fd);
    fg_occ_fd = fd;
}

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
    if (p == MAP_FAILED) { close(fd); return; }
    fg_occ = p;
    fg_occ_fd = fd;
    pthread_atfork(NULL, NULL, fg_occ_atfork_child);
}

static inline struct fg_occupancy *fg_occupancy(void)
{
    pthread_once(&fg_occ_once, fg_occ_open);
    return fg_occ;
}

/* This process's PID namespace: the inode of /proc/self/ns/pid. */
static inline uint64_t fg_pidns(void)
{
    static uint64_t ns;
    if (!ns) {
        struct stat st;
        ns = stat("/proc/self/ns/pid", &st) == 0 ? (uint64_t)st.st_ino : 1;
    }
    return ns;
}

static inline int fg_slot_mine(struct fg_occupancy *o, int i)
{
    return __atomic_load_n(&o->proc[i].pid, __ATOMIC_ACQUIRE) == getpid()
        && __atomic_load_n(&o->pidns[i], __ATOMIC_ACQUIRE) == fg_pidns();
}

/* Lock (type F_WRLCK) or unlock (F_UNLCK) slot i's first byte, without
 * waiting. Returns 0 on success. */
static inline int fg_slot_lock(struct fg_occupancy *o, int i, short type)
{
    struct flock fl = {.l_type = type, .l_whence = SEEK_SET,
                       .l_start = (off_t)((char *)&o->proc[i] - (char *)o), .l_len = 1};
    return fcntl(fg_occ_fd, F_OFD_SETLK, &fl);
}

/* Whether slot i belongs to a live process: ours, or locked by its owner. */
static inline int fg_slot_alive(struct fg_occupancy *o, int i)
{
    struct fg_proc *p = &o->proc[i];
    int32_t pid = __atomic_load_n(&p->pid, __ATOMIC_ACQUIRE);
    if (!pid) return 0;
    if (fg_slot_mine(o, i)) return 1;
    struct flock fl = {.l_type = F_WRLCK, .l_whence = SEEK_SET,
                       .l_start = (off_t)((char *)p - (char *)o), .l_len = 1};
    if (fcntl(fg_occ_fd, F_OFD_GETLK, &fl) == 0) return fl.l_type != F_UNLCK;
    /* No lock support: judge by PID where that means something. */
    if (__atomic_load_n(&o->pidns[i], __ATOMIC_ACQUIRE) != fg_pidns()) return 1;
    return kill(pid, 0) == 0 || errno == EPERM;
}

/* Whether slot i is on MIG device (gi, ci) of its GPU; gi -1: the whole
 * GPU, ci -1: any compute instance of GI gi. */
static inline int fg_slot_on(struct fg_occupancy *o, int i, int gi, int ci)
{
    const struct fg_proc_mig *m = &o->proc_mig[i];
    int sgi = (int)__atomic_load_n(&m->gi, __ATOMIC_ACQUIRE) - 1, sci = (int)__atomic_load_n(&m->ci, __ATOMIC_ACQUIRE) - 1;
    return sgi == gi && (ci < 0 || sci == ci);
}

/* Slot for (this process, gpu, MIG device gi/ci: -1 for the whole GPU),
 * created on first use. A free slot is taken by locking it first: nobody
 * live holds it then, and readers stop reclaiming it. */
static inline struct fg_proc *fg_proc_slot(int gpu, int gi, int ci)
{
    struct fg_occupancy *o = fg_occupancy();
    if (!o) return NULL;
    struct fg_proc *found = NULL;
    /* Threads share the locks (one open file description): claim one at a time. */
    pthread_mutex_lock(&fg_occ_claim);
    for (int i = 0; i < FG_MAX_PROCS && !found; i++)
        if (fg_slot_mine(o, i) && o->proc[i].gpu == gpu && fg_slot_on(o, i, gi, ci))
            found = &o->proc[i];
    for (int i = 0; i < FG_MAX_PROCS && !found; i++) {
        if (fg_slot_alive(o, i) || fg_slot_lock(o, i, F_WRLCK) != 0) continue;
        int32_t cur = __atomic_load_n(&o->proc[i].pid, __ATOMIC_ACQUIRE);
        /* The pid is published first, so a reader that saw the old owner
         * cannot reclaim the slot; the other fields follow. */
        if (__atomic_compare_exchange_n(&o->proc[i].pid, &cur, getpid(), 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            __atomic_store_n(&o->pidns[i], fg_pidns(), __ATOMIC_RELEASE);
            __atomic_store_n(&o->proc[i].mem, 0, __ATOMIC_RELEASE);
            __atomic_store_n(&o->proc_mig[i].gi, (uint8_t)(gi + 1), __ATOMIC_RELEASE);
            __atomic_store_n(&o->proc_mig[i].ci, (uint8_t)(gi < 0 ? 0 : ci + 1), __ATOMIC_RELEASE);
            __atomic_store_n(&o->proc[i].gpu, gpu, __ATOMIC_RELEASE);
            found = &o->proc[i];
        } else {
            fg_slot_lock(o, i, F_UNLCK);
        }
    }
    pthread_mutex_unlock(&fg_occ_claim);
    return found;
}

/* Drop this process's slots (self: at exit) and any left by dead processes. */
static inline void fg_proc_cleanup(int self)
{
    struct fg_occupancy *o = fg_occupancy();
    if (!o) return;
    for (int i = 0; i < FG_MAX_PROCS; i++) {
        int32_t pid = __atomic_load_n(&o->proc[i].pid, __ATOMIC_ACQUIRE);
        if (!pid) continue;
        if (fg_slot_mine(o, i)) {
            if (self && __atomic_compare_exchange_n(&o->proc[i].pid, &pid, 0, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
                fg_slot_lock(o, i, F_UNLCK);
        } else if (!fg_slot_alive(o, i)) {
            __atomic_compare_exchange_n(&o->proc[i].pid, &pid, 0, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
        }
    }
}

/* GPU memory in use: every live process, whatever its namespace. */
static inline uint64_t fg_gpu_mem_used(int gpu)
{
    struct fg_occupancy *o = fg_occupancy();
    uint64_t sum = 0;
    if (!o) return 0;
    for (int i = 0; i < FG_MAX_PROCS; i++)
        if (o->proc[i].gpu == gpu && fg_slot_alive(o, i))
            sum += __atomic_load_n(&o->proc[i].mem, __ATOMIC_ACQUIRE);
    return sum;
}

/* This process's GPU memory on a GPU (all its MIG devices there). */
static inline uint64_t fg_own_mem(int gpu)
{
    struct fg_occupancy *o = fg_occupancy();
    uint64_t sum = 0;
    if (!o) return 0;
    for (int i = 0; i < FG_MAX_PROCS; i++)
        if (o->proc[i].gpu == gpu && fg_slot_mine(o, i))
            sum += __atomic_load_n(&o->proc[i].mem, __ATOMIC_ACQUIRE);
    return sum;
}

/* Memory in use on GPU instance gi of a GPU: every live process's on its
 * compute instances (they share the GI's memory). */
static inline uint64_t fg_gi_mem_used(int gpu, int gi)
{
    struct fg_occupancy *o = fg_occupancy();
    uint64_t sum = 0;
    if (!o) return 0;
    for (int i = 0; i < FG_MAX_PROCS; i++)
        if (o->proc[i].gpu == gpu && fg_slot_on(o, i, gi, -1) && fg_slot_alive(o, i))
            sum += __atomic_load_n(&o->proc[i].mem, __ATOMIC_ACQUIRE);
    return sum;
}

/* GPU memory limit of this process in bytes, 0 for none. A container given
 * a fraction of a GPU (NVIDIA's GPU fractioning) gets NVIDIA_GPU_MEMORY_LIMIT
 * in MiB, and the driver (MPS memory accounting) holds each of its processes
 * to it on each GPU: allocations beyond it fail with out of memory, and
 * cuMemGetInfo reports it as the total. */
static uint64_t fg_limit;
static pthread_once_t fg_limit_once = PTHREAD_ONCE_INIT;

static void fg_read_limit(void)
{
    const char *v = getenv("NVIDIA_GPU_MEMORY_LIMIT");
    char *end;
    unsigned long long mib = v ? strtoull(v, &end, 10) : 0;
    fg_limit = v && *v && !*end ? (uint64_t)mib << 20 : 0;
}

static inline uint64_t fg_mem_limit(void)
{
    pthread_once(&fg_limit_once, fg_read_limit);
    return fg_limit;
}

/* Whether n more bytes of device memory fit on a GPU for this process: in
 * the GPU's memory, within the process's limit, and, on GPU instance gi
 * (-1: the whole GPU) of gi_bytes, in what the GI's processes leave of it. */
static inline int fg_mem_fits(int gpu, int gi, uint64_t gi_bytes, uint64_t n)
{
    if (fg_gpu_mem_used(gpu) + n > fg_mem_bytes()) return 0;
    uint64_t limit = fg_mem_limit();
    if (limit && fg_own_mem(gpu) + n > limit) return 0;
    return gi < 0 || fg_gi_mem_used(gpu, gi) + n <= gi_bytes;
}

#endif
