/*
 * Simulated execution time for the fake GPUs.
 *
 * Every stream has a timeline: asynchronous work (kernels, copies, library
 * calls) is appended to it and returns immediately; synchronisation waits
 * until the timeline catches up with real time. Events take timestamps on
 * the timeline, so cudaEventElapsedTime reports the simulated durations.
 * Busy time is added to the tray's occupancy state, which NVML turns into
 * utilisation, power and temperature.
 *
 * Costs are rough GB200 figures, multiplied by latency_scale.
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <time.h>
#include "occupancy.h"

typedef int CUresult;
typedef void *CUstream;
#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_VALUE 1
#define CUDA_ERROR_NOT_READY 600

#define API __attribute__((visibility("default")))

int fg_current_phys(void); /* cuda_stub.c */

/* ---- cost model (nanoseconds before latency_scale) ------------------------ */

#define LAUNCH_NS 3000.0          /* kernel launch + minimal execution */
#define WAVE_NS 2500.0            /* one wave of blocks across all SMs */
#define COPY_NS 2000.0            /* copy setup */
#define HOST_LINK_B_PER_NS 400.0  /* NVLink-C2C host <-> GPU, ~400 GB/s */
#define HBM_B_PER_NS 4000.0       /* device-to-device, 8 TB/s read + write */
#define NVLINK_B_PER_NS 900.0     /* GPU <-> GPU over NVLink5 */

uint64_t fg_kernel_ns(unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz)
{
    double blocks = (double)gx * gy * gz;
    unsigned tpb = bx * by * bz;
    if (tpb == 0) tpb = 1;
    unsigned per_sm = 2048 / tpb;
    if (per_sm < 1) per_sm = 1;
    if (per_sm > 32) per_sm = 32;
    double waves = blocks / ((double)FG_SM_COUNT * per_sm);
    if (waves < 1) waves = 1;
    return (uint64_t)(LAUNCH_NS + (waves - 1) * WAVE_NS);
}

uint64_t fg_copy_ns(size_t bytes, int kind) /* 0 host link, 1 device, 2 peer */
{
    double bw = kind == 1 ? HBM_B_PER_NS : kind == 2 ? NVLINK_B_PER_NS : HOST_LINK_B_PER_NS;
    return (uint64_t)(COPY_NS + (double)bytes / bw);
}

/* ---- streams and events --------------------------------------------------- */

/* Like the hardware's push buffer, a stream holds at most QUEUE_DEPTH
 * pending operations; launching more blocks the CPU until the oldest one
 * completes. Without this a tight launch loop would queue minutes of work. */
#define QUEUE_DEPTH 1024

struct fg_stream {
    unsigned magic;
    int phys;
    uint64_t ready_ns; /* when all work queued so far is done */
    uint64_t done[QUEUE_DEPTH]; /* completion times of recent operations (ring) */
    unsigned next;
};
#define STREAM_MAGIC 0x57AE4A11u

struct fg_event {
    unsigned magic;
    int recorded;
    uint64_t at_ns;
};
#define EVENT_MAGIC 0xE7E47001u

static pthread_mutex_t tl_lock = PTHREAD_MUTEX_INITIALIZER;
/* The legacy default stream per GPU; it also waits for all other streams. */
static struct fg_stream default_stream[FG_MAX_GPUS];
static uint64_t device_ready[FG_MAX_GPUS];

static struct fg_stream *stream_of(CUstream h)
{
    struct fg_stream *s = h;
    /* NULL, CU_STREAM_LEGACY (1) and CU_STREAM_PER_THREAD (2) are the default stream. */
    if ((uintptr_t)h > 2 && s->magic == STREAM_MAGIC) return s;
    int p = fg_current_phys();
    default_stream[p].magic = STREAM_MAGIC;
    default_stream[p].phys = p;
    return &default_stream[p];
}

static int is_default(struct fg_stream *s) { return s >= default_stream && s < default_stream + FG_MAX_GPUS; }

static void sleep_until(uint64_t t)
{
    struct timespec ts = {.tv_sec = (time_t)(t / 1000000000ULL), .tv_nsec = (long)(t % 1000000000ULL)};
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL) == EINTR) {}
}

/* Appends `ns` of work to a stream; returns when it will be done. Exported so
 * the cuBLAS, cuDNN and NCCL stubs share the same timelines. */
API uint64_t fakegpu_enqueue(CUstream h, uint64_t ns)
{
    ns = (uint64_t)((double)ns * fg_latency_scale());
    pthread_mutex_lock(&tl_lock);
    struct fg_stream *s = stream_of(h);
    uint64_t oldest = s->done[s->next];
    pthread_mutex_unlock(&tl_lock);
    if (oldest > fg_now_ns()) sleep_until(oldest); /* queue full: wait for a slot */

    uint64_t now = fg_now_ns();
    pthread_mutex_lock(&tl_lock);
    uint64_t start = s->ready_ns > now ? s->ready_ns : now;
    if (is_default(s) && device_ready[s->phys] > start) start = device_ready[s->phys];
    s->ready_ns = start + ns;
    if (s->ready_ns > device_ready[s->phys]) device_ready[s->phys] = s->ready_ns;
    s->done[s->next] = s->ready_ns;
    s->next = (s->next + 1) % QUEUE_DEPTH;
    uint64_t done = s->ready_ns;
    int phys = s->phys;
    pthread_mutex_unlock(&tl_lock);

    struct fg_occupancy *o = fg_occupancy();
    if (o && ns) __atomic_add_fetch(&o->gpu[phys].busy_ns, ns, __ATOMIC_RELAXED);
    return done;
}

/* NVLink traffic of the stream's GPU, counted in the occupancy state (the
 * switch tray reads it as fabric telemetry). Exported for the NCCL stub. */
API void fakegpu_nvlink_traffic(CUstream h, uint64_t tx, uint64_t rx)
{
    pthread_mutex_lock(&tl_lock);
    int phys = stream_of(h)->phys;
    pthread_mutex_unlock(&tl_lock);
    struct fg_occupancy *o = fg_occupancy();
    if (!o) return;
    __atomic_add_fetch(&o->gpu[phys].nvlink_tx, tx, __ATOMIC_RELAXED);
    __atomic_add_fetch(&o->gpu[phys].nvlink_rx, rx, __ATOMIC_RELAXED);
}

/* Blocking operations: queue the work, then wait for it. */
API void fakegpu_run(CUstream h, uint64_t ns) { sleep_until(fakegpu_enqueue(h, ns)); }

static uint64_t stream_ready(CUstream h)
{
    pthread_mutex_lock(&tl_lock);
    struct fg_stream *s = stream_of(h);
    uint64_t t = is_default(s) ? device_ready[s->phys] : s->ready_ns;
    pthread_mutex_unlock(&tl_lock);
    return t;
}

API CUresult cuStreamCreate(CUstream *out, unsigned flags)
{
    (void)flags;
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    struct fg_stream *s = calloc(1, sizeof *s);
    s->magic = STREAM_MAGIC;
    s->phys = fg_current_phys();
    *out = s;
    return CUDA_SUCCESS;
}

API CUresult cuStreamCreateWithPriority(CUstream *out, unsigned flags, int prio)
{
    (void)prio;
    return cuStreamCreate(out, flags);
}

API CUresult cuStreamDestroy_v2(CUstream h)
{
    struct fg_stream *s = h;
    if ((uintptr_t)h > 2 && s->magic == STREAM_MAGIC && !is_default(s)) {
        s->magic = 0;
        free(s);
    }
    return CUDA_SUCCESS;
}

API CUresult cuStreamSynchronize(CUstream h) { sleep_until(stream_ready(h)); return CUDA_SUCCESS; }

API CUresult cuStreamQuery(CUstream h) { return fg_now_ns() >= stream_ready(h) ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY; }

API CUresult cuCtxSynchronize(void)
{
    pthread_mutex_lock(&tl_lock);
    uint64_t t = device_ready[fg_current_phys()];
    pthread_mutex_unlock(&tl_lock);
    sleep_until(t);
    return CUDA_SUCCESS;
}

API CUresult cuCtxSynchronize_v2(void *ctx) { (void)ctx; return cuCtxSynchronize(); }

API CUresult cuEventCreate(void **out, unsigned flags)
{
    (void)flags;
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    struct fg_event *e = calloc(1, sizeof *e);
    e->magic = EVENT_MAGIC;
    *out = e;
    return CUDA_SUCCESS;
}

API CUresult cuEventDestroy_v2(void *h)
{
    struct fg_event *e = h;
    if (e && e->magic == EVENT_MAGIC) {
        e->magic = 0;
        free(e);
    }
    return CUDA_SUCCESS;
}

API CUresult cuEventRecord(void *h, CUstream st)
{
    struct fg_event *e = h;
    if (!e || e->magic != EVENT_MAGIC) return CUDA_ERROR_INVALID_VALUE;
    uint64_t now = fg_now_ns(), t = stream_ready(st);
    e->at_ns = t > now ? t : now;
    e->recorded = 1;
    return CUDA_SUCCESS;
}

API CUresult cuEventRecordWithFlags(void *h, CUstream st, unsigned flags) { (void)flags; return cuEventRecord(h, st); }

API CUresult cuEventQuery(void *h)
{
    struct fg_event *e = h;
    if (!e || e->magic != EVENT_MAGIC) return CUDA_ERROR_INVALID_VALUE;
    return !e->recorded || fg_now_ns() >= e->at_ns ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}

API CUresult cuEventSynchronize(void *h)
{
    struct fg_event *e = h;
    if (!e || e->magic != EVENT_MAGIC) return CUDA_ERROR_INVALID_VALUE;
    if (e->recorded) sleep_until(e->at_ns);
    return CUDA_SUCCESS;
}

API CUresult cuEventElapsedTime(float *ms, void *a, void *b)
{
    struct fg_event *ea = a, *eb = b;
    if (!ms || !ea || !eb || ea->magic != EVENT_MAGIC || eb->magic != EVENT_MAGIC) return CUDA_ERROR_INVALID_VALUE;
    *ms = (float)((double)(int64_t)(eb->at_ns - ea->at_ns) / 1e6);
    return CUDA_SUCCESS;
}

API CUresult cuEventElapsedTime_v2(float *ms, void *a, void *b) { return cuEventElapsedTime(ms, a, b); }

API CUresult cuStreamWaitEvent(CUstream h, void *ev, unsigned flags)
{
    (void)flags;
    struct fg_event *e = ev;
    if (!e || e->magic != EVENT_MAGIC) return CUDA_ERROR_INVALID_VALUE;
    pthread_mutex_lock(&tl_lock);
    struct fg_stream *s = stream_of(h);
    if (e->recorded && e->at_ns > s->ready_ns) s->ready_ns = e->at_ns;
    pthread_mutex_unlock(&tl_lock);
    return CUDA_SUCCESS;
}

/* ---- kernel launches --------------------------------------------------------- */

API CUresult cuLaunchKernel(void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                            unsigned smem, CUstream st, void **params, void **extra)
{
    (void)f; (void)smem; (void)params; (void)extra;
    fakegpu_enqueue(st, fg_kernel_ns(gx, gy, gz, bx, by, bz));
    return CUDA_SUCCESS;
}

API CUresult cuLaunchCooperativeKernel(void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                                       unsigned bz, unsigned smem, CUstream st, void **params)
{
    return cuLaunchKernel(f, gx, gy, gz, bx, by, bz, smem, st, params, NULL);
}

typedef struct {
    unsigned gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes;
    CUstream hStream;
} CUlaunchConfig;

API CUresult cuLaunchKernelEx(const CUlaunchConfig *c, void *f, void **params, void **extra)
{
    (void)f; (void)params; (void)extra;
    if (!c) return CUDA_ERROR_INVALID_VALUE;
    fakegpu_enqueue(c->hStream, fg_kernel_ns(c->gridDimX, c->gridDimY, c->gridDimZ, c->blockDimX, c->blockDimY, c->blockDimZ));
    return CUDA_SUCCESS;
}

/* Host callbacks run when the stream reaches them. */
API CUresult cuLaunchHostFunc(CUstream st, void (*fn)(void *), void *data)
{
    sleep_until(stream_ready(st));
    if (fn) fn(data);
    return CUDA_SUCCESS;
}

/* A captured graph replays as one kernel's worth of work. */
API CUresult cuGraphLaunch(void *exec, CUstream st)
{
    (void)exec;
    fakegpu_enqueue(st, (uint64_t)LAUNCH_NS);
    return CUDA_SUCCESS;
}
