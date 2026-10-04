/*
 * Fake NCCL (libnccl.so.2). Every exported symbol returns ncclSuccess
 * (generated from symbols/nccl.syms). Communicators remember their size,
 * rank and device so frameworks can query them. Collectives take simulated
 * time on their stream (NVLink bandwidth with the usual bus-bandwidth
 * factors) and behave as if every peer contributed exactly what this rank
 * did (device memory is host memory under the fake driver), so cross-rank
 * consistency checks such as DDP's parameter verification pass.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include "stub_common.h"

typedef int ncclResult_t;
#define ncclSuccess 0
#define ncclInvalidArgument 4

#define NCCL_VERSION_CODE 23007 /* 2.30.7 */

typedef struct { char internal[128]; } ncclUniqueId;

struct fake_comm {
    int nranks, rank, dev;
};
typedef struct fake_comm *ncclComm_t;

static int current_device(void)
{
    int (*get)(int *) = (int (*)(int *))dlsym(RTLD_DEFAULT, "cudaGetDevice");
    int dev = 0;
    if (get) get(&dev);
    return dev;
}

static ncclResult_t new_comm(ncclComm_t *out, int nranks, int rank)
{
    if (!out || nranks < 1 || rank < 0 || rank >= nranks) return ncclInvalidArgument;
    struct fake_comm *c = calloc(1, sizeof *c);
    c->nranks = nranks;
    c->rank = rank;
    c->dev = current_device();
    *out = c;
    return ncclSuccess;
}

/* ---- version / errors -------------------------------------------------- */

API ncclResult_t ncclGetVersion(int *v)
{
    if (!v) return ncclInvalidArgument;
    *v = NCCL_VERSION_CODE;
    return ncclSuccess;
}

API const char *ncclGetErrorString(ncclResult_t r)
{
    switch (r) {
    case 0: return "no error";
    case 1: return "unhandled cuda error";
    case 2: return "unhandled system error";
    case 3: return "internal error";
    case 4: return "invalid argument";
    case 5: return "invalid usage";
    case 6: return "remote process exited or there was a network error";
    case 7: return "NCCL operation in progress";
    default: return "unknown result code";
    }
}

API const char *ncclGetLastError(ncclComm_t comm) { (void)comm; return ""; }
API void ncclResetDebugInit(void) {}
API void ncclParamDumpAll(void) {}

/* ---- communicators ----------------------------------------------------- */

API ncclResult_t ncclGetUniqueId(ncclUniqueId *id)
{
    if (!id) return ncclInvalidArgument;
    memset(id, 0, sizeof *id);
    snprintf(id->internal, sizeof id->internal, "fakegpu-nccl-%ld", random());
    return ncclSuccess;
}

API ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId id, int rank)
{
    (void)id;
    return new_comm(comm, nranks, rank);
}

API ncclResult_t ncclCommInitRankConfig(ncclComm_t *comm, int nranks, ncclUniqueId id, int rank, void *config)
{
    (void)config;
    return ncclCommInitRank(comm, nranks, id, rank);
}

API ncclResult_t ncclCommInitRankScalable(ncclComm_t *comm, int nranks, int rank, int nid, ncclUniqueId *ids, void *config)
{
    (void)nid; (void)ids; (void)config;
    return new_comm(comm, nranks, rank);
}

API ncclResult_t ncclCommInitAll(ncclComm_t *comms, int ndev, const int *devlist)
{
    if (!comms) return ncclInvalidArgument;
    for (int i = 0; i < ndev; i++) {
        new_comm(&comms[i], ndev, i);
        comms[i]->dev = devlist ? devlist[i] : i;
    }
    return ncclSuccess;
}

/* Split/shrink/grow keep the caller's rank, clamped into the new size. */
API ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *out, void *config)
{
    (void)color; (void)key; (void)config;
    if (!out) return ncclInvalidArgument;
    if (color < 0) { *out = NULL; return ncclSuccess; } /* NCCL_SPLIT_NOCOLOR */
    return new_comm(out, comm ? comm->nranks : 1, comm ? comm->rank : 0);
}

API ncclResult_t ncclCommShrink(ncclComm_t comm, int *excl, int nexcl, ncclComm_t *out, void *config, int flags)
{
    (void)excl; (void)config; (void)flags;
    int n = comm && comm->nranks - nexcl > 0 ? comm->nranks - nexcl : 1;
    int r = comm && comm->rank < n ? comm->rank : 0;
    return new_comm(out, n, r);
}

API ncclResult_t ncclCommGrow(ncclComm_t comm, int nranks, const ncclUniqueId *id, int rank, ncclComm_t *out, void *config)
{
    (void)comm; (void)id; (void)config;
    return new_comm(out, nranks, rank);
}

API ncclResult_t ncclCommGetUniqueId(ncclComm_t comm, ncclUniqueId *id) { (void)comm; return ncclGetUniqueId(id); }

API ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *err)
{
    (void)comm;
    if (!err) return ncclInvalidArgument;
    *err = ncclSuccess;
    return ncclSuccess;
}

API ncclResult_t ncclCommCount(const ncclComm_t comm, int *n)
{
    if (!comm || !n) return ncclInvalidArgument;
    *n = comm->nranks;
    return ncclSuccess;
}

API ncclResult_t ncclCommUserRank(const ncclComm_t comm, int *r)
{
    if (!comm || !r) return ncclInvalidArgument;
    *r = comm->rank;
    return ncclSuccess;
}

API ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int *d)
{
    if (!comm || !d) return ncclInvalidArgument;
    *d = comm->dev;
    return ncclSuccess;
}

API ncclResult_t ncclCommMemStats(ncclComm_t comm, int stat, uint64_t *v)
{
    (void)comm; (void)stat;
    if (!v) return ncclInvalidArgument;
    *v = 0;
    return ncclSuccess;
}

/* ---- memory / registration --------------------------------------------- */

API ncclResult_t ncclMemAlloc(void **p, size_t n)
{
    if (!p) return ncclInvalidArgument;
    *p = calloc(1, n ? n : 1);
    return ncclSuccess;
}

API ncclResult_t ncclMemFree(void *p) { free(p); return ncclSuccess; }

API ncclResult_t ncclCommRegister(const ncclComm_t comm, void *buf, size_t n, void **handle)
{
    (void)comm; (void)buf; (void)n;
    if (handle) *handle = stub_handle();
    return ncclSuccess;
}

API ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void *buf, size_t n, void **win, int flags)
{
    (void)flags;
    return ncclCommRegister(comm, buf, n, win);
}

API ncclResult_t ncclWinGetUserPtr(ncclComm_t comm, void *win, void **ptr)
{
    (void)comm; (void)win;
    if (ptr) *ptr = NULL;
    return ncclSuccess;
}

API ncclResult_t ncclRedOpCreatePreMulSum(int *op, void *scalar, int type, int residence, ncclComm_t comm)
{
    (void)scalar; (void)type; (void)residence; (void)comm;
    if (op) *op = 5; /* first user-defined op slot after the 5 built-ins */
    return ncclSuccess;
}

/* ---- collectives: every peer mirrors this rank -------------------------- */

#define COLLECTIVE_NS 10000.0 /* launch + synchronisation across ranks */

/* bus_factor: bytes each rank moves per byte of payload, e.g. 2(n-1)/n for
 * all-reduce, (n-1)/n for all-gather and reduce-scatter. */
static void collective_work(void *stream, double bytes, double bus_factor)
{
    stub_gpu_work(stream, COLLECTIVE_NS + bytes * bus_factor / STUB_NVLINK_B_PER_NS);
    /* Every rank sends and receives bus_factor x the payload over NVLink. */
    stub_nvlink_traffic(stream, bytes * bus_factor, bytes * bus_factor);
}

static double ring(ncclComm_t c, double k)
{
    double n = c ? c->nranks : 1;
    return n > 1 ? k * (n - 1) / n : 0;
}

static size_t type_size(int t)
{
    switch (t) {
    case 0: case 1: case 10: case 11: return 1; /* int8 uint8 fp8 */
    case 6: case 9: return 2;                   /* half bfloat16 */
    case 2: case 3: case 7: return 4;           /* int32 uint32 float */
    default: return 8;                          /* int64 uint64 double */
    }
}

static void copy(void *dst, const void *src, size_t count, int type)
{
    if (dst && src && dst != src) memmove(dst, src, count * type_size(type));
}

API ncclResult_t ncclAllReduce(const void *s, void *r, size_t n, int t, int op, ncclComm_t c, void *st)
{
    (void)op;
    copy(r, s, n, t);
    collective_work(st, (double)n * type_size(t), ring(c, 2));
    return ncclSuccess;
}

API ncclResult_t ncclReduce(const void *s, void *r, size_t n, int t, int op, int root, ncclComm_t c, void *st)
{
    (void)op; (void)root;
    copy(r, s, n, t);
    collective_work(st, (double)n * type_size(t), c && c->nranks > 1 ? 1 : 0);
    return ncclSuccess;
}

API ncclResult_t ncclBroadcast(const void *s, void *r, size_t n, int t, int root, ncclComm_t c, void *st)
{
    (void)root;
    copy(r, s, n, t);
    collective_work(st, (double)n * type_size(t), c && c->nranks > 1 ? 1 : 0);
    return ncclSuccess;
}

API ncclResult_t ncclReduceScatter(const void *s, void *r, size_t n, int t, int op, ncclComm_t c, void *st)
{
    (void)op;
    if (s && r) copy(r, (const char *)s + (size_t)(c ? c->rank : 0) * n * type_size(t), n, t);
    collective_work(st, (double)n * type_size(t) * (c ? c->nranks : 1), ring(c, 1));
    return ncclSuccess;
}

API ncclResult_t ncclAllGather(const void *s, void *r, size_t n, int t, ncclComm_t c, void *st)
{
    int ranks = c ? c->nranks : 1, me = c ? c->rank : 0;
    size_t chunk = n * type_size(t);
    if (!r || !s) return ncclSuccess;
    /* In-place calls already hold our chunk at slot `me`; copy from there. */
    const char *src = s == (char *)r + (size_t)me * chunk ? (char *)r + (size_t)me * chunk : s;
    for (int i = 0; i < ranks; i++) copy((char *)r + (size_t)i * chunk, src, n, t);
    collective_work(st, (double)chunk * ranks, ring(c, 1));
    return ncclSuccess;
}

API ncclResult_t ncclAlltoAll(const void *s, void *r, size_t n, int t, ncclComm_t c, void *st)
{
    copy(r, s, n * (size_t)(c ? c->nranks : 1), t);
    collective_work(st, (double)n * type_size(t) * (c ? c->nranks : 1), ring(c, 1));
    return ncclSuccess;
}

API ncclResult_t ncclGather(const void *s, void *r, size_t n, int t, int root, ncclComm_t c, void *st)
{
    (void)root;
    return ncclAllGather(s, r, n, t, c, st);
}

API ncclResult_t ncclScatter(const void *s, void *r, size_t n, int t, int root, ncclComm_t c, void *st)
{
    (void)root;
    if (s && r) copy(r, (const char *)s + (size_t)(c ? c->rank : 0) * n * type_size(t), n, t);
    collective_work(st, (double)n * type_size(t), c && c->nranks > 1 ? 1 : 0);
    return ncclSuccess;
}

API ncclResult_t ncclSend(const void *s, size_t n, int t, int peer, ncclComm_t c, void *st)
{
    (void)s; (void)peer; (void)c;
    stub_gpu_work(st, COLLECTIVE_NS + (double)n * type_size(t) / STUB_NVLINK_B_PER_NS);
    stub_nvlink_traffic(st, (double)n * type_size(t), 0);
    return ncclSuccess;
}

/* A receive gets what this rank would have sent: nothing better to offer. */
API ncclResult_t ncclRecv(void *r, size_t n, int t, int peer, ncclComm_t c, void *st)
{
    (void)r; (void)peer; (void)c;
    stub_gpu_work(st, COLLECTIVE_NS + (double)n * type_size(t) / STUB_NVLINK_B_PER_NS);
    stub_nvlink_traffic(st, 0, (double)n * type_size(t));
    return ncclSuccess;
}
