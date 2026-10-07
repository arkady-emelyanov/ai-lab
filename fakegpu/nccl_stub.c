/*
 * Fake NCCL (libnccl.so.2). Every exported symbol returns ncclSuccess
 * (generated from symbols/nccl.syms). Communicators remember their size,
 * rank and device so frameworks can query them. Collectives take simulated
 * time on their stream and behave as if every peer contributed exactly what
 * this rank did (device memory is host memory under the fake driver), so
 * cross-rank consistency checks such as DDP's parameter verification pass.
 *
 * Topology: like real NCCL at init, every rank learns which NVLink partition
 * each peer's GPU is in (NVML fabric info: cluster UUID and clique). Ranks
 * share that through files in nccl_dir (/shared in the lab), keyed by the
 * communicator's unique id; ncclCommSplit, which is collective, exchanges
 * colours the same way. Collectives within one partition run at NVLink
 * speed; across partitions they are hierarchical, NVLink inside each one and
 * InfiniBand (one NIC per GPU) between them. Without nccl_dir, or when the
 * peers do not all report in time, every rank is assumed to be in one
 * partition: NCCL calls never fail.
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include "fakegpu.h"
#include "stub_common.h"

typedef int ncclResult_t;
#define ncclSuccess 0
#define ncclInvalidArgument 4

#define NCCL_VERSION_CODE 23007 /* 2.30.7 */

typedef struct { char internal[128]; } ncclUniqueId;

struct fake_comm {
    int nranks, rank, dev;
    char key[64];       /* exchange name ("" when the topology is not known) */
    int splits;         /* ncclCommSplit calls so far: same order on every rank */
    char part[128];     /* this rank's NVLink partition */
    int groups;         /* NVLink partitions the ranks span (1: one) */
    int min_group;      /* ranks in the smallest of them */
    int *group;         /* partition index of each rank; NULL when groups == 1 */
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
    c->groups = c->min_group = 1;
    *out = c;
    return ncclSuccess;
}

/* ---- topology: NVLink partition of every rank --------------------------- */

typedef struct { unsigned char clusterUuid[16]; int status; unsigned cliqueId; unsigned char state; } fabric_info_t;

/* The NVLink partition of CUDA device dev, found as real NCCL does: its PCI
 * bus id from CUDA, then NVML fabric info. A GPU in no partition (clique 0)
 * reaches no peer over NVLink, so it is a partition of its own. If NVML
 * cannot tell, "nvlink": every such rank counts as one partition. */
static void partition_of(int dev, char *out, size_t len)
{
    snprintf(out, len, "nvlink");
    int (*bus_id)(char *, int, int) = (int (*)(char *, int, int))stub_libcuda_sym("cuDeviceGetPCIBusId");
    void *nvml = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
    if (!bus_id || !nvml) return;
    int (*init)(void) = (int (*)(void))dlsym(nvml, "nvmlInit_v2");
    int (*by_bus)(const char *, void **) = (int (*)(const char *, void **))dlsym(nvml, "nvmlDeviceGetHandleByPciBusId_v2");
    int (*fabric)(void *, fabric_info_t *) = (int (*)(void *, fabric_info_t *))dlsym(nvml, "nvmlDeviceGetGpuFabricInfo");
    char bus[32];
    void *h;
    fabric_info_t f = {0};
    if (!init || !by_bus || !fabric || bus_id(bus, sizeof bus, dev) || init() || by_bus(bus, &h) || fabric(h, &f))
        return;
    if (f.cliqueId == 0) { /* this GPU alone: its tray (as configured, also in pods) and bus */
        char host[128] = "";
        fg_init();
        if (fg_cfg.host[0]) snprintf(host, sizeof host, "%s", fg_cfg.host);
        else gethostname(host, sizeof host - 1);
        snprintf(out, len, "none.%s.%s", host, bus);
        return;
    }
    int n = 0;
    for (int i = 0; i < 16; i++) n += snprintf(out + n, len - n, "%02x", f.clusterUuid[i]);
    snprintf(out + n, len - n, ".%u", f.cliqueId);
}

static double mono_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec / 1e9;
}

/* Removes exchange directories nobody cleaned up (ranks that crashed). */
static void sweep_stale(void)
{
    DIR *d = opendir(fg_cfg.nccl_dir);
    if (!d) return;
    time_t old = time(NULL) - 86400;
    struct dirent *e;
    while ((e = readdir(d))) {
        char dir[512];
        struct stat st;
        if (e->d_name[0] == '.') continue;
        snprintf(dir, sizeof dir, "%s/%s", fg_cfg.nccl_dir, e->d_name);
        if (stat(dir, &st) || !S_ISDIR(st.st_mode) || st.st_mtime > old) continue;
        DIR *in = opendir(dir);
        struct dirent *f;
        while (in && (f = readdir(in))) {
            char path[800];
            snprintf(path, sizeof path, "%s/%s", dir, f->d_name);
            if (strcmp(f->d_name, ".") && strcmp(f->d_name, "..")) unlink(path);
        }
        if (in) closedir(in);
        rmdir(dir);
    }
    closedir(d);
}

/* Publishes this rank's line in <nccl_dir>/<name>/<me> and reads all n
 * ranks' lines into lines[n][128]. 0 when all arrived in time. */
static int exchange(const char *name, int n, int me, const char *line, char (*lines)[128])
{
    fg_init();
    if (!fg_cfg.nccl_dir[0]) return -1;
    char dir[512], path[600], tmp[620];
    snprintf(dir, sizeof dir, "%s/%s", fg_cfg.nccl_dir, name);
    if (mkdir(dir, 0700) && errno != EEXIST) return -1;
    snprintf(path, sizeof path, "%s/%d", dir, me);
    snprintf(tmp, sizeof tmp, "%s/.%d.tmp", dir, me);
    FILE *f = fopen(tmp, "w");
    if (!f) return -1;
    fprintf(f, "%s\n", line);
    if (fclose(f) || rename(tmp, path)) return -1;
    double deadline = mono_s() + fg_cfg.nccl_timeout_s;
    for (;;) {
        int got = 0;
        for (int r = 0; r < n; r++) {
            if (!lines[r][0]) {
                snprintf(path, sizeof path, "%s/%d", dir, r);
                FILE *g = fopen(path, "r");
                if (g) {
                    if (fgets(lines[r], 128, g)) lines[r][strcspn(lines[r], "\n")] = 0;
                    fclose(g);
                }
            }
            got += lines[r][0] != 0;
        }
        if (got == n) {
            /* Read everything: say so (see cleanup). */
            snprintf(path, sizeof path, "%s/done.%d", dir, me);
            FILE *d = fopen(path, "w");
            if (d) fclose(d);
            return 0;
        }
        if (mono_s() > deadline) return -1;
        usleep(20000);
    }
}

/* Removes exchange name once all n ranks have read it (their done.<rank>
 * markers exist); until then a slower peer may still need the entries. */
static void cleanup(const char *name, int n)
{
    char path[600];
    for (int r = 0; r < n; r++) {
        snprintf(path, sizeof path, "%s/%s/done.%d", fg_cfg.nccl_dir, name, r);
        if (access(path, F_OK)) return;
    }
    for (int r = 0; r < n; r++) {
        snprintf(path, sizeof path, "%s/%s/%d", fg_cfg.nccl_dir, name, r);
        unlink(path);
        snprintf(path, sizeof path, "%s/%s/done.%d", fg_cfg.nccl_dir, name, r);
        unlink(path);
    }
    snprintf(path, sizeof path, "%s/%s", fg_cfg.nccl_dir, name);
    rmdir(path);
}

/* Groups ranks by partition: c->groups, c->min_group, c->group. */
static void set_groups(struct fake_comm *c, char (*parts)[128], int n)
{
    int *group = calloc((size_t)n, sizeof *group), *size = calloc((size_t)n, sizeof *size), groups = 0;
    for (int r = 0; r < n; r++) {
        int g = 0;
        while (g < r && strcmp(parts[g], parts[r])) g++;
        group[r] = g < r ? group[g] : groups++;
        size[group[r]]++;
    }
    int min = n;
    for (int g = 0; g < groups; g++) if (size[g] < min) min = size[g];
    free(size);
    free(c->group);
    c->groups = groups;
    c->min_group = min;
    c->group = groups > 1 ? group : (free(group), NULL);
}

static void warn_unknown(const char *what)
{
    static int warned;
    if (fg_cfg.nccl_dir[0] && !warned++)
        fprintf(stderr, "fakegpu NCCL: %s: not every rank reported its NVLink partition within %gs; "
                        "assuming one partition (NVLink speed)\n", what, fg_cfg.nccl_timeout_s);
}

static void discover(struct fake_comm *c, const ncclUniqueId *id)
{
    partition_of(c->dev, c->part, sizeof c->part);
    if (c->nranks == 1 || !id) return;
    char key[33];
    for (int i = 0; i < 16; i++) snprintf(key + 2 * i, 3, "%02x", (unsigned char)id->internal[i]);
    char (*parts)[128] = calloc((size_t)c->nranks, sizeof *parts);
    if (exchange(key, c->nranks, c->rank, c->part, parts) == 0) {
        snprintf(c->key, sizeof c->key, "%s", key);
        set_groups(c, parts, c->nranks);
        if (c->rank == 0) sweep_stale();
    } else {
        warn_unknown("ncclCommInitRank");
    }
    free(parts);
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

/* 16 random bytes: the communicator's name for the topology exchange. */
API ncclResult_t ncclGetUniqueId(ncclUniqueId *id)
{
    if (!id) return ncclInvalidArgument;
    memset(id, 0, sizeof *id);
    if (getrandom(id->internal, 16, 0) != 16) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        snprintf(id->internal, 16, "%08lx%07x", (unsigned long)ts.tv_nsec ^ (unsigned long)getpid(), (unsigned)ts.tv_sec);
    }
    snprintf(id->internal + 16, sizeof id->internal - 16, "fakegpu-nccl");
    return ncclSuccess;
}

API ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId id, int rank)
{
    ncclResult_t r = new_comm(comm, nranks, rank);
    if (r == ncclSuccess) discover(*comm, &id);
    return r;
}

API ncclResult_t ncclCommInitRankConfig(ncclComm_t *comm, int nranks, ncclUniqueId id, int rank, void *config)
{
    (void)config;
    return ncclCommInitRank(comm, nranks, id, rank);
}

API ncclResult_t ncclCommInitRankScalable(ncclComm_t *comm, int nranks, int rank, int nid, ncclUniqueId *ids, void *config)
{
    (void)config;
    ncclResult_t r = new_comm(comm, nranks, rank);
    if (r == ncclSuccess) discover(*comm, nid > 0 ? ids : NULL);
    return r;
}

/* One process, all devices: their partitions are known locally. */
API ncclResult_t ncclCommInitAll(ncclComm_t *comms, int ndev, const int *devlist)
{
    if (!comms || ndev < 1) return ncclInvalidArgument;
    char (*parts)[128] = calloc((size_t)ndev, sizeof *parts);
    for (int i = 0; i < ndev; i++) {
        new_comm(&comms[i], ndev, i);
        comms[i]->dev = devlist ? devlist[i] : i;
        partition_of(comms[i]->dev, comms[i]->part, sizeof comms[i]->part);
        snprintf(parts[i], sizeof parts[i], "%s", comms[i]->part);
    }
    for (int i = 0; i < ndev; i++) set_groups(comms[i], parts, ndev);
    free(parts);
    return ncclSuccess;
}

/* Copies a parent's partition count when a child's members are unknown. */
static void inherit(struct fake_comm *child, const struct fake_comm *parent)
{
    if (!parent) return;
    snprintf(child->part, sizeof child->part, "%s", parent->part);
    child->groups = parent->groups;
    child->min_group = parent->min_group;
}

/* Collective over the parent: every rank publishes its colour and key, so
 * each one knows its new communicator's members and their partitions. */
API ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *out, void *config)
{
    (void)config;
    if (!out) return ncclInvalidArgument;
    if (comm && comm->key[0] && comm->nranks > 1) {
        char name[64], line[160];
        snprintf(name, sizeof name, "%.40s.s%d", comm->key, comm->splits++);
        snprintf(line, sizeof line, "%d %d %s", color, key, comm->part);
        char (*lines)[128] = calloc((size_t)comm->nranks, sizeof *lines);
        if (exchange(name, comm->nranks, comm->rank, line, lines) == 0) {
            *out = NULL;
            if (color >= 0) {
                /* Members: same colour, ordered by key, then parent rank. */
                int n = 0, me = 0, *rank = calloc((size_t)comm->nranks, sizeof *rank);
                int *keys = calloc((size_t)comm->nranks, sizeof *keys);
                char (*parts)[128] = calloc((size_t)comm->nranks, sizeof *parts);
                for (int r = 0; r < comm->nranks; r++) {
                    int col, k;
                    char part[128];
                    if (sscanf(lines[r], "%d %d %127s", &col, &k, part) != 3 || col != color) continue;
                    int at = n++;
                    while (at > 0 && (keys[at - 1] > k)) {
                        rank[at] = rank[at - 1], keys[at] = keys[at - 1];
                        memcpy(parts[at], parts[at - 1], sizeof parts[at]);
                        at--;
                    }
                    rank[at] = r, keys[at] = k;
                    snprintf(parts[at], sizeof parts[at], "%s", part);
                }
                for (int i = 0; i < n; i++) if (rank[i] == comm->rank) me = i;
                new_comm(out, n ? n : 1, me);
                snprintf((*out)->key, sizeof (*out)->key, "%.40s.c%d", name, color);
                snprintf((*out)->part, sizeof (*out)->part, "%s", comm->part);
                set_groups(*out, parts, n ? n : 1);
                free(rank); free(keys); free(parts);
            }
            cleanup(name, comm->nranks);
            free(lines);
            return ncclSuccess;
        }
        free(lines);
        warn_unknown("ncclCommSplit");
    }
    /* Members unknown: keep the caller's rank and the parent's size. */
    if (color < 0) { *out = NULL; return ncclSuccess; } /* NCCL_SPLIT_NOCOLOR */
    ncclResult_t r = new_comm(out, comm ? comm->nranks : 1, comm ? comm->rank : 0);
    if (r == ncclSuccess) inherit(*out, comm);
    return r;
}

/* The parent's ranks without excl, renumbered in order. */
API ncclResult_t ncclCommShrink(ncclComm_t comm, int *excl, int nexcl, ncclComm_t *out, void *config, int flags)
{
    (void)config; (void)flags;
    int n = comm && comm->nranks - nexcl > 0 ? comm->nranks - nexcl : 1;
    int r = comm && comm->rank < n ? comm->rank : 0;
    if (comm && excl) {
        r = comm->rank;
        for (int i = 0; i < nexcl; i++) if (excl[i] < comm->rank) r--;
    }
    ncclResult_t res = new_comm(out, n, r < 0 ? 0 : r);
    if (res != ncclSuccess) return res;
    inherit(*out, comm);
    if (comm && comm->group && excl) {
        char (*parts)[128] = calloc((size_t)n, sizeof *parts);
        for (int p = 0, i = 0; p < comm->nranks && i < n; p++) {
            int out_ = 0;
            for (int x = 0; x < nexcl; x++) out_ |= excl[x] == p;
            if (!out_) snprintf(parts[i++], sizeof parts[0], "%d", comm->group[p]);
        }
        set_groups(*out, parts, n);
        free(parts);
    }
    return res;
}

API ncclResult_t ncclCommGrow(ncclComm_t comm, int nranks, const ncclUniqueId *id, int rank, ncclComm_t *out, void *config)
{
    (void)id; (void)config;
    ncclResult_t r = new_comm(out, nranks, rank);
    if (r == ncclSuccess) inherit(*out, comm);
    return r;
}

static void free_comm(ncclComm_t c)
{
    if (!c) return;
    if (c->key[0]) cleanup(c->key, c->nranks);
    free(c->group);
    free(c);
}

API ncclResult_t ncclCommDestroy(ncclComm_t c) { free_comm(c); return ncclSuccess; }
API ncclResult_t ncclCommAbort(ncclComm_t c) { free_comm(c); return ncclSuccess; }

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

#define RING 0
#define ROOTED 1

/* Bus factor: bytes each of n ranks moves per byte of payload: k(n-1)/n for
 * rings (all-reduce k = 2, all-gather and reduce-scatter k = 1), 1 for
 * rooted collectives (broadcast, reduce). */
static double bus(int n, double k, int kind)
{
    return n > 1 ? (kind == ROOTED ? 1 : k * (n - 1) / n) : 0;
}

static double ib_bytes_per_ns(void) { fg_init(); return fg_cfg.ib_gbps / 8; }

/* Within one NVLink partition: a ring over NVLink. Across partitions, as
 * NCCL does across nodes: the same collective inside each partition over
 * NVLink, and between the partitions over InfiniBand, where each partition
 * moves its share through its GPUs' NICs (the smallest partition limits). */
static void collective_work(void *stream, ncclComm_t c, double bytes, double k, int kind)
{
    int n = c ? c->nranks : 1;
    if (!c || c->groups <= 1) {
        double nvlink = bytes * bus(n, k, kind);
        stub_gpu_work(stream, COLLECTIVE_NS + nvlink / STUB_NVLINK_B_PER_NS);
        stub_nvlink_traffic(stream, nvlink, nvlink);
        return;
    }
    double nvlink = bytes * bus(c->min_group, k, kind);
    double ib = bytes * bus(c->groups, k, kind) / c->min_group;
    stub_gpu_work(stream, 2 * COLLECTIVE_NS + nvlink / STUB_NVLINK_B_PER_NS + ib / ib_bytes_per_ns());
    stub_nvlink_traffic(stream, nvlink, nvlink);
    stub_ib_traffic(stream, ib, ib);
}

/* Point to point: NVLink to a peer in this rank's partition, else its NIC. */
static void p2p_work(void *stream, ncclComm_t c, int peer, double bytes, int send)
{
    int ib = c && c->group && peer >= 0 && peer < c->nranks && c->group[peer] != c->group[c->rank];
    stub_gpu_work(stream, COLLECTIVE_NS + bytes / (ib ? ib_bytes_per_ns() : STUB_NVLINK_B_PER_NS));
    if (ib) stub_ib_traffic(stream, send ? bytes : 0, send ? 0 : bytes);
    else stub_nvlink_traffic(stream, send ? bytes : 0, send ? 0 : bytes);
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
    collective_work(st, c, (double)n * type_size(t), 2, RING);
    return ncclSuccess;
}

API ncclResult_t ncclReduce(const void *s, void *r, size_t n, int t, int op, int root, ncclComm_t c, void *st)
{
    (void)op; (void)root;
    copy(r, s, n, t);
    collective_work(st, c, (double)n * type_size(t), 1, ROOTED);
    return ncclSuccess;
}

API ncclResult_t ncclBroadcast(const void *s, void *r, size_t n, int t, int root, ncclComm_t c, void *st)
{
    (void)root;
    copy(r, s, n, t);
    collective_work(st, c, (double)n * type_size(t), 1, ROOTED);
    return ncclSuccess;
}

API ncclResult_t ncclReduceScatter(const void *s, void *r, size_t n, int t, int op, ncclComm_t c, void *st)
{
    (void)op;
    if (s && r) copy(r, (const char *)s + (size_t)(c ? c->rank : 0) * n * type_size(t), n, t);
    collective_work(st, c, (double)n * type_size(t) * (c ? c->nranks : 1), 1, RING);
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
    collective_work(st, c, (double)chunk * ranks, 1, RING);
    return ncclSuccess;
}

API ncclResult_t ncclAlltoAll(const void *s, void *r, size_t n, int t, ncclComm_t c, void *st)
{
    copy(r, s, n * (size_t)(c ? c->nranks : 1), t);
    collective_work(st, c, (double)n * type_size(t) * (c ? c->nranks : 1), 1, RING);
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
    collective_work(st, c, (double)n * type_size(t), 1, ROOTED);
    return ncclSuccess;
}

API ncclResult_t ncclSend(const void *s, size_t n, int t, int peer, ncclComm_t c, void *st)
{
    (void)s;
    p2p_work(st, c, peer, (double)n * type_size(t), 1);
    return ncclSuccess;
}

/* A receive gets what this rank would have sent: nothing better to offer. */
API ncclResult_t ncclRecv(void *r, size_t n, int t, int peer, ncclComm_t c, void *st)
{
    (void)r;
    p2p_work(st, c, peer, (double)n * type_size(t), 0);
    return ncclSuccess;
}
