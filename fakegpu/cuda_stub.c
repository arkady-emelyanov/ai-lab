/*
 * Fake CUDA driver (libcuda.so.1).
 *
 * Every exported cu* symbol exists (see stubs generated from
 * symbols/cuda.syms) and returns CUDA_SUCCESS without doing anything.
 * The functions below are the ones callers read results from, so they
 * return a consistent fake device. "Device memory" is host memory, so
 * copies round-trip correctly; kernel launches are no-ops.
 *
 * CUDA_VISIBLE_DEVICES is honoured (indices or GPU- UUIDs), like the real
 * driver, so Slurm GPU allocation is reflected in what a job sees.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <sys/file.h>
#include <sys/random.h>
#include "occupancy.h"

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef void *CUcontext;
typedef void *CUstream;
typedef struct { unsigned char bytes[16]; } CUuuid;

#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_VALUE 1
#define CUDA_ERROR_OUT_OF_MEMORY 2
#define CUDA_ERROR_NO_DEVICE 100
#define CUDA_ERROR_INVALID_DEVICE 101
#define CUDA_ERROR_NOT_FOUND 500
#define CUDA_ERROR_NOT_SUPPORTED 801

#define API __attribute__((visibility("default")))

/* ---- device enumeration ------------------------------------------------ */

static int vis_map[FG_MAX_GPUS]; /* CUDA ordinal -> physical index */
static int vis_count;
static pthread_once_t vis_once = PTHREAD_ONCE_INIT;

static int match_uuid(const char *tok, size_t len)
{
    char u[64];
    for (int i = 0; i < fg_count(); i++) {
        fg_uuid_str(fg_phys(i), u, sizeof u);
        if (len >= 8 && strncasecmp(u, tok, len) == 0) return i;
    }
    return -1;
}

static void vis_init(void)
{
    int n = fg_count();
    const char *env = getenv("CUDA_VISIBLE_DEVICES");
    if (!env) {
        for (int i = 0; i < n; i++) vis_map[vis_count++] = fg_phys(i);
        return;
    }
    /* Like the real driver: stop at the first invalid or duplicate entry. */
    const char *p = env;
    while (*p && vis_count < n) {
        while (*p == ' ') p++;
        size_t len = strcspn(p, ",");
        int idx = -1;
        if (len && strncmp(p, "GPU-", 4) == 0) {
            idx = match_uuid(p, len);
        } else if (len) {
            char *end;
            long v = strtol(p, &end, 10);
            if (end == p + len && v >= 0 && v < n) idx = (int)v;
        }
        if (idx < 0) break;
        idx = fg_phys(idx); /* entries count among the GPUs present */
        for (int i = 0; i < vis_count; i++)
            if (vis_map[i] == idx) idx = -1;
        if (idx < 0) break;
        vis_map[vis_count++] = idx;
        p += len;
        if (*p == ',') p++;
    }
}

static int fg_debug(void)
{
    static int d = -1;
    if (d < 0) {
        const char *e = getenv("FAKEGPU_DEBUG");
        d = e && *e && *e != '0';
    }
    return d;
}

static int phys_at(CUdevice dev, const char *fn)
{
    pthread_once(&vis_once, vis_init);
    if (dev >= 0 && dev < vis_count) return vis_map[dev];
    if (fg_debug()) fprintf(stderr, "[fakegpu] %s: invalid device %d\n", fn, dev);
    return -1;
}
#define phys(dev) phys_at(dev, __func__)

/* Contexts are just tagged allocations remembering their device. */
struct fake_ctx { unsigned magic; CUdevice dev; };
#define CTX_MAGIC 0xFA6E6C7Bu
static struct fake_ctx primary[FG_MAX_GPUS];
static __thread CUcontext cur_ctx;

static CUcontext primary_ctx(CUdevice dev)
{
    primary[dev].magic = CTX_MAGIC;
    primary[dev].dev = dev;
    return &primary[dev];
}

void *fg_fake_handle(void) { return calloc(1, 64); }
#define fake_handle fg_fake_handle

/* Physical index of the current context's GPU (timing.c, occupancy). */
int fg_current_phys(void)
{
    struct fake_ctx *c = cur_ctx;
    int p = phys_at(c && c->magic == CTX_MAGIC ? c->dev : 0, "current context");
    return p < 0 ? 0 : p;
}

/* ---- init / version ---------------------------------------------------- */

API CUresult cuInit(unsigned flags) { (void)flags; return CUDA_SUCCESS; }

API CUresult cuDriverGetVersion(int *v)
{
    if (!v) return CUDA_ERROR_INVALID_VALUE;
    *v = fg_cuda_version();
    return CUDA_SUCCESS;
}

static const char *err_str(CUresult e)
{
    switch (e) {
    case CUDA_SUCCESS: return "no error";
    case CUDA_ERROR_INVALID_VALUE: return "invalid argument";
    case CUDA_ERROR_OUT_OF_MEMORY: return "out of memory";
    case CUDA_ERROR_NO_DEVICE: return "no CUDA-capable device is detected";
    case CUDA_ERROR_INVALID_DEVICE: return "invalid device ordinal";
    case CUDA_ERROR_NOT_FOUND: return "named symbol not found";
    case CUDA_ERROR_NOT_SUPPORTED: return "operation not supported";
    default: return "unknown error";
    }
}

API CUresult cuGetErrorString(CUresult e, const char **s)
{
    if (!s) return CUDA_ERROR_INVALID_VALUE;
    *s = err_str(e);
    return CUDA_SUCCESS;
}

API CUresult cuGetErrorName(CUresult e, const char **s)
{
    if (!s) return CUDA_ERROR_INVALID_VALUE;
    switch (e) {
    case CUDA_SUCCESS: *s = "CUDA_SUCCESS"; break;
    case CUDA_ERROR_INVALID_VALUE: *s = "CUDA_ERROR_INVALID_VALUE"; break;
    case CUDA_ERROR_OUT_OF_MEMORY: *s = "CUDA_ERROR_OUT_OF_MEMORY"; break;
    case CUDA_ERROR_NO_DEVICE: *s = "CUDA_ERROR_NO_DEVICE"; break;
    case CUDA_ERROR_INVALID_DEVICE: *s = "CUDA_ERROR_INVALID_DEVICE"; break;
    case CUDA_ERROR_NOT_FOUND: *s = "CUDA_ERROR_NOT_FOUND"; break;
    case CUDA_ERROR_NOT_SUPPORTED: *s = "CUDA_ERROR_NOT_SUPPORTED"; break;
    default: *s = "CUDA_ERROR_UNKNOWN";
    }
    return CUDA_SUCCESS;
}

/* ---- devices ----------------------------------------------------------- */

API CUresult cuDeviceGetCount(int *n)
{
    if (!n) return CUDA_ERROR_INVALID_VALUE;
    pthread_once(&vis_once, vis_init);
    *n = vis_count;
    return CUDA_SUCCESS;
}

API CUresult cuDeviceGet(CUdevice *dev, int ordinal)
{
    if (!dev) return CUDA_ERROR_INVALID_VALUE;
    if (phys(ordinal) < 0) return CUDA_ERROR_INVALID_DEVICE;
    *dev = ordinal;
    return CUDA_SUCCESS;
}

API CUresult cuDeviceGetName(char *name, int len, CUdevice dev)
{
    if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;
    if (phys(dev) < 0) return CUDA_ERROR_INVALID_DEVICE;
    snprintf(name, (size_t)len, "%s", fg_name());
    return CUDA_SUCCESS;
}

API CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev)
{
    if (!bytes) return CUDA_ERROR_INVALID_VALUE;
    if (phys(dev) < 0) return CUDA_ERROR_INVALID_DEVICE;
    *bytes = fg_mem_bytes();
    return CUDA_SUCCESS;
}

API CUresult cuDeviceTotalMem(unsigned *bytes, CUdevice dev)
{
    if (!bytes) return CUDA_ERROR_INVALID_VALUE;
    if (phys(dev) < 0) return CUDA_ERROR_INVALID_DEVICE;
    *bytes = 0xFFFFFFFFu;
    return CUDA_SUCCESS;
}

API CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev)
{
    int p = phys(dev);
    if (!uuid) return CUDA_ERROR_INVALID_VALUE;
    if (p < 0) return CUDA_ERROR_INVALID_DEVICE;
    fg_uuid(p, uuid->bytes);
    return CUDA_SUCCESS;
}

API CUresult cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev) { return cuDeviceGetUuid(uuid, dev); }

API CUresult cuDeviceGetLuid(char *luid, unsigned *mask, CUdevice dev)
{
    (void)luid; (void)mask; (void)dev;
    return CUDA_ERROR_NOT_SUPPORTED; /* Windows only */
}

API CUresult cuDeviceGetPCIBusId(char *buf, int len, CUdevice dev)
{
    int p = phys(dev);
    if (!buf || len <= 0) return CUDA_ERROR_INVALID_VALUE;
    if (p < 0) return CUDA_ERROR_INVALID_DEVICE;
    fg_pci_str(p, buf, (size_t)len, 0);
    return CUDA_SUCCESS;
}

API CUresult cuDeviceGetByPCIBusId(CUdevice *dev, const char *bus)
{
    char b[32];
    pthread_once(&vis_once, vis_init);
    for (int d = 0; d < vis_count; d++) {
        fg_pci_str(vis_map[d], b, sizeof b, 0);
        if (strcasecmp(b, bus) == 0 || strcasestr(b, bus)) { *dev = d; return CUDA_SUCCESS; }
    }
    return CUDA_ERROR_INVALID_DEVICE;
}

static int attribute(int attr, int p)
{
    switch (attr) {
    case 1: return 1024;                     /* MAX_THREADS_PER_BLOCK */
    case 2: case 3: return 1024;             /* MAX_BLOCK_DIM_X/Y */
    case 4: return 64;                       /* MAX_BLOCK_DIM_Z */
    case 5: return 2147483647;               /* MAX_GRID_DIM_X */
    case 6: case 7: return 65535;            /* MAX_GRID_DIM_Y/Z */
    case 8: return 49152;                    /* MAX_SHARED_MEMORY_PER_BLOCK */
    case 9: return 65536;                    /* TOTAL_CONSTANT_MEMORY */
    case 10: return 32;                      /* WARP_SIZE */
    case 11: return 2147483647;              /* MAX_PITCH */
    case 12: return 65536;                   /* MAX_REGISTERS_PER_BLOCK */
    case 13: return 1965000;                 /* CLOCK_RATE kHz */
    case 14: return 512;                     /* TEXTURE_ALIGNMENT */
    case 15: return 1;                       /* GPU_OVERLAP */
    case 16: return FG_SM_COUNT;             /* MULTIPROCESSOR_COUNT */
    case 17: return 0;                       /* KERNEL_EXEC_TIMEOUT */
    case 18: return 0;                       /* INTEGRATED */
    case 19: return 1;                       /* CAN_MAP_HOST_MEMORY */
    case 20: return 0;                       /* COMPUTE_MODE default */
    case 31: return 1;                       /* CONCURRENT_KERNELS */
    case 32: return 1;                       /* ECC_ENABLED */
    case 33: return FG_PCI_BUS;              /* PCI_BUS_ID */
    case 34: return 0;                       /* PCI_DEVICE_ID */
    case 35: return 0;                       /* TCC_DRIVER */
    case 36: return 3996000;                 /* MEMORY_CLOCK_RATE kHz */
    case 37: return 8192;                    /* GLOBAL_MEMORY_BUS_WIDTH */
    case 38: return 126 << 20;               /* L2_CACHE_SIZE */
    case 39: return 2048;                    /* MAX_THREADS_PER_MULTIPROCESSOR */
    case 40: return 4;                       /* ASYNC_ENGINE_COUNT */
    case 41: return 1;                       /* UNIFIED_ADDRESSING */
    case 50: return (int)fg_pci_domain[p];   /* PCI_DOMAIN_ID */
    case 75: return FG_CC_MAJOR;             /* COMPUTE_CAPABILITY_MAJOR */
    case 76: return FG_CC_MINOR;             /* COMPUTE_CAPABILITY_MINOR */
    case 81: return 233472;                  /* MAX_SHARED_MEMORY_PER_MULTIPROCESSOR */
    case 82: return 65536;                   /* MAX_REGISTERS_PER_MULTIPROCESSOR */
    case 83: return 1;                       /* MANAGED_MEMORY */
    case 84: return 0;                       /* MULTI_GPU_BOARD */
    case 86: return 1;                       /* HOST_NATIVE_ATOMIC_SUPPORTED */
    case 89: return 1;                       /* CONCURRENT_MANAGED_ACCESS */
    case 95: return 1;                       /* COOPERATIVE_LAUNCH */
    case 97: return 232448;                  /* MAX_SHARED_MEMORY_PER_BLOCK_OPTIN */
    case 106: return 32;                     /* MAX_BLOCKS_PER_MULTIPROCESSOR */
    case 115: return 1;                      /* MEMORY_POOLS_SUPPORTED */
    case 102: return 1;                      /* VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED (vmm.c) */
    case 103: return 1;                      /* HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED */
    case 110: return 1;                      /* GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED */
    case 128: fg_init(); return fg_cfg.fabric_dir[0] != 0; /* HANDLE_TYPE_FABRIC_SUPPORTED */
    case 132: return 1;                      /* MULTICAST_SUPPORTED (NVSwitch) */
    default: return 0;
    }
}

API CUresult cuDeviceGetAttribute(int *val, int attr, CUdevice dev)
{
    int p = phys(dev);
    if (!val) return CUDA_ERROR_INVALID_VALUE;
    if (p < 0) return CUDA_ERROR_INVALID_DEVICE;
    *val = attribute(attr, p);
    return CUDA_SUCCESS;
}

API CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev)
{
    if (phys(dev) < 0) return CUDA_ERROR_INVALID_DEVICE;
    if (major) *major = FG_CC_MAJOR;
    if (minor) *minor = FG_CC_MINOR;
    return CUDA_SUCCESS;
}

/* All GPUs in the box are NVLinked to each other. */
API CUresult cuDeviceCanAccessPeer(int *can, CUdevice a, CUdevice b)
{
    if (!can) return CUDA_ERROR_INVALID_VALUE;
    if (phys(a) < 0 || phys(b) < 0) return CUDA_ERROR_INVALID_DEVICE;
    *can = a != b;
    return CUDA_SUCCESS;
}

API CUresult cuDeviceGetP2PAttribute(int *val, int attr, CUdevice a, CUdevice b)
{
    if (!val) return CUDA_ERROR_INVALID_VALUE;
    if (phys(a) < 0 || phys(b) < 0) return CUDA_ERROR_INVALID_DEVICE;
    *val = attr == 1 ? 0 : 1; /* PERFORMANCE_RANK 0, everything else supported */
    return CUDA_SUCCESS;
}

/* ---- contexts ---------------------------------------------------------- */

API CUresult cuDevicePrimaryCtxRetain(CUcontext *ctx, CUdevice dev)
{
    if (!ctx) return CUDA_ERROR_INVALID_VALUE;
    if (phys(dev) < 0) return CUDA_ERROR_INVALID_DEVICE;
    *ctx = primary_ctx(dev);
    return CUDA_SUCCESS;
}

API CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned *flags, int *active)
{
    if (phys(dev) < 0) return CUDA_ERROR_INVALID_DEVICE;
    if (flags) *flags = 0;
    if (active) *active = primary[dev].magic == CTX_MAGIC;
    return CUDA_SUCCESS;
}

API CUresult cuCtxCreate_v2(CUcontext *ctx, unsigned flags, CUdevice dev)
{
    (void)flags;
    if (!ctx) return CUDA_ERROR_INVALID_VALUE;
    if (phys(dev) < 0) return CUDA_ERROR_INVALID_DEVICE;
    struct fake_ctx *c = calloc(1, sizeof *c);
    c->magic = CTX_MAGIC;
    c->dev = dev;
    *ctx = cur_ctx = c;
    return CUDA_SUCCESS;
}

API CUresult cuCtxCreate(CUcontext *ctx, unsigned flags, CUdevice dev) { return cuCtxCreate_v2(ctx, flags, dev); }

API CUresult cuCtxGetCurrent(CUcontext *ctx)
{
    if (!ctx) return CUDA_ERROR_INVALID_VALUE;
    *ctx = cur_ctx;
    return CUDA_SUCCESS;
}

API CUresult cuCtxSetCurrent(CUcontext ctx) { cur_ctx = ctx; return CUDA_SUCCESS; }
API CUresult cuCtxPushCurrent_v2(CUcontext ctx) { cur_ctx = ctx; return CUDA_SUCCESS; }

API CUresult cuCtxPopCurrent_v2(CUcontext *ctx)
{
    if (ctx) *ctx = cur_ctx;
    return CUDA_SUCCESS;
}

API CUresult cuCtxGetDevice(CUdevice *dev)
{
    if (!dev) return CUDA_ERROR_INVALID_VALUE;
    struct fake_ctx *c = cur_ctx;
    *dev = c && c->magic == CTX_MAGIC ? c->dev : 0;
    return CUDA_SUCCESS;
}

API CUresult cuCtxGetDevice_v2(CUdevice *dev, CUcontext ctx)
{
    if (!dev) return CUDA_ERROR_INVALID_VALUE;
    struct fake_ctx *c = ctx ? ctx : cur_ctx;
    *dev = c && c->magic == CTX_MAGIC ? c->dev : 0;
    return CUDA_SUCCESS;
}

API CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned *v)
{
    (void)ctx;
    if (v) *v = 3020;
    return CUDA_SUCCESS;
}

API CUresult cuCtxGetStreamPriorityRange(int *least, int *greatest)
{
    if (least) *least = 0;
    if (greatest) *greatest = -5;
    return CUDA_SUCCESS;
}

API CUresult cuCtxGetLimit(size_t *v, int limit)
{
    (void)limit;
    if (v) *v = 1024;
    return CUDA_SUCCESS;
}

API CUresult cuCtxGetFlags(unsigned *flags)
{
    if (flags) *flags = 0;
    return CUDA_SUCCESS;
}

/* ---- memory: device memory is host memory ------------------------------ */

API CUresult cuMemGetInfo_v2(size_t *free_b, size_t *total_b)
{
    if (free_b) *free_b = fg_mem_bytes() - (512ULL << 20);
    if (total_b) *total_b = fg_mem_bytes();
    return CUDA_SUCCESS;
}

/* Allocations are tracked so pointer queries (range, owning device, memory
 * type) answer the way the real driver would, and device allocations are
 * accounted per process and GPU in the tray's occupancy state. Device
 * buffers are lazily backed, page-aligned mmaps: untouched "GPU memory"
 * costs no host RAM, and any of them can be shared through CUDA IPC. */
enum { MEM_DEVICE = 2, MEM_HOST = 1 };
enum {
    MAPPED_NO = 0,   /* calloc'ed (host memory) */
    MAPPED_OWN = 1,  /* our mmap */
    MAPPED_VMM = 2,  /* a cuMemMap'ed range: vmm.c owns its pages */
    MAPPED_IPC = 3,  /* another process' allocation, opened through CUDA IPC */
};
struct alloc_rec {
    uintptr_t base;
    size_t size;
    int type, dev, phys, mapped;
    unsigned long long id;
    struct alloc_rec *next;
    int ipc_fd; /* MAPPED_OWN shared through CUDA IPC: the file behind it, else -1 */
    char ipc_name[24];
};
static struct alloc_rec *allocs;
static unsigned long long alloc_seq;
static pthread_mutex_t alloc_lock = PTHREAD_MUTEX_INITIALIZER;

static int current_dev(void)
{
    struct fake_ctx *c = cur_ctx;
    return c && c->magic == CTX_MAGIC ? c->dev : 0;
}

static void occupancy_exit(void) { fg_proc_cleanup(1); }
static void atexit_register(void) { atexit(occupancy_exit); }

static void account(int phys, int64_t delta)
{
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, atexit_register);
    struct fg_proc *slot = fg_proc_slot(phys);
    if (slot) __atomic_add_fetch(&slot->mem, (uint64_t)delta, __ATOMIC_RELAXED);
}

static void *track(size_t n, int type)
{
    n = n ? n : 1;
    int phys = fg_current_phys();
    if (type == MEM_DEVICE) {
        fg_proc_cleanup(0);
        if (fg_gpu_mem_used(phys) + n > fg_mem_bytes()) return NULL; /* out of GPU memory */
    }
    int mapped = type == MEM_DEVICE ? MAPPED_OWN : MAPPED_NO;
    void *m = mapped ? mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0)
                     : calloc(1, n);
    if (m == MAP_FAILED || !m) return NULL;
    struct alloc_rec *r = malloc(sizeof *r);
    if (!r) {
        if (mapped) munmap(m, n); else free(m);
        return NULL;
    }
    pthread_mutex_lock(&alloc_lock);
    *r = (struct alloc_rec){.base = (uintptr_t)m, .size = n, .type = type, .dev = current_dev(), .phys = phys,
                            .mapped = mapped, .id = ++alloc_seq, .next = allocs, .ipc_fd = -1};
    allocs = r;
    pthread_mutex_unlock(&alloc_lock);
    if (type == MEM_DEVICE) account(phys, (int64_t)n);
    return m;
}

static void untrack(void *m)
{
    struct alloc_rec *dead = NULL;
    pthread_mutex_lock(&alloc_lock);
    for (struct alloc_rec **r = &allocs; *r; r = &(*r)->next) {
        if ((*r)->base == (uintptr_t)m) {
            dead = *r;
            *r = dead->next;
            break;
        }
    }
    pthread_mutex_unlock(&alloc_lock);
    if (!dead) return;
    if (dead->mapped == MAPPED_VMM) { /* vmm.c owns the pages and the accounting */
        free(dead);
        return;
    }
    if (dead->mapped == MAPPED_IPC) { /* another process' memory: only our mapping goes */
        munmap(m, dead->size);
        free(dead);
        return;
    }
    if (dead->type == MEM_DEVICE) account(dead->phys, -(int64_t)dead->size);
    if (dead->mapped) munmap(m, dead->size); else free(m);
    if (dead->ipc_fd >= 0) {
        char path[sizeof fg_cfg.ipc_dir + 32];
        snprintf(path, sizeof path, "%s/%s", fg_cfg.ipc_dir, dead->ipc_name);
        unlink(path);
        close(dead->ipc_fd);
    }
    free(dead);
}

/* Copies the record containing address p into *out. */
static int lookup(uintptr_t p, struct alloc_rec *out)
{
    int found = 0;
    pthread_mutex_lock(&alloc_lock);
    for (struct alloc_rec *r = allocs; r; r = r->next) {
        if (p >= r->base && p < r->base + r->size) { *out = *r; found = 1; break; }
    }
    pthread_mutex_unlock(&alloc_lock);
    return found;
}

/* For vmm.c: ranges mapped with cuMemMap are device memory to copies and
 * pointer queries. phys -1 marks memory of another GPU (imported). */
void fg_vmm_track(uintptr_t base, size_t n, int dev, int phys)
{
    struct alloc_rec *r = malloc(sizeof *r);
    if (!r) return;
    pthread_mutex_lock(&alloc_lock);
    *r = (struct alloc_rec){.base = base, .size = n, .type = MEM_DEVICE, .dev = dev, .phys = phys,
                            .mapped = MAPPED_VMM, .id = ++alloc_seq, .next = allocs, .ipc_fd = -1};
    allocs = r;
    pthread_mutex_unlock(&alloc_lock);
}

void fg_vmm_untrack(uintptr_t base) { untrack((void *)base); }
void fg_vmm_account(int phys, int64_t delta) { account(phys, delta); }
int fg_vmm_phys(CUdevice dev) { return phys(dev); }
int fg_vmm_current_dev(void) { return current_dev(); }

/* ---- legacy CUDA IPC (cudaIpcGetMemHandle / cudaIpcOpenMemHandle) ------ */

/* An allocation shared through IPC moves, in place, onto a file in ipc_dir:
 * the same address now maps shared pages, which any process on the tray
 * maps too, whatever container it runs in (ipc_dir is mounted into every
 * GPU pod), as with the real driver, which needs only the same node. A
 * process on another tray has no such file, so the handle does not open
 * there. The exporter holds a lock on the file until it frees the
 * allocation or exits; files nobody holds are swept (fg_collect_unheld).
 * CUipcMemHandle is 64 opaque bytes; ours: */
#define CUDA_ERROR_INVALID_CONTEXT 201
#define CUDA_ERROR_INVALID_HANDLE 400
typedef struct { char reserved[64]; } CUipcMemHandle;
struct ipc_handle {
    char magic[8];
    char name[24];
    uint64_t size, ino, devno;
    char pad[8];
};
_Static_assert(sizeof(struct ipc_handle) == sizeof(CUipcMemHandle), "CUipcMemHandle is 64 bytes");
static const char IPC_MAGIC[8] = "FGIPCMEM";
void fg_collect_unheld(const char *dir); /* vmm.c */

/* Copies the allocation's contents into fd, leaving zero chunks as holes so
 * a large, mostly untouched buffer stays sparse. */
static int fill_file(int fd, const char *base, size_t n)
{
    static const char zero[1 << 16];
    for (size_t off = 0; off < n; off += sizeof zero) {
        size_t len = n - off < sizeof zero ? n - off : sizeof zero;
        if (memcmp(base + off, zero, len) && pwrite(fd, base + off, len, (off_t)off) != (ssize_t)len) return -1;
    }
    return 0;
}

/* Under alloc_lock: puts allocation r on a new file in ipc_dir. */
static CUresult ipc_share(struct alloc_rec *r)
{
    static unsigned long long n;
    unsigned rnd = 0;
    if (getrandom(&rnd, sizeof rnd, 0) != sizeof rnd) rnd = (unsigned)getpid();
    fg_init();
    mkdir(fg_cfg.ipc_dir, 01777);
    fg_collect_unheld(fg_cfg.ipc_dir);
    char path[sizeof fg_cfg.ipc_dir + 32];
    snprintf(r->ipc_name, sizeof r->ipc_name, "%08x.%x.%llx", rnd, (unsigned)getpid(), ++n);
    snprintf(path, sizeof path, "%s/%s", fg_cfg.ipc_dir, r->ipc_name);
    int fd = open(path, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) return CUDA_ERROR_OUT_OF_MEMORY;
    flock(fd, LOCK_SH);
    if (ftruncate(fd, (off_t)r->size) || fill_file(fd, (const char *)r->base, r->size) ||
        mmap((void *)r->base, r->size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0) == MAP_FAILED) {
        unlink(path);
        close(fd);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    r->ipc_fd = fd;
    return CUDA_SUCCESS;
}

API CUresult cuIpcGetMemHandle(CUipcMemHandle *out, CUdeviceptr p)
{
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    pthread_mutex_lock(&alloc_lock);
    struct alloc_rec *r = allocs;
    for (; r; r = r->next)
        if (p >= r->base && p < r->base + r->size) break;
    CUresult rc = !r || r->mapped != MAPPED_OWN ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
    if (rc == CUDA_SUCCESS && r->ipc_fd < 0) rc = ipc_share(r);
    struct stat st;
    if (rc == CUDA_SUCCESS && fstat(r->ipc_fd, &st)) rc = CUDA_ERROR_OUT_OF_MEMORY;
    if (rc == CUDA_SUCCESS) {
        struct ipc_handle h = {.size = r->size, .ino = (uint64_t)st.st_ino, .devno = (uint64_t)st.st_dev};
        memcpy(h.magic, IPC_MAGIC, sizeof h.magic);
        memcpy(h.name, r->ipc_name, sizeof h.name);
        memcpy(out, &h, sizeof h);
    }
    pthread_mutex_unlock(&alloc_lock);
    return rc;
}

API CUresult cuIpcOpenMemHandle_v2(CUdeviceptr *p, CUipcMemHandle handle, unsigned flags)
{
    (void)flags;
    struct ipc_handle h;
    memcpy(&h, &handle, sizeof h);
    if (!p || memcmp(h.magic, IPC_MAGIC, sizeof h.magic) || !memchr(h.name, 0, sizeof h.name) || strchr(h.name, '/'))
        return CUDA_ERROR_INVALID_VALUE;
    int own = 0;
    pthread_mutex_lock(&alloc_lock);
    for (struct alloc_rec *r = allocs; r && !own; r = r->next)
        own = r->ipc_fd >= 0 && !strcmp(r->ipc_name, h.name);
    pthread_mutex_unlock(&alloc_lock);
    if (own) return CUDA_ERROR_INVALID_CONTEXT; /* as the driver: not in its own process */
    char path[sizeof fg_cfg.ipc_dir + 32];
    struct stat st;
    fg_init();
    snprintf(path, sizeof path, "%s/%s", fg_cfg.ipc_dir, h.name);
    int fd = open(path, O_RDWR | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return CUDA_ERROR_INVALID_HANDLE; /* another node, or freed */
    void *m = MAP_FAILED;
    if (!fstat(fd, &st) && (uint64_t)st.st_ino == h.ino && (uint64_t)st.st_dev == h.devno)
        m = mmap(NULL, h.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (m == MAP_FAILED) return CUDA_ERROR_INVALID_HANDLE;
    struct alloc_rec *r = malloc(sizeof *r);
    if (!r) {
        munmap(m, h.size);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    pthread_mutex_lock(&alloc_lock);
    *r = (struct alloc_rec){.base = (uintptr_t)m, .size = h.size, .type = MEM_DEVICE, .dev = current_dev(), .phys = -1,
                            .mapped = MAPPED_IPC, .id = ++alloc_seq, .next = allocs, .ipc_fd = -1};
    allocs = r;
    pthread_mutex_unlock(&alloc_lock);
    *p = (CUdeviceptr)(uintptr_t)m;
    return CUDA_SUCCESS;
}

API CUresult cuIpcOpenMemHandle(CUdeviceptr *p, CUipcMemHandle handle, unsigned flags)
{
    return cuIpcOpenMemHandle_v2(p, handle, flags);
}

API CUresult cuIpcCloseMemHandle(CUdeviceptr p)
{
    struct alloc_rec r;
    if (!lookup((uintptr_t)p, &r) || r.base != p || r.mapped != MAPPED_IPC) return CUDA_ERROR_INVALID_VALUE;
    untrack((void *)(uintptr_t)p);
    return CUDA_SUCCESS;
}

/* Events carry no data here: an opened event handle is a new event. */
typedef struct { char reserved[64]; } CUipcEventHandle;
CUresult cuEventCreate(void **out, unsigned flags); /* timing.c */

API CUresult cuIpcGetEventHandle(CUipcEventHandle *out, void *event)
{
    if (!out || !event) return CUDA_ERROR_INVALID_VALUE;
    memset(out, 0, sizeof *out);
    memcpy(out->reserved, "FGIPCEVT", 8);
    return CUDA_SUCCESS;
}

API CUresult cuIpcOpenEventHandle(void **event, CUipcEventHandle handle)
{
    if (!event || memcmp(handle.reserved, "FGIPCEVT", 8)) return CUDA_ERROR_INVALID_VALUE;
    return cuEventCreate(event, 0x2 /* CU_EVENT_DISABLE_TIMING, as IPC events are */);
}

static CUresult alloc(CUdeviceptr *p, size_t n)
{
    if (!p) return CUDA_ERROR_INVALID_VALUE;
    void *m = track(n, MEM_DEVICE);
    if (!m) return CUDA_ERROR_OUT_OF_MEMORY;
    *p = (CUdeviceptr)(uintptr_t)m;
    return CUDA_SUCCESS;
}

API CUresult cuMemAlloc_v2(CUdeviceptr *p, size_t n) { return alloc(p, n); }
API CUresult cuMemAllocAsync(CUdeviceptr *p, size_t n, CUstream s) { (void)s; return alloc(p, n); }
API CUresult cuMemAllocManaged(CUdeviceptr *p, size_t n, unsigned f) { (void)f; return alloc(p, n); }
API CUresult cuMemAllocFromPoolAsync(CUdeviceptr *p, size_t n, void *pool, CUstream s) { (void)pool; (void)s; return alloc(p, n); }

API CUresult cuMemAllocPitch_v2(CUdeviceptr *p, size_t *pitch, size_t w, size_t h, unsigned elem)
{
    (void)elem;
    if (pitch) *pitch = w;
    return alloc(p, w * h);
}

API CUresult cuMemFree_v2(CUdeviceptr p) { untrack((void *)(uintptr_t)p); return CUDA_SUCCESS; }
API CUresult cuMemFreeAsync(CUdeviceptr p, CUstream s) { (void)s; return cuMemFree_v2(p); }

API CUresult cuMemAllocHost_v2(void **p, size_t n)
{
    if (!p) return CUDA_ERROR_INVALID_VALUE;
    *p = track(n, MEM_HOST);
    return *p ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}

API CUresult cuMemHostAlloc(void **p, size_t n, unsigned f) { (void)f; return cuMemAllocHost_v2(p, n); }
API CUresult cuMemFreeHost(void *p) { untrack(p); return CUDA_SUCCESS; }

API CUresult cuMemHostGetFlags(unsigned *flags, void *p)
{
    (void)p;
    if (flags) *flags = 0;
    return CUDA_SUCCESS;
}

API CUresult cuMemGetAddressRange_v2(CUdeviceptr *base, size_t *size, CUdeviceptr p)
{
    struct alloc_rec r;
    if (!lookup((uintptr_t)p, &r)) return CUDA_ERROR_NOT_FOUND;
    if (base) *base = r.base;
    if (size) *size = r.size;
    return CUDA_SUCCESS;
}

API CUresult cuMemGetAllocationGranularity(size_t *g, const void *prop, int option)
{
    (void)prop;
    if (!g) return CUDA_ERROR_INVALID_VALUE;
    *g = option ? (2ULL << 20) : (2ULL << 20); /* minimum and recommended: 2 MiB */
    return CUDA_SUCCESS;
}

API CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *d, void *h, unsigned f)
{
    (void)f;
    if (d) *d = (CUdeviceptr)(uintptr_t)h;
    return CUDA_SUCCESS;
}

#define DPTR(x) ((void *)(uintptr_t)(x))

uint64_t fakegpu_enqueue(CUstream h, uint64_t ns);
void fakegpu_run(CUstream h, uint64_t ns);
void fakegpu_nvlink_traffic(CUstream h, uint64_t tx, uint64_t rx);
uint64_t fg_copy_ns(size_t bytes, int kind);
enum { LINK_HOST = 0, LINK_HBM = 1, LINK_PEER = 2 };

/* Data is copied for real up to copy_max_mb; larger copies only cost time. */
static void copy(void *d, const void *s, size_t n)
{
    if (n <= fg_copy_max()) memmove(d, s, n);
}

static CUresult copy_async(void *d, const void *s, size_t n, int link, CUstream st)
{
    copy(d, s, n);
    if (link == LINK_PEER) fakegpu_nvlink_traffic(st, n, 0);
    fakegpu_enqueue(st, fg_copy_ns(n, link));
    return CUDA_SUCCESS;
}

static CUresult copy_sync(void *d, const void *s, size_t n, int link)
{
    copy(d, s, n);
    if (link == LINK_PEER) fakegpu_nvlink_traffic(NULL, n, 0);
    fakegpu_run(NULL, fg_copy_ns(n, link));
    return CUDA_SUCCESS;
}

/* Unified addressing: tell device and host pointers apart by our records. */
static int link_between(const void *d, const void *s)
{
    struct alloc_rec rd, rs;
    int dd = lookup((uintptr_t)d, &rd) && rd.type == MEM_DEVICE;
    int sd = lookup((uintptr_t)s, &rs) && rs.type == MEM_DEVICE;
    if (dd && sd) return rd.phys == rs.phys ? LINK_HBM : LINK_PEER;
    return LINK_HOST;
}

API CUresult cuMemcpy(CUdeviceptr d, CUdeviceptr s, size_t n) { return copy_sync(DPTR(d), DPTR(s), n, link_between(DPTR(d), DPTR(s))); }
API CUresult cuMemcpyAsync(CUdeviceptr d, CUdeviceptr s, size_t n, CUstream st) { return copy_async(DPTR(d), DPTR(s), n, link_between(DPTR(d), DPTR(s)), st); }
API CUresult cuMemcpyHtoD_v2(CUdeviceptr d, const void *s, size_t n) { return copy_sync(DPTR(d), s, n, LINK_HOST); }
API CUresult cuMemcpyDtoH_v2(void *d, CUdeviceptr s, size_t n) { return copy_sync(d, DPTR(s), n, LINK_HOST); }
API CUresult cuMemcpyDtoD_v2(CUdeviceptr d, CUdeviceptr s, size_t n) { return copy_sync(DPTR(d), DPTR(s), n, link_between(DPTR(d), DPTR(s))); }
API CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr d, const void *s, size_t n, CUstream st) { return copy_async(DPTR(d), s, n, LINK_HOST, st); }
API CUresult cuMemcpyDtoHAsync_v2(void *d, CUdeviceptr s, size_t n, CUstream st) { return copy_async(d, DPTR(s), n, LINK_HOST, st); }
API CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr d, CUdeviceptr s, size_t n, CUstream st) { return copy_async(DPTR(d), DPTR(s), n, link_between(DPTR(d), DPTR(s)), st); }

API CUresult cuMemcpyPeer(CUdeviceptr d, CUcontext dc, CUdeviceptr s, CUcontext sc, size_t n)
{
    (void)dc; (void)sc;
    return copy_sync(DPTR(d), DPTR(s), n, LINK_PEER);
}

API CUresult cuMemcpyPeerAsync(CUdeviceptr d, CUcontext dc, CUdeviceptr s, CUcontext sc, size_t n, CUstream st)
{
    (void)dc; (void)sc;
    return copy_async(DPTR(d), DPTR(s), n, LINK_PEER, st);
}

static CUresult fill(CUdeviceptr d, unsigned v, size_t n, size_t width, CUstream st, int sync)
{
    if (n * width <= fg_copy_max()) {
        if (width == 1) {
            memset(DPTR(d), (int)v, n);
        } else if (width == 2) {
            unsigned short *p = DPTR(d);
            for (size_t i = 0; i < n; i++) p[i] = (unsigned short)v;
        } else {
            unsigned *p = DPTR(d);
            for (size_t i = 0; i < n; i++) p[i] = v;
        }
    }
    uint64_t ns = fg_copy_ns(n * width, LINK_HBM);
    if (sync) fakegpu_run(NULL, ns); else fakegpu_enqueue(st, ns);
    return CUDA_SUCCESS;
}

API CUresult cuMemsetD8_v2(CUdeviceptr d, unsigned char v, size_t n) { return fill(d, v, n, 1, NULL, 1); }
API CUresult cuMemsetD16_v2(CUdeviceptr d, unsigned short v, size_t n) { return fill(d, v, n, 2, NULL, 1); }
API CUresult cuMemsetD32_v2(CUdeviceptr d, unsigned v, size_t n) { return fill(d, v, n, 4, NULL, 1); }
API CUresult cuMemsetD8Async(CUdeviceptr d, unsigned char v, size_t n, CUstream s) { return fill(d, v, n, 1, s, 0); }
API CUresult cuMemsetD16Async(CUdeviceptr d, unsigned short v, size_t n, CUstream s) { return fill(d, v, n, 2, s, 0); }
API CUresult cuMemsetD32Async(CUdeviceptr d, unsigned v, size_t n, CUstream s) { return fill(d, v, n, 4, s, 0); }

/* ---- handles: streams, events, modules, functions ---------------------- */


API CUresult cuModuleLoad(void **m, const char *f) { (void)f; if (m) *m = fake_handle(); return CUDA_SUCCESS; }
API CUresult cuModuleLoadData(void **m, const void *img) { (void)img; if (m) *m = fake_handle(); return CUDA_SUCCESS; }

API CUresult cuModuleLoadDataEx(void **m, const void *img, unsigned n, void *o, void **ov)
{
    (void)n; (void)o; (void)ov;
    return cuModuleLoadData(m, img);
}

API CUresult cuModuleLoadFatBinary(void **m, const void *fb) { return cuModuleLoadData(m, fb); }

API CUresult cuModuleGetFunction(void **f, void *m, const char *name)
{
    (void)m; (void)name;
    if (f) *f = fake_handle();
    return CUDA_SUCCESS;
}

API CUresult cuLibraryLoadData(void **lib, const void *code, void *jo, void **jv, unsigned nj,
                               void *lo, void **lv, unsigned nl)
{
    (void)code; (void)jo; (void)jv; (void)nj; (void)lo; (void)lv; (void)nl;
    if (lib) *lib = fake_handle();
    return CUDA_SUCCESS;
}

API CUresult cuLibraryGetKernel(void **k, void *lib, const char *name)
{
    (void)lib; (void)name;
    if (k) *k = fake_handle();
    return CUDA_SUCCESS;
}

API CUresult cuLibraryGetModule(void **m, void *lib) { (void)lib; if (m) *m = fake_handle(); return CUDA_SUCCESS; }

API CUresult cuKernelGetFunction(void **f, void *k) { (void)k; if (f) *f = fake_handle(); return CUDA_SUCCESS; }

/* CUfunction_attribute values for a kernel compiled for this GPU. CUB (scans,
 * sorts: torch.cumsum, top-k, sampling) picks its tuning policy from the PTX
 * version and fails with "invalid device function" when it is 0. */
API CUresult cuFuncGetAttribute(int *v, int attr, void *f)
{
    (void)f;
    if (!v) return CUDA_ERROR_INVALID_VALUE;
    switch (attr) {
    case 0: *v = 1024; break;                          /* MAX_THREADS_PER_BLOCK */
    case 4: *v = 32; break;                            /* NUM_REGS */
    case 5: case 6: *v = FG_CC_MAJOR * 10 + FG_CC_MINOR; break; /* PTX_VERSION, BINARY_VERSION */
    case 8: *v = 49152; break;                         /* MAX_DYNAMIC_SHARED_SIZE_BYTES */
    case 9: *v = -1; break;                            /* PREFERRED_SHARED_MEMORY_CARVEOUT: default */
    default: *v = 0;
    }
    return CUDA_SUCCESS;
}

API CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *n, void *f, int bs, size_t sm)
{
    (void)f; (void)bs; (void)sm;
    if (n) *n = 1;
    return CUDA_SUCCESS;
}

/* CUpointer_attribute values and their result types. Pointers we did not
 * allocate are reported as unregistered host memory, like the real driver. */
static CUresult pointer_attribute(void *data, int attr, CUdeviceptr p)
{
    struct alloc_rec r;
    int known = lookup((uintptr_t)p, &r);
    switch (attr) {
    case 1: *(void **)data = known ? primary_ctx(r.dev) : NULL; break;            /* CONTEXT */
    case 2: if (!known) return CUDA_ERROR_INVALID_VALUE;                          /* MEMORY_TYPE */
            *(unsigned *)data = (unsigned)r.type; break;
    case 3: *(CUdeviceptr *)data = p; break;                                      /* DEVICE_POINTER */
    case 4: *(void **)data = DPTR(p); break;                                      /* HOST_POINTER */
    case 6: *(int *)data = 1; break;                                              /* SYNC_MEMOPS */
    case 7: *(unsigned long long *)data = known ? r.id : 0; break;                /* BUFFER_ID */
    case 8: *(int *)data = 0; break;                                              /* IS_MANAGED */
    case 9: if (!known) return CUDA_ERROR_INVALID_VALUE;                          /* DEVICE_ORDINAL */
            *(int *)data = r.dev; break;
    case 10: *(int *)data = known; break;                                         /* IS_LEGACY_CUDA_IPC_CAPABLE */
    case 11: case 19: *(CUdeviceptr *)data = known ? r.base : p; break;           /* RANGE_START / MAPPING_BASE */
    case 12: case 18: *(size_t *)data = known ? r.size : 0; break;                /* RANGE_SIZE / MAPPING_SIZE */
    case 13: *(int *)data = known; break;                                         /* MAPPED */
    case 15: *(int *)data = 0; break;                                             /* IS_GPU_DIRECT_RDMA_CAPABLE */
    case 16: *(unsigned *)data = 3; break;                                        /* ACCESS_FLAGS = RW */
    case 17: *(void **)data = NULL; break;                                        /* MEMPOOL_HANDLE */
    case 20: *(unsigned long long *)data = known ? r.id : 0; break;               /* MEMORY_BLOCK_ID */
    default: *(unsigned long long *)data = 0;
    }
    return CUDA_SUCCESS;
}

API CUresult cuPointerGetAttribute(void *data, int attr, CUdeviceptr p)
{
    if (!data) return CUDA_ERROR_INVALID_VALUE;
    return pointer_attribute(data, attr, p);
}

/* The plural form never fails; unknown pointers read as zeros. */
API CUresult cuPointerGetAttributes(unsigned n, int *attrs, void **data, CUdeviceptr p)
{
    if (!attrs || !data) return CUDA_ERROR_INVALID_VALUE;
    for (unsigned i = 0; i < n; i++) {
        if (!data[i]) continue;
        if (pointer_attribute(data[i], attrs[i], p) != CUDA_SUCCESS) *(unsigned *)data[i] = 0;
    }
    return CUDA_SUCCESS;
}

/* ---- streams: capture queries (nothing is ever captured) ---------------- */

#define CAPTURE_STATUS_NONE 0

API CUresult cuStreamIsCapturing(CUstream s, int *status)
{
    (void)s;
    if (!status) return CUDA_ERROR_INVALID_VALUE;
    *status = CAPTURE_STATUS_NONE;
    return CUDA_SUCCESS;
}

API CUresult cuStreamGetCaptureInfo_v2(CUstream s, int *status, unsigned long long *id, void **graph,
                                       const void **deps, size_t *ndeps)
{
    (void)s;
    if (status) *status = CAPTURE_STATUS_NONE;
    if (id) *id = 0;
    if (graph) *graph = NULL;
    if (deps) *deps = NULL;
    if (ndeps) *ndeps = 0;
    return CUDA_SUCCESS;
}

API CUresult cuStreamGetCaptureInfo_v3(CUstream s, int *status, unsigned long long *id, void **graph,
                                       const void **deps, const void **edges, size_t *ndeps)
{
    if (edges) *edges = NULL;
    return cuStreamGetCaptureInfo_v2(s, status, id, graph, deps, ndeps);
}

API CUresult cuThreadExchangeStreamCaptureMode(int *mode)
{
    static __thread int thread_mode;
    if (!mode) return CUDA_ERROR_INVALID_VALUE;
    int old = thread_mode;
    thread_mode = *mode;
    *mode = old;
    return CUDA_SUCCESS;
}

API CUresult cuStreamGetPriority(CUstream s, int *prio) { (void)s; if (prio) *prio = 0; return CUDA_SUCCESS; }
API CUresult cuStreamGetFlags(CUstream s, unsigned *flags) { (void)s; if (flags) *flags = 0; return CUDA_SUCCESS; }
API CUresult cuStreamGetDevice(CUstream s, CUdevice *dev) { (void)s; if (dev) *dev = current_dev(); return CUDA_SUCCESS; }
API CUresult cuStreamGetCtx(CUstream s, CUcontext *ctx) { (void)s; if (ctx) *ctx = cur_ctx; return CUDA_SUCCESS; }

API CUresult cuStreamGetId(CUstream s, unsigned long long *id)
{
    if (id) *id = (unsigned long long)(uintptr_t)s;
    return CUDA_SUCCESS;
}

/* Capture never starts, so there is never a graph to hand back. */
API CUresult cuStreamEndCapture(CUstream s, void **graph)
{
    (void)s;
    if (graph) *graph = fake_handle();
    return CUDA_SUCCESS;
}

API CUresult cuGraphCreate(void **graph, unsigned flags) { (void)flags; if (graph) *graph = fake_handle(); return CUDA_SUCCESS; }

API CUresult cuGraphInstantiateWithFlags(void **exec, void *graph, unsigned long long flags)
{
    (void)graph; (void)flags;
    if (exec) *exec = fake_handle();
    return CUDA_SUCCESS;
}

API CUresult cuGraphGetNodes(void *graph, void **nodes, size_t *n)
{
    (void)graph; (void)nodes;
    if (n) *n = 0;
    return CUDA_SUCCESS;
}

/* ---- modules / kernels / occupancy ------------------------------------- */

API CUresult cuModuleGetLoadingMode(int *mode)
{
    if (mode) *mode = 2; /* CU_MODULE_LAZY_LOADING */
    return CUDA_SUCCESS;
}

/* Globals get a zeroed host buffer so reads and writes stay valid. */
API CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, void *mod, const char *name)
{
    (void)mod; (void)name;
    const size_t size = 4096;
    if (dptr) alloc(dptr, size);
    if (bytes) *bytes = size;
    return CUDA_SUCCESS;
}

API CUresult cuLibraryGetGlobal(CUdeviceptr *dptr, size_t *bytes, void *lib, const char *name)
{
    return cuModuleGetGlobal_v2(dptr, bytes, lib, name);
}

API CUresult cuLibraryGetManaged(CUdeviceptr *dptr, size_t *bytes, void *lib, const char *name)
{
    return cuModuleGetGlobal_v2(dptr, bytes, lib, name);
}

API CUresult cuKernelGetAttribute(int *v, int attr, void *k, CUdevice dev)
{
    (void)dev;
    return cuFuncGetAttribute(v, attr, k);
}

API CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *n, void *f, int bs, size_t sm, unsigned flags)
{
    (void)flags;
    return cuOccupancyMaxActiveBlocksPerMultiprocessor(n, f, bs, sm);
}

API CUresult cuOccupancyMaxPotentialBlockSize(int *min_grid, int *block, void *f, void *cb, size_t sm, int limit)
{
    (void)f; (void)cb; (void)sm;
    if (min_grid) *min_grid = FG_SM_COUNT;
    if (block) *block = limit > 0 && limit < 1024 ? limit : 1024;
    return CUDA_SUCCESS;
}

API CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int *min_grid, int *block, void *f, void *cb, size_t sm,
                                                       int limit, unsigned flags)
{
    (void)flags;
    return cuOccupancyMaxPotentialBlockSize(min_grid, block, f, cb, sm, limit);
}

API CUresult cuFuncGetName(const char **name, void *f) { (void)f; if (name) *name = "fakegpu_kernel"; return CUDA_SUCCESS; }
API CUresult cuKernelGetName(const char **name, void *k) { (void)k; if (name) *name = "fakegpu_kernel"; return CUDA_SUCCESS; }

/* ---- contexts (legacy names / getters) --------------------------------- */

API CUresult cuCtxPushCurrent(CUcontext ctx) { return cuCtxPushCurrent_v2(ctx); }
API CUresult cuCtxPopCurrent(CUcontext *ctx) { return cuCtxPopCurrent_v2(ctx); }

API CUresult cuCtxGetCacheConfig(int *cfg) { if (cfg) *cfg = 0; return CUDA_SUCCESS; }
API CUresult cuCtxGetSharedMemConfig(int *cfg) { if (cfg) *cfg = 0; return CUDA_SUCCESS; }

API CUresult cuDeviceGetDefaultMemPool(void **pool, CUdevice dev)
{
    static char pools[FG_MAX_GPUS];
    int p = phys(dev);
    if (!pool) return CUDA_ERROR_INVALID_VALUE;
    if (p < 0) return CUDA_ERROR_INVALID_DEVICE;
    *pool = &pools[p];
    return CUDA_SUCCESS;
}

API CUresult cuDeviceGetMemPool(void **pool, CUdevice dev) { return cuDeviceGetDefaultMemPool(pool, dev); }

API CUresult cuMemPoolGetAttribute(void *pool, int attr, void *value)
{
    (void)pool; (void)attr;
    if (!value) return CUDA_ERROR_INVALID_VALUE;
    *(unsigned long long *)value = 0; /* thresholds, usage counters, reuse flags */
    return CUDA_SUCCESS;
}

/* ---- entry points ------------------------------------------------------ */

static CUresult generic_stub(void) { return CUDA_SUCCESS; }

/* Generated weak no-ops live in their own section (see Makefile), so a
 * symbol can be told apart from a hand-written implementation. */
extern const char __start_fg_weak[], __stop_fg_weak[];

static int is_generated(void *fn)
{
    return (const char *)fn >= __start_fg_weak && (const char *)fn < __stop_fg_weak;
}

/* libcudart resolves every driver function through here. Map the base name
 * to a versioned symbol, preferring hand-written implementations over the
 * generated no-ops; anything unknown gets a success-returning no-op. */
API CUresult cuGetProcAddress_v2(const char *sym, void **pfn, int ver, unsigned long long flags, int *status)
{
    (void)flags;
    if (!sym || !pfn) return CUDA_ERROR_INVALID_VALUE;
    static void *self;
    if (!self) {
        Dl_info info;
        dladdr((void *)cuGetProcAddress_v2, &info);
        self = dlopen(info.dli_fname, RTLD_LAZY | RTLD_NOLOAD);
    }
    static const char *suffixes[] = {"_v4", "_v3", "_v2", ""};
    char name[256];
    void *fn = NULL, *fallback = NULL;
    for (size_t i = 0; i < sizeof suffixes / sizeof *suffixes && !fn; i++) {
        snprintf(name, sizeof name, "%s%s", sym, suffixes[i]);
        void *f = dlsym(self, name);
        if (!f) continue;
        if (!is_generated(f)) fn = f;
        else if (!fallback) fallback = f;
    }
    if (!fn && fallback && fg_debug()) fprintf(stderr, "[fakegpu] cuGetProcAddress(%s, %d): generated no-op\n", sym, ver);
    if (!fn) fn = fallback;
    if (!fn && fg_debug()) fprintf(stderr, "[fakegpu] cuGetProcAddress(%s, %d): no-op stub\n", sym, ver);
    *pfn = fn ? fn : (void *)generic_stub;
    if (status) *status = 0; /* CU_GET_PROC_ADDRESS_SUCCESS */
    return CUDA_SUCCESS;
}

API CUresult cuGetProcAddress(const char *sym, void **pfn, int ver, unsigned long long flags)
{
    return cuGetProcAddress_v2(sym, pfn, ver, flags, NULL);
}
