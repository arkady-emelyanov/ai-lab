/*
 * CUDA virtual memory management, shareable handles and multicast objects
 * (cuMemCreate, cuMemMap, cuMemExportToShareableHandle, cuMulticastCreate,
 * ...): what NCCL's NVLS, PyTorch's symmetric memory and FlashInfer's
 * MNNVL all-reduce build their buffers from on GB200.
 *
 * "GPU memory" is host memory, so an allocation is a file and a mapping is
 * an mmap of it:
 *   - an allocation that may only be shared as a POSIX file descriptor is a
 *     memfd; exporting it hands out a dup of that fd, which the receiving
 *     process maps, so both see the same bytes (as with the real driver,
 *     the fd travels over a Unix socket);
 *   - an allocation that may be shared as a fabric handle (MNNVL, across
 *     trays) is a file in fabric_dir, which every tray of the NVLink domain
 *     reaches (/shared in the lab). The 64-byte fabric handle names it.
 *     Its creator and importers hold a shared flock on it, which the kernel
 *     drops when they exit or die; files no process holds any more are
 *     deleted the next time fabric memory is created, as the real driver
 *     frees a dead process' memory.
 *
 * Fabric handles follow the NVLink partitions, as IMEX does on GB200: one
 * can only be exported from, and imported into, a GPU whose partition
 * (cluster UUID and clique, from NVML like NCCL reads them) is the one the
 * memory was created in. A GPU in no partition has no fabric memory.
 *
 * A multicast object has backing of its own, so its address works as soon
 * as it is mapped; binding memory to it is recorded and checked. Writes
 * through it reach only that backing, not every bound GPU: no GPU kernel
 * runs here, and only kernels would use it.
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include "occupancy.h"

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef unsigned long long CUmemGenericAllocationHandle;

#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_VALUE 1
#define CUDA_ERROR_OUT_OF_MEMORY 2
#define CUDA_ERROR_INVALID_DEVICE 101
#define CUDA_ERROR_ALREADY_MAPPED 208
#define CUDA_ERROR_NOT_MAPPED 211
#define CUDA_ERROR_INVALID_HANDLE 400
#define CUDA_ERROR_NOT_PERMITTED 800
#define CUDA_ERROR_NOT_SUPPORTED 801

#define API __attribute__((visibility("default")))

enum { HANDLE_POSIX_FD = 0x1, HANDLE_FABRIC = 0x8 };
enum { LOCATION_DEVICE = 1 };
#define GRANULARITY (2ULL << 20)

/* CUmemAllocationProp, CUmemAccessDesc, CUmulticastObjectProp (cuda.h). */
typedef struct {
    int type;
    int requestedHandleTypes;
    struct { int type, id; } location;
    void *win32HandleMetaData;
    unsigned char allocFlags[8];
} mem_prop;
typedef struct { struct { int type, id; } location; int flags; } access_desc;
typedef struct {
    unsigned numDevices;
    size_t size;
    unsigned long long handleTypes, flags;
} mc_prop;

/* Provided by cuda_stub.c */
void fg_vmm_track(uintptr_t base, size_t n, int dev, int phys);
void fg_vmm_untrack(uintptr_t base);
void fg_vmm_account(int phys, int64_t delta);
int fg_vmm_phys(CUdevice dev);
int fg_vmm_current_dev(void);

/* ---- NVLink partition of a device ---------------------------------------- */

struct partition { unsigned char cluster[16]; unsigned clique; };
typedef struct { unsigned char clusterUuid[16]; int status; unsigned cliqueId; unsigned char state; } fabric_info_t;
API CUresult cuDeviceGetPCIBusId(char *buf, int len, CUdevice dev);

/* As NCCL finds it: the device's PCI bus id, then NVML's fabric info (the
 * clique the GPU took at its last reset). Clique 0: in no partition. */
static int partition_of(CUdevice dev, struct partition *out)
{
    memset(out, 0, sizeof *out);
    static void *nvml;
    if (!nvml) nvml = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
    if (!nvml) return -1;
    int (*init)(void) = (int (*)(void))dlsym(nvml, "nvmlInit_v2");
    int (*by_bus)(const char *, void **) = (int (*)(const char *, void **))dlsym(nvml, "nvmlDeviceGetHandleByPciBusId_v2");
    int (*fabric)(void *, fabric_info_t *) = (int (*)(void *, fabric_info_t *))dlsym(nvml, "nvmlDeviceGetGpuFabricInfo");
    char bus[32];
    void *h;
    fabric_info_t f = {0};
    if (!init || !by_bus || !fabric || cuDeviceGetPCIBusId(bus, sizeof bus, dev) || init() || by_bus(bus, &h) || fabric(h, &f))
        return -1;
    memcpy(out->cluster, f.clusterUuid, 16);
    out->clique = f.cliqueId;
    return 0;
}


/* ---- objects: allocations and multicast objects --------------------------- */

#define OBJ_MAGIC 0xFA6E3D11u
enum { OBJ_MEM = 1, OBJ_MC = 2 };

/* CUmemFabricHandle: 64 opaque bytes. Ours name the file in fabric_dir. */
struct fabric_handle {
    char magic[8];
    unsigned char cluster[16];
    uint32_t clique, kind;
    uint64_t size;
    char name[24];
};
_Static_assert(sizeof(struct fabric_handle) == 64, "CUmemFabricHandle is 64 bytes");
static const char FABRIC_MAGIC[8] = "FGFABRIC";

/* Whether GPU dev is, now, in the NVLink partition fabric memory f belongs to. */
static int in_partition(CUdevice dev, const struct fabric_handle *f)
{
    struct partition p;
    return partition_of(dev, &p) == 0 && p.clique && p.clique == f->clique && !memcmp(p.cluster, f->cluster, 16);
}

struct binding { size_t mc_offset, size; struct obj *mem; struct binding *next; };

struct obj {
    unsigned magic;
    int kind, fd;
    size_t size;
    unsigned long long types;  /* shareable handle types it was created with */
    int dev, phys;             /* the GPU it lives on; phys -1: another process' */
    int refs;                  /* the handle plus each mapping */
    int accounted;             /* counted in this process' GPU memory */
    struct fabric_handle fabric;  /* fabric objects: their handle */
    char path[sizeof fg_cfg.fabric_dir + 32];  /* created here: unlink when gone */
    unsigned mc_devices, mc_added;
    struct binding *bindings;
};

struct mapping { uintptr_t base; size_t size; struct obj *obj; struct mapping *next; };
struct reservation { uintptr_t base; size_t size; struct reservation *next; };

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static struct mapping *mappings;
static struct reservation *reservations;
static unsigned long long seq;

static struct obj *obj_of(CUmemGenericAllocationHandle h)
{
    struct obj *o = (struct obj *)(uintptr_t)h;
    return o && o->magic == OBJ_MAGIC ? o : NULL;
}

/* Under lock. */
static void obj_put(struct obj *o)
{
    if (--o->refs > 0) return;
    if (o->accounted) fg_vmm_account(o->phys, -(int64_t)o->size);
    if (o->path[0]) unlink(o->path);
    close(o->fd);
    for (struct binding *b = o->bindings, *next; b; b = next) {
        next = b->next;
        obj_put(b->mem);
        free(b);
    }
    o->magic = 0;
    free(o);
}

static int round_ok(size_t n) { return n && n % GRANULARITY == 0; }

/* Deletes the files in dir no live process holds a lock on (their users
 * exited without releasing them): fabric memory here, CUDA IPC memory in
 * cuda_stub.c. Young files are left alone: their creator may not hold its
 * lock yet. */
void fg_collect_unheld(const char *dir)
{
    DIR *d = opendir(dir);
    if (!d) return;
    time_t now = time(NULL);
    for (struct dirent *e; (e = readdir(d));) {
        if (e->d_name[0] == '.') continue;
        char path[512];
        struct stat st;
        snprintf(path, sizeof path, "%s/%s", dir, e->d_name);
        if (lstat(path, &st) || !S_ISREG(st.st_mode) || now - st.st_mtime < 10) continue;
        int fd = open(path, O_RDWR | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0) continue;
        if (flock(fd, LOCK_EX | LOCK_NB) == 0) unlink(path);
        close(fd);
    }
    closedir(d);
}

/* Backing for a new object: a memfd, or a file in fabric_dir. */
static int backing(struct obj *o, int kind, size_t size, const struct partition *part)
{
    if (!(o->types & HANDLE_FABRIC)) {
        o->fd = memfd_create(kind == OBJ_MC ? "fakegpu-mc" : "fakegpu-mem", MFD_CLOEXEC);
        return o->fd >= 0 && ftruncate(o->fd, (off_t)size) == 0 ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
    }
    fg_init();
    if (!fg_cfg.fabric_dir[0]) return CUDA_ERROR_NOT_SUPPORTED;
    if (!part->clique) return CUDA_ERROR_NOT_PERMITTED; /* no partition, no fabric memory */
    unsigned r = 0;
    if (getrandom(&r, sizeof r, 0) != sizeof r) r = (unsigned)getpid();
    struct fabric_handle *f = &o->fabric;
    memcpy(f->magic, FABRIC_MAGIC, sizeof f->magic);
    memcpy(f->cluster, part->cluster, 16);
    f->clique = part->clique;
    f->kind = (uint32_t)kind;
    f->size = size;
    snprintf(f->name, sizeof f->name, "%08x.%x.%llx", r, (unsigned)getpid(), ++seq);
    mkdir(fg_cfg.fabric_dir, 01777);
    fg_collect_unheld(fg_cfg.fabric_dir);
    snprintf(o->path, sizeof o->path, "%s/%s", fg_cfg.fabric_dir, f->name);
    o->fd = open(o->path, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (o->fd < 0) {
        o->path[0] = 0;
        return CUDA_ERROR_NOT_PERMITTED;
    }
    flock(o->fd, LOCK_SH);
    if (ftruncate(o->fd, (off_t)size)) return CUDA_ERROR_OUT_OF_MEMORY;
    return CUDA_SUCCESS;
}

static CUresult create(CUmemGenericAllocationHandle *h, int kind, size_t size, unsigned long long types, CUdevice dev,
                       struct obj **out)
{
    struct partition part = {0};
    if (types & HANDLE_FABRIC) partition_of(dev, &part);
    struct obj *o = calloc(1, sizeof *o);
    if (!o) return CUDA_ERROR_OUT_OF_MEMORY;
    *o = (struct obj){.magic = OBJ_MAGIC, .kind = kind, .fd = -1, .size = size, .types = types,
                      .dev = dev, .phys = fg_vmm_phys(dev), .refs = 1};
    CUresult rc = backing(o, kind, size, &part);
    if (rc != CUDA_SUCCESS) {
        if (o->fd >= 0) close(o->fd);
        if (o->path[0]) unlink(o->path);
        free(o);
        return rc;
    }
    *h = (CUmemGenericAllocationHandle)(uintptr_t)o;
    if (out) *out = o;
    return CUDA_SUCCESS;
}

/* ---- allocations ------------------------------------------------------- */

API CUresult cuMemCreate(CUmemGenericAllocationHandle *h, size_t size, const mem_prop *prop, unsigned long long flags)
{
    (void)flags;
    if (!h || !prop || !round_ok(size) || prop->location.type != LOCATION_DEVICE) return CUDA_ERROR_INVALID_VALUE;
    int phys = fg_vmm_phys(prop->location.id);
    if (phys < 0) return CUDA_ERROR_INVALID_DEVICE;
    if (fg_gpu_mem_used(phys) + size > fg_mem_bytes()) return CUDA_ERROR_OUT_OF_MEMORY;
    struct obj *o;
    CUresult rc = create(h, OBJ_MEM, size, (unsigned)prop->requestedHandleTypes, prop->location.id, &o);
    if (rc != CUDA_SUCCESS) return rc;
    o->accounted = 1;
    fg_vmm_account(phys, (int64_t)size);
    return CUDA_SUCCESS;
}

API CUresult cuMemRelease(CUmemGenericAllocationHandle h)
{
    pthread_mutex_lock(&lock);
    struct obj *o = obj_of(h);
    if (o) obj_put(o);
    pthread_mutex_unlock(&lock);
    return o ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}

API CUresult cuMemGetAllocationPropertiesFromHandle(mem_prop *prop, CUmemGenericAllocationHandle h)
{
    struct obj *o = obj_of(h);
    if (!prop || !o || o->kind != OBJ_MEM) return !prop ? CUDA_ERROR_INVALID_VALUE : CUDA_ERROR_INVALID_HANDLE;
    memset(prop, 0, sizeof *prop);
    prop->type = 1; /* CU_MEM_ALLOCATION_TYPE_PINNED */
    prop->requestedHandleTypes = (int)o->types;
    prop->location.type = LOCATION_DEVICE;
    prop->location.id = o->dev < 0 ? fg_vmm_current_dev() : o->dev;
    return CUDA_SUCCESS;
}

/* ---- shareable handles ------------------------------------------------- */

API CUresult cuMemExportToShareableHandle(void *out, CUmemGenericAllocationHandle h, int type, unsigned long long flags)
{
    (void)flags;
    struct obj *o = obj_of(h);
    if (!out) return CUDA_ERROR_INVALID_VALUE;
    if (!o) return CUDA_ERROR_INVALID_HANDLE;
    if (!(o->types & (unsigned)type)) return CUDA_ERROR_INVALID_VALUE; /* not requested at creation */
    if (type == HANDLE_POSIX_FD) {
        int fd = dup(o->fd);
        if (fd < 0) return CUDA_ERROR_OUT_OF_MEMORY;
        *(int *)out = fd;
        return CUDA_SUCCESS;
    }
    if (type == HANDLE_FABRIC) {
        /* Exporting needs the GPU to still be in that partition. */
        if (o->dev >= 0 && !in_partition(o->dev, &o->fabric)) return CUDA_ERROR_NOT_PERMITTED;
        memcpy(out, &o->fabric, sizeof o->fabric);
        return CUDA_SUCCESS;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

API CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *h, void *os_handle, int type)
{
    if (!h) return CUDA_ERROR_INVALID_VALUE;
    int dev = fg_vmm_current_dev();
    struct obj *o = calloc(1, sizeof *o);
    if (!o) return CUDA_ERROR_OUT_OF_MEMORY;
    *o = (struct obj){.magic = OBJ_MAGIC, .kind = OBJ_MEM, .fd = -1, .types = (unsigned)type, .dev = -1, .phys = -1, .refs = 1};
    if (type == HANDLE_POSIX_FD) {
        struct stat st;
        char link[64], target[128] = "";
        int fd = (int)(intptr_t)os_handle;
        if (fd < 0 || fstat(fd, &st) || (o->fd = fcntl(fd, F_DUPFD_CLOEXEC, 0)) < 0) {
            free(o);
            return CUDA_ERROR_INVALID_VALUE;
        }
        o->size = (size_t)st.st_size;
        snprintf(link, sizeof link, "/proc/self/fd/%d", o->fd);
        if (readlink(link, target, sizeof target - 1) > 0 && strstr(target, "fakegpu-mc")) o->kind = OBJ_MC;
    } else if (type == HANDLE_FABRIC) {
        const struct fabric_handle *f = os_handle;
        if (!f || memcmp(f->magic, FABRIC_MAGIC, sizeof f->magic) || memchr(f->name, 0, sizeof f->name) == NULL) {
            free(o);
            return CUDA_ERROR_INVALID_VALUE;
        }
        /* IMEX: only GPUs of the partition the memory belongs to reach it. */
        if (!in_partition(dev, f)) {
            free(o);
            return CUDA_ERROR_NOT_PERMITTED;
        }
        char path[sizeof o->path];
        fg_init();
        snprintf(path, sizeof path, "%s/%s", fg_cfg.fabric_dir, f->name);
        if (!fg_cfg.fabric_dir[0] || (o->fd = open(path, O_RDWR | O_CLOEXEC)) < 0) {
            free(o);
            return CUDA_ERROR_INVALID_VALUE;
        }
        flock(o->fd, LOCK_SH);
        o->fabric = *f;
        o->kind = (int)f->kind;
        o->size = f->size;
    } else {
        free(o);
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    *h = (CUmemGenericAllocationHandle)(uintptr_t)o;
    return CUDA_SUCCESS;
}

/* ---- address space and mappings ------------------------------------------ */

API CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr addr,
                                 unsigned long long flags)
{
    (void)addr; (void)flags;
    if (!ptr || !round_ok(size)) return CUDA_ERROR_INVALID_VALUE;
    size_t align = alignment > GRANULARITY ? alignment : GRANULARITY;
    char *m = mmap(NULL, size + align, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (m == MAP_FAILED) return CUDA_ERROR_OUT_OF_MEMORY;
    uintptr_t base = ((uintptr_t)m + align - 1) & ~(uintptr_t)(align - 1);
    if (base > (uintptr_t)m) munmap(m, base - (uintptr_t)m);
    if ((uintptr_t)m + size + align > base + size) munmap((void *)(base + size), (uintptr_t)m + size + align - base - size);
    struct reservation *r = malloc(sizeof *r);
    if (!r) {
        munmap((void *)base, size);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    pthread_mutex_lock(&lock);
    *r = (struct reservation){base, size, reservations};
    reservations = r;
    pthread_mutex_unlock(&lock);
    *ptr = base;
    return CUDA_SUCCESS;
}

API CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size)
{
    pthread_mutex_lock(&lock);
    for (struct reservation **r = &reservations; *r; r = &(*r)->next) {
        if ((*r)->base == ptr && (*r)->size == size) {
            struct reservation *dead = *r;
            *r = dead->next;
            pthread_mutex_unlock(&lock);
            munmap((void *)ptr, size);
            free(dead);
            return CUDA_SUCCESS;
        }
    }
    pthread_mutex_unlock(&lock);
    return CUDA_ERROR_INVALID_VALUE;
}

/* Under lock. */
static int reserved(uintptr_t p, size_t n)
{
    for (struct reservation *r = reservations; r; r = r->next)
        if (p >= r->base && p + n <= r->base + r->size) return 1;
    return 0;
}

static struct mapping *mapping_at(uintptr_t p)
{
    for (struct mapping *m = mappings; m; m = m->next)
        if (p >= m->base && p < m->base + m->size) return m;
    return NULL;
}

API CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle h,
                      unsigned long long flags)
{
    (void)flags;
    pthread_mutex_lock(&lock);
    struct obj *o = obj_of(h);
    CUresult rc = CUDA_SUCCESS;
    if (!o) rc = CUDA_ERROR_INVALID_HANDLE;
    else if (!round_ok(size) || offset % GRANULARITY || offset + size > o->size || !reserved(ptr, size))
        rc = CUDA_ERROR_INVALID_VALUE;
    else
        for (uintptr_t p = ptr; p < ptr + size && rc == CUDA_SUCCESS; p += GRANULARITY)
            if (mapping_at(p)) rc = CUDA_ERROR_ALREADY_MAPPED;
    struct mapping *m = rc == CUDA_SUCCESS ? malloc(sizeof *m) : NULL;
    if (rc == CUDA_SUCCESS && (!m || mmap((void *)ptr, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, o->fd,
                                         (off_t)offset) == MAP_FAILED))
        rc = CUDA_ERROR_OUT_OF_MEMORY;
    if (rc != CUDA_SUCCESS) {
        free(m);
        pthread_mutex_unlock(&lock);
        return rc;
    }
    *m = (struct mapping){ptr, size, o, mappings};
    mappings = m;
    o->refs++;
    pthread_mutex_unlock(&lock);
    fg_vmm_track(ptr, size, o->dev < 0 ? fg_vmm_current_dev() : o->dev, o->kind == OBJ_MC ? -1 : o->phys);
    return CUDA_SUCCESS;
}

API CUresult cuMemUnmap(CUdeviceptr ptr, size_t size)
{
    int found = 0;
    pthread_mutex_lock(&lock);
    for (struct mapping **m = &mappings; *m;) {
        struct mapping *cur = *m;
        if (cur->base >= ptr && cur->base + cur->size <= ptr + size) {
            *m = cur->next;
            mmap((void *)cur->base, cur->size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0);
            fg_vmm_untrack(cur->base);
            obj_put(cur->obj);
            free(cur);
            found = 1;
        } else {
            m = &cur->next;
        }
    }
    pthread_mutex_unlock(&lock);
    return found ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

API CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const access_desc *desc, size_t count)
{
    if (!desc || !count) return CUDA_ERROR_INVALID_VALUE;
    for (size_t i = 0; i < count; i++)
        if (desc[i].location.type == LOCATION_DEVICE && fg_vmm_phys(desc[i].location.id) < 0) return CUDA_ERROR_INVALID_DEVICE;
    pthread_mutex_lock(&lock);
    int ok = 1;
    for (uintptr_t p = ptr; p < ptr + size && ok; p += GRANULARITY) ok = mapping_at(p) != NULL;
    pthread_mutex_unlock(&lock);
    return ok ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

API CUresult cuMemGetAccess(unsigned long long *flags, const void *location, CUdeviceptr ptr)
{
    (void)location;
    if (!flags) return CUDA_ERROR_INVALID_VALUE;
    pthread_mutex_lock(&lock);
    *flags = mapping_at(ptr) ? 3 : 0; /* PROT_READWRITE */
    pthread_mutex_unlock(&lock);
    return CUDA_SUCCESS;
}

API CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *h, void *addr)
{
    if (!h) return CUDA_ERROR_INVALID_VALUE;
    pthread_mutex_lock(&lock);
    struct mapping *m = mapping_at((uintptr_t)addr);
    if (m) {
        m->obj->refs++;
        *h = (CUmemGenericAllocationHandle)(uintptr_t)m->obj;
    }
    pthread_mutex_unlock(&lock);
    return m ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

/* ---- multicast --------------------------------------------------------- */

API CUresult cuMulticastGetGranularity(size_t *g, const mc_prop *prop, int option)
{
    (void)option;
    if (!g || !prop) return CUDA_ERROR_INVALID_VALUE;
    *g = GRANULARITY;
    return CUDA_SUCCESS;
}

API CUresult cuMulticastCreate(CUmemGenericAllocationHandle *h, const mc_prop *prop)
{
    if (!h || !prop || !prop->numDevices || !round_ok(prop->size)) return CUDA_ERROR_INVALID_VALUE;
    struct obj *o;
    CUresult rc = create(h, OBJ_MC, prop->size, prop->handleTypes, fg_vmm_current_dev(), &o);
    if (rc == CUDA_SUCCESS) o->mc_devices = prop->numDevices;
    return rc;
}

/* A device joins a multicast object only from the object's partition. */
static int may_join(const struct obj *mc, CUdevice dev)
{
    return !(mc->types & HANDLE_FABRIC) || in_partition(dev, &mc->fabric);
}

API CUresult cuMulticastAddDevice(CUmemGenericAllocationHandle h, CUdevice dev)
{
    struct obj *o = obj_of(h);
    if (!o || o->kind != OBJ_MC) return CUDA_ERROR_INVALID_HANDLE;
    if (fg_vmm_phys(dev) < 0) return CUDA_ERROR_INVALID_DEVICE;
    if (!may_join(o, dev)) return CUDA_ERROR_NOT_PERMITTED;
    pthread_mutex_lock(&lock);
    CUresult rc = o->mc_devices && o->mc_added >= o->mc_devices ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
    if (rc == CUDA_SUCCESS) o->mc_added++;
    pthread_mutex_unlock(&lock);
    return rc;
}

static CUresult bind(struct obj *mc, size_t mc_offset, struct obj *mem, size_t mem_offset, size_t size)
{
    if (!mc || mc->kind != OBJ_MC || !mem || mem->kind != OBJ_MEM) return CUDA_ERROR_INVALID_HANDLE;
    if (!round_ok(size) || mc_offset % GRANULARITY || mem_offset % GRANULARITY || mc_offset + size > mc->size ||
        mem_offset + size > mem->size)
        return CUDA_ERROR_INVALID_VALUE;
    if (mem->dev >= 0 && !may_join(mc, mem->dev)) return CUDA_ERROR_NOT_PERMITTED;
    struct binding *b = malloc(sizeof *b);
    if (!b) return CUDA_ERROR_OUT_OF_MEMORY;
    pthread_mutex_lock(&lock);
    *b = (struct binding){mc_offset, size, mem, mc->bindings};
    mc->bindings = b;
    mem->refs++;
    pthread_mutex_unlock(&lock);
    return CUDA_SUCCESS;
}

API CUresult cuMulticastBindMem(CUmemGenericAllocationHandle mc, size_t mc_offset, CUmemGenericAllocationHandle mem,
                                size_t mem_offset, size_t size, unsigned long long flags)
{
    (void)flags;
    return bind(obj_of(mc), mc_offset, obj_of(mem), mem_offset, size);
}

API CUresult cuMulticastBindAddr(CUmemGenericAllocationHandle mc, size_t mc_offset, CUdeviceptr ptr, size_t size,
                                 unsigned long long flags)
{
    (void)flags;
    pthread_mutex_lock(&lock);
    struct mapping *m = mapping_at(ptr);
    struct obj *mem = m ? m->obj : NULL;
    pthread_mutex_unlock(&lock);
    if (!mem) return CUDA_ERROR_INVALID_VALUE;
    return bind(obj_of(mc), mc_offset, mem, ptr - m->base, size);
}

API CUresult cuMulticastUnbind(CUmemGenericAllocationHandle h, CUdevice dev, size_t mc_offset, size_t size)
{
    (void)dev;
    struct obj *o = obj_of(h);
    if (!o || o->kind != OBJ_MC) return CUDA_ERROR_INVALID_HANDLE;
    pthread_mutex_lock(&lock);
    for (struct binding **b = &o->bindings; *b;) {
        if ((*b)->mc_offset >= mc_offset && (*b)->mc_offset + (*b)->size <= mc_offset + size) {
            struct binding *dead = *b;
            *b = dead->next;
            obj_put(dead->mem);
            free(dead);
        } else {
            b = &(*b)->next;
        }
    }
    pthread_mutex_unlock(&lock);
    return CUDA_SUCCESS;
}
