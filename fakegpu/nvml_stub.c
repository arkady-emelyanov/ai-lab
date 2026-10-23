/*
 * Fake NVML (libnvidia-ml.so.1).
 *
 * Every exported nvml* symbol exists (see stubs generated from
 * symbols/nvml.syms) and returns NVML_SUCCESS without doing anything.
 * The functions below fill in results for the queries that nvidia-smi,
 * Slurm's gpu/nvml plugin and the NVIDIA k8s/DRA stack rely on.
 *
 * Telemetry (utilisation, memory, processes, power, energy, temperature,
 * clocks) comes from the tray's occupancy state, which libcuda fills as
 * processes allocate memory and run simulated work.
 *
 * Topology: every GPU has FG_NVLINKS NVLink5 links, all going to NVSwitch,
 * active unless the tray's BMC disabled them (sideband_dir), and reports the
 * configured fabric cluster UUID / clique, so all trays configured with the
 * same cluster_uuid form one NVLink domain.
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <math.h>
#include "mig.h"

typedef int nvmlReturn_t;

#define NVML_SUCCESS 0
#define NVML_ERROR_INVALID_ARGUMENT 2
#define NVML_ERROR_NOT_SUPPORTED 3
#define NVML_ERROR_NOT_FOUND 6
#define NVML_ERROR_INSUFFICIENT_SIZE 7

#define API __attribute__((visibility("default")))

struct fake_dev { int idx; };
typedef struct fake_dev *nvmlDevice_t;
static struct fake_dev devs[FG_MAX_GPUS];

/* Handles are per visible GPU; dev_idx gives the physical index that
 * identity and state are keyed by (fakegpu.h). */
static int dev_idx(nvmlDevice_t d)
{
    if (d < devs || d >= devs + fg_count()) return -1;
    return fg_phys((int)(d - devs));
}

#define CHECK_DEV(d)                                                \
    int idx = dev_idx(d);                                           \
    if (idx < 0) return NVML_ERROR_INVALID_ARGUMENT;                \
    (void)idx

/* MIG handles (MIG section below): a GPU instance, a compute instance, and a
 * MIG device (an nvmlDevice_t of its own), each naming its GPU by visible
 * index and its instances by id. Valid while the instances exist. */
struct mig_handle { unsigned magic; int vis, gi, ci; };
#define GI_MAGIC 0x6151A7CEu
#define CI_MAGIC 0xC151A7CEu
#define MIGDEV_MAGIC 0x41D0E71Cu
static struct mig_handle gi_handles[FG_MAX_GPUS][FG_MIG_SLOTS];
static struct mig_handle ci_handles[FG_MAX_GPUS][FG_MIG_SLOTS][FG_MIG_CIS];
static struct mig_handle mig_dev_handles[FG_MAX_GPUS][FG_MIG_SLOTS][FG_MIG_CIS];

/* The MIG device a handle is, if it is one and its instances exist. */
static struct mig_handle *mig_dev_of(const void *d)
{
    const struct mig_handle *h = d;
    if (h < &mig_dev_handles[0][0][0] || h >= &mig_dev_handles[0][0][0] + sizeof mig_dev_handles / sizeof *h)
        return NULL;
    if (h->magic != MIGDEV_MAGIC || !mig_ci_exists(fg_phys(h->vis), h->gi, h->ci, NULL, NULL)) return NULL;
    return (struct mig_handle *)h;
}

static nvmlReturn_t copy_str(char *dst, unsigned len, const char *src);
static nvmlReturn_t mig_dev_uuid(struct mig_handle *h, char *uuid, unsigned len);
static nvmlReturn_t mig_dev_name(struct mig_handle *h, char *name, unsigned len);
static unsigned long long mig_dev_mem(struct mig_handle *h);
static int mig_dev_index(struct mig_handle *h);
static int mig_find_uuid(const char *uuid, nvmlDevice_t *d);

static nvmlReturn_t copy_str(char *dst, unsigned len, const char *src)
{
    if (!dst) return NVML_ERROR_INVALID_ARGUMENT;
    if (strlen(src) + 1 > len) return NVML_ERROR_INSUFFICIENT_SIZE;
    memcpy(dst, src, strlen(src) + 1);
    return NVML_SUCCESS;
}

/* ---- init / system ----------------------------------------------------- */

API nvmlReturn_t nvmlInit_v2(void) { return NVML_SUCCESS; }
API nvmlReturn_t nvmlInit(void) { return NVML_SUCCESS; }
API nvmlReturn_t nvmlInitWithFlags(unsigned flags) { (void)flags; return NVML_SUCCESS; }
API nvmlReturn_t nvmlShutdown(void) { return NVML_SUCCESS; }

API const char *nvmlErrorString(nvmlReturn_t r)
{
    switch (r) {
    case NVML_SUCCESS: return "Success";
    case NVML_ERROR_INVALID_ARGUMENT: return "Invalid Argument";
    case NVML_ERROR_NOT_SUPPORTED: return "Not Supported";
    case NVML_ERROR_NOT_FOUND: return "Not Found";
    case NVML_ERROR_INSUFFICIENT_SIZE: return "Insufficient Size";
    case MIG_IN_USE: return "In use by another client";
    case MIG_INSUFFICIENT_RESOURCES: return "Insufficient Resources";
    default: return "Unknown Error";
    }
}

API nvmlReturn_t nvmlSystemGetDriverVersion(char *v, unsigned len) { return copy_str(v, len, fg_driver_version()); }

/* NVML's version: the CUDA major version, then the driver's (e.g. 13.580.95.05). */
API nvmlReturn_t nvmlSystemGetNVMLVersion(char *v, unsigned len)
{
    char s[64];
    snprintf(s, sizeof s, "%d.%s", fg_cuda_version() / 1000, fg_driver_version());
    return copy_str(v, len, s);
}

API nvmlReturn_t nvmlSystemGetCudaDriverVersion(int *v)
{
    if (!v) return NVML_ERROR_INVALID_ARGUMENT;
    *v = fg_cuda_version();
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int *v) { return nvmlSystemGetCudaDriverVersion(v); }

/* Like the real NVML: the executable path (first word of the command line). */
API nvmlReturn_t nvmlSystemGetProcessName(unsigned pid, char *name, unsigned len)
{
    char path[64], buf[256] = "";
    snprintf(path, sizeof path, "/proc/%u/cmdline", pid);
    FILE *f = fopen(path, "r");
    if (!f) return NVML_ERROR_NOT_FOUND;
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;
    return copy_str(name, len, buf);
}

/* ---- device enumeration ------------------------------------------------ */

API nvmlReturn_t nvmlDeviceGetCount_v2(unsigned *n)
{
    if (!n) return NVML_ERROR_INVALID_ARGUMENT;
    *n = (unsigned)fg_count();
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetCount(unsigned *n) { return nvmlDeviceGetCount_v2(n); }

API nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned i, nvmlDevice_t *d)
{
    if (!d || i >= (unsigned)fg_count()) return NVML_ERROR_INVALID_ARGUMENT;
    devs[i].idx = (int)i;
    *d = &devs[i];
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned i, nvmlDevice_t *d) { return nvmlDeviceGetHandleByIndex_v2(i, d); }

API nvmlReturn_t nvmlDeviceGetHandleByUUID(const char *uuid, nvmlDevice_t *d)
{
    char u[64];
    if (!uuid || !d) return NVML_ERROR_INVALID_ARGUMENT;
    if (!strncasecmp(uuid, "MIG-", 4)) return mig_find_uuid(uuid, d) ? NVML_SUCCESS : NVML_ERROR_NOT_FOUND;
    for (int i = 0; i < fg_count(); i++) {
        fg_uuid_str(fg_phys(i), u, sizeof u);
        if (!strcasecmp(u, uuid)) return nvmlDeviceGetHandleByIndex_v2((unsigned)i, d);
    }
    return NVML_ERROR_NOT_FOUND;
}

API nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char *bus, nvmlDevice_t *d)
{
    char b[32], l[32];
    if (!bus || !d) return NVML_ERROR_INVALID_ARGUMENT;
    for (int i = 0; i < fg_count(); i++) {
        fg_pci_str(fg_phys(i), b, sizeof b, 0);
        fg_pci_str(fg_phys(i), l, sizeof l, 1);
        if (!strcasecmp(b, bus) || !strcasecmp(l, bus)) return nvmlDeviceGetHandleByIndex_v2((unsigned)i, d);
    }
    return NVML_ERROR_NOT_FOUND;
}

API nvmlReturn_t nvmlDeviceGetHandleByPciBusId(const char *bus, nvmlDevice_t *d) { return nvmlDeviceGetHandleByPciBusId_v2(bus, d); }

API nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t d, unsigned *i)
{
    struct mig_handle *mh = mig_dev_of(d);
    if (mh) {
        if (!i) return NVML_ERROR_INVALID_ARGUMENT;
        *i = (unsigned)mig_dev_index(mh);
        return NVML_SUCCESS;
    }
    CHECK_DEV(d);
    if (!i) return NVML_ERROR_INVALID_ARGUMENT;
    *i = (unsigned)(d - devs);
    return NVML_SUCCESS;
}

/* The /dev/nvidiaN number (Slurm builds the device path from it): physical. */
API nvmlReturn_t nvmlDeviceGetMinorNumber(nvmlDevice_t d, unsigned *m)
{
    if (mig_dev_of(d)) return NVML_ERROR_NOT_SUPPORTED; /* MIG devices have capability nodes, not a minor */
    CHECK_DEV(d);
    if (!m) return NVML_ERROR_INVALID_ARGUMENT;
    *m = (unsigned)idx;
    return NVML_SUCCESS;
}

/* ---- identity ---------------------------------------------------------- */

API nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t d, char *name, unsigned len)
{
    struct mig_handle *mh = mig_dev_of(d);
    if (mh) return mig_dev_name(mh, name, len);
    CHECK_DEV(d);
    return copy_str(name, len, fg_name());
}

API nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t d, char *uuid, unsigned len)
{
    char u[64];
    struct mig_handle *mh = mig_dev_of(d);
    if (mh) return mig_dev_uuid(mh, uuid, len);
    CHECK_DEV(d);
    fg_uuid_str(idx, u, sizeof u);
    return copy_str(uuid, len, u);
}

API nvmlReturn_t nvmlDeviceGetSerial(nvmlDevice_t d, char *s, unsigned len)
{
    char buf[32];
    unsigned char u[16];
    CHECK_DEV(d);
    fg_uuid(idx, u);
    snprintf(buf, sizeof buf, "16523240%02u%02u%02u", u[0] % 100, u[1] % 100, u[2] % 100);
    return copy_str(s, len, buf);
}

API nvmlReturn_t nvmlDeviceGetBoardPartNumber(nvmlDevice_t d, char *s, unsigned len)
{
    CHECK_DEV(d);
    return copy_str(s, len, "699-2G548-0200-000");
}

API nvmlReturn_t nvmlDeviceGetVbiosVersion(nvmlDevice_t d, char *s, unsigned len)
{
    CHECK_DEV(d);
    return copy_str(s, len, fg_vbios_version());
}

API nvmlReturn_t nvmlDeviceGetBrand(nvmlDevice_t d, int *b)
{
    CHECK_DEV(d);
    if (!b) return NVML_ERROR_INVALID_ARGUMENT;
    *b = 2; /* NVML_BRAND_TESLA */
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetArchitecture(nvmlDevice_t d, unsigned *a)
{
    CHECK_DEV(d);
    if (!a) return NVML_ERROR_INVALID_ARGUMENT;
    *a = 10; /* NVML_DEVICE_ARCH_BLACKWELL */
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t d, int *major, int *minor)
{
    CHECK_DEV(d);
    if (!major || !minor) return NVML_ERROR_INVALID_ARGUMENT;
    *major = FG_CC_MAJOR;
    *minor = FG_CC_MINOR;
    return NVML_SUCCESS;
}

typedef struct {
    char busIdLegacy[16];
    unsigned domain, bus, device, pciDeviceId, pciSubSystemId;
    char busId[32];
} nvmlPciInfo_t;

static void fill_pci(nvmlPciInfo_t *p, unsigned domain, unsigned bus, unsigned devid, unsigned subsys)
{
    memset(p, 0, sizeof *p);
    p->domain = domain;
    p->bus = bus;
    p->pciDeviceId = devid;
    p->pciSubSystemId = subsys;
    snprintf(p->busIdLegacy, sizeof p->busIdLegacy, "%04X:%02X:00.0", domain, bus);
    snprintf(p->busId, sizeof p->busId, "%08X:%02X:00.0", domain, bus);
}

API nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t d, nvmlPciInfo_t *p)
{
    CHECK_DEV(d);
    if (!p) return NVML_ERROR_INVALID_ARGUMENT;
    fill_pci(p, fg_pci_domain[idx], FG_PCI_BUS, FG_PCI_DEVICE_ID, FG_PCI_SUBSYSTEM_ID);
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetPciInfo_v2(nvmlDevice_t d, nvmlPciInfo_t *p) { return nvmlDeviceGetPciInfo_v3(d, p); }
API nvmlReturn_t nvmlDeviceGetPciInfo(nvmlDevice_t d, nvmlPciInfo_t *p) { return nvmlDeviceGetPciInfo_v3(d, p); }

/* ---- telemetry model ---------------------------------------------------- */

/* Power: the GPU profile (fakegpu.h). */
#define IDLE_MW (fg_idle_power_mw())
#define MAX_MW (fg_max_power_mw())
#define IDLE_MC 32000.0     /* milli-degrees C */
#define FULL_MC 75000.0
#define TEMP_TAU_NS 20e9    /* thermal time constant */
#define UTIL_WINDOW_NS 1000000000ULL

/* Utilisation over the last window of at least a second, shared by all
 * readers so concurrent tools see the same value. */
static unsigned utilization(int idx)
{
    struct fg_occupancy *o = fg_occupancy();
    if (!o) return 0;
    struct fg_gpu_state *g = &o->gpu[idx];
    uint64_t now = fg_now_ns(), busy = __atomic_load_n(&g->busy_ns, __ATOMIC_RELAXED);
    uint64_t since = __atomic_load_n(&g->sample_ns, __ATOMIC_RELAXED);
    if (since == 0) {
        __atomic_store_n(&g->sample_ns, now, __ATOMIC_RELAXED);
        __atomic_store_n(&g->sample_busy, busy, __ATOMIC_RELAXED);
        return 0;
    }
    if (now - since >= UTIL_WINDOW_NS) {
        double u = (double)(busy - __atomic_load_n(&g->sample_busy, __ATOMIC_RELAXED)) / (double)(now - since) * 100.0;
        __atomic_store_n(&g->util, (uint32_t)(u > 100 ? 100 : u), __ATOMIC_RELAXED);
        __atomic_store_n(&g->sample_ns, now, __ATOMIC_RELAXED);
        __atomic_store_n(&g->sample_busy, busy, __ATOMIC_RELAXED);
    }
    return __atomic_load_n(&g->util, __ATOMIC_RELAXED);
}

/* Small deterministic wobble so readings look alive: +-1 at amplitude 1. */
static double jitter(int idx, double hz)
{
    double t = (double)fg_now_ns() / 1e9;
    return sin(t * hz * 6.283 + idx * 1.7) * 0.6 + sin(t * hz * 2.7 * 6.283 + idx) * 0.4;
}

static unsigned power_mw(int idx)
{
    double u = utilization(idx) / 100.0;
    double p = IDLE_MW + u * (MAX_MW - IDLE_MW);
    return (unsigned)(p * (1.0 + 0.02 * jitter(idx, 0.5)));
}

/* Temperature approaches its load-dependent target with a thermal lag. */
static unsigned temperature_mc(int idx)
{
    struct fg_occupancy *o = fg_occupancy();
    double target = IDLE_MC + utilization(idx) / 100.0 * (FULL_MC - IDLE_MC);
    if (!o) return (unsigned)target;
    struct fg_gpu_state *g = &o->gpu[idx];
    uint64_t now = fg_now_ns(), last = __atomic_load_n(&g->temp_ns, __ATOMIC_RELAXED);
    double cur = __atomic_load_n(&g->temp_mc, __ATOMIC_RELAXED);
    if (last == 0 || cur == 0) cur = IDLE_MC;
    else cur += (target - cur) * (1.0 - exp(-(double)(now - last) / TEMP_TAU_NS));
    __atomic_store_n(&g->temp_mc, (uint32_t)cur, __ATOMIC_RELAXED);
    __atomic_store_n(&g->temp_ns, now, __ATOMIC_RELAXED);
    return (unsigned)(cur + 400.0 * jitter(idx, 0.2));
}

/* Energy since the state was created: idle draw all along plus the extra
 * draw for every busy nanosecond. */
static unsigned long long energy_mj(int idx)
{
    struct fg_occupancy *o = fg_occupancy();
    static uint64_t start;
    if (!start) start = fg_now_ns();
    double busy = o ? (double)__atomic_load_n(&o->gpu[idx].busy_ns, __ATOMIC_RELAXED) : 0;
    double secs = (double)(fg_now_ns() - start) / 1e9;
    return (unsigned long long)(IDLE_MW * secs + (MAX_MW - IDLE_MW) * busy / 1e9);
}

static unsigned long long mem_used(int idx)
{
    fg_proc_cleanup(0);
    return fg_gpu_mem_used(idx);
}

/* ---- memory / telemetry ------------------------------------------------ */

typedef struct { unsigned long long total, free, used; } nvmlMemory_t;
typedef struct { unsigned version; unsigned long long total, reserved, free, used; } nvmlMemory_v2_t;

#define RESERVED_BYTES (512ULL << 20)

API nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t d, nvmlMemory_t *m)
{
    struct mig_handle *mh = mig_dev_of(d);
    if (mh) {
        if (!m) return NVML_ERROR_INVALID_ARGUMENT;
        m->total = mig_dev_mem(mh);
        m->used = 16ULL << 20; /* a MIG device's reserved memory, as nvidia-smi shows it idle */
        m->free = m->total - m->used;
        return NVML_SUCCESS;
    }
    CHECK_DEV(d);
    if (!m) return NVML_ERROR_INVALID_ARGUMENT;
    m->total = fg_mem_bytes();
    m->used = RESERVED_BYTES + mem_used(idx);
    m->free = m->used < m->total ? m->total - m->used : 0;
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t d, nvmlMemory_v2_t *m)
{
    struct mig_handle *mh = mig_dev_of(d);
    if (mh) {
        if (!m) return NVML_ERROR_INVALID_ARGUMENT;
        m->total = mig_dev_mem(mh);
        m->reserved = 16ULL << 20;
        m->used = 0;
        m->free = m->total - m->reserved;
        return NVML_SUCCESS;
    }
    CHECK_DEV(d);
    if (!m) return NVML_ERROR_INVALID_ARGUMENT;
    m->total = fg_mem_bytes();
    m->reserved = RESERVED_BYTES;
    m->used = mem_used(idx);
    m->free = m->total > m->reserved + m->used ? m->total - m->reserved - m->used : 0;
    return NVML_SUCCESS;
}

typedef struct { unsigned long long bar1Total, bar1Free, bar1Used; } nvmlBAR1Memory_t;

API nvmlReturn_t nvmlDeviceGetBAR1MemoryInfo(nvmlDevice_t d, nvmlBAR1Memory_t *m)
{
    CHECK_DEV(d);
    if (!m) return NVML_ERROR_INVALID_ARGUMENT;
    m->bar1Total = m->bar1Free = fg_mem_bytes();
    m->bar1Used = 0;
    return NVML_SUCCESS;
}

typedef struct { unsigned gpu, memory; } nvmlUtilization_t;

API nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t d, nvmlUtilization_t *u)
{
    CHECK_DEV(d);
    if (!u) return NVML_ERROR_INVALID_ARGUMENT;
    u->gpu = utilization(idx);
    u->memory = u->gpu * 2 / 3; /* memory-bound share of the busy time */
    return NVML_SUCCESS;
}

static nvmlReturn_t set_uint(nvmlDevice_t d, unsigned *out, unsigned v)
{
    CHECK_DEV(d);
    if (!out) return NVML_ERROR_INVALID_ARGUMENT;
    *out = v;
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t d, int sensor, unsigned *t)
{
    (void)sensor;
    int idx = dev_idx(d);
    return set_uint(d, t, idx < 0 ? 0 : (temperature_mc(idx) + 500) / 1000);
}

/* nvmlTemperature_t: version, sensorType, temperature (int). */
API nvmlReturn_t nvmlDeviceGetTemperatureV(nvmlDevice_t d, void *t)
{
    CHECK_DEV(d);
    if (!t) return NVML_ERROR_INVALID_ARGUMENT;
    ((int *)t)[2] = (int)((temperature_mc(idx) + 500) / 1000);
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t d, unsigned *mw)
{
    int idx = dev_idx(d);
    return set_uint(d, mw, idx < 0 ? 0 : power_mw(idx));
}

API nvmlReturn_t nvmlDeviceGetTotalEnergyConsumption(nvmlDevice_t d, unsigned long long *mj)
{
    CHECK_DEV(d);
    if (!mj) return NVML_ERROR_INVALID_ARGUMENT;
    *mj = energy_mj(idx);
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetCurrentClocksEventReasons(nvmlDevice_t d, unsigned long long *r)
{
    CHECK_DEV(d);
    if (!r) return NVML_ERROR_INVALID_ARGUMENT;
    *r = utilization(idx) ? 0 : 0x1ULL; /* GpuIdle when nothing runs */
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetCurrentClocksThrottleReasons(nvmlDevice_t d, unsigned long long *r)
{
    return nvmlDeviceGetCurrentClocksEventReasons(d, r);
}
API nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t d, unsigned *mw) { return set_uint(d, mw, (unsigned)fg_power_limit_mw()); }
API nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t d, unsigned *mw) { return set_uint(d, mw, (unsigned)fg_power_limit_mw()); }
API nvmlReturn_t nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice_t d, unsigned *mw) { return set_uint(d, mw, (unsigned)fg_power_limit_mw()); }
API nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t d, unsigned *s) { (void)s; CHECK_DEV(d); return NVML_ERROR_NOT_SUPPORTED; } /* liquid cooled */
API nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t d, unsigned *p)
{
    int idx = dev_idx(d);
    return set_uint(d, p, idx >= 0 && utilization(idx) ? 0 : 8); /* P0 busy, P8 idle */
}
API nvmlReturn_t nvmlDeviceGetPowerState(nvmlDevice_t d, unsigned *p) { return set_uint(d, p, 0); }
API nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t d, unsigned *m) { return set_uint(d, m, 0); }
API nvmlReturn_t nvmlDeviceGetPersistenceMode(nvmlDevice_t d, unsigned *m) { return set_uint(d, m, 1); }
API nvmlReturn_t nvmlDeviceGetDisplayMode(nvmlDevice_t d, unsigned *m) { return set_uint(d, m, 0); }
API nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t d, unsigned *m) { return set_uint(d, m, 0); }

API nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t d, unsigned *cur, unsigned *pend)
{
    CHECK_DEV(d);
    if (!cur || !pend) return NVML_ERROR_INVALID_ARGUMENT;
    *cur = *pend = 1;
    return NVML_SUCCESS;
}

/* Slurm treats a non-zero MIG mode as "MIG enabled". */
/* MIG: see the MIG section at the end. */

/* Clocks: graphics=0 sm=1 mem=2 video=3 */
static unsigned clock_mhz(int type, int max)
{
    switch (type) {
    case 2: return 3996;
    case 3: return max ? 1530 : 1320;
    default: return max ? 1965 : 1155;
    }
}

API nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t d, int type, unsigned *c)
{
    int idx = dev_idx(d);
    return set_uint(d, c, clock_mhz(type, idx >= 0 && utilization(idx) > 0));
}
API nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t d, int type, unsigned *c) { return set_uint(d, c, clock_mhz(type, 1)); }
API nvmlReturn_t nvmlDeviceGetApplicationsClock(nvmlDevice_t d, int type, unsigned *c) { return set_uint(d, c, clock_mhz(type, 1)); }
API nvmlReturn_t nvmlDeviceGetDefaultApplicationsClock(nvmlDevice_t d, int type, unsigned *c) { return set_uint(d, c, clock_mhz(type, 1)); }

API nvmlReturn_t nvmlDeviceGetSupportedMemoryClocks(nvmlDevice_t d, unsigned *count, unsigned *mhz)
{
    CHECK_DEV(d);
    if (!count) return NVML_ERROR_INVALID_ARGUMENT;
    if (*count < 1 || !mhz) { *count = 1; return NVML_ERROR_INSUFFICIENT_SIZE; }
    *count = 1;
    mhz[0] = clock_mhz(2, 1);
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetSupportedGraphicsClocks(nvmlDevice_t d, unsigned mem, unsigned *count, unsigned *mhz)
{
    static const unsigned clocks[] = {1965, 1800, 1500, 1155};
    const unsigned n = sizeof clocks / sizeof *clocks;
    (void)mem;
    CHECK_DEV(d);
    if (!count) return NVML_ERROR_INVALID_ARGUMENT;
    if (*count < n || !mhz) { *count = n; return NVML_ERROR_INSUFFICIENT_SIZE; }
    *count = n;
    memcpy(mhz, clocks, sizeof clocks);
    return NVML_SUCCESS;
}

/* Processes come from the occupancy state. v1 entries are 16 bytes; v2/v3
 * add GPU and compute instance ids (24 bytes). */
typedef struct { unsigned pid; unsigned long long usedGpuMemory; } nvmlProcessInfo_v1_t;
typedef struct { unsigned pid; unsigned long long usedGpuMemory; unsigned gpuInstanceId, computeInstanceId; } nvmlProcessInfo_v2_t;

/* The PID under which this process sees a slot's owner: its own PID if the
 * owner is in our PID namespace; otherwise the /proc entry whose namespace
 * and innermost NSpid match (a container's process seen from the tray).
 * 0 when we cannot see the owner (another container), as real NVML in a
 * container lists only what it can see. */
static int32_t visible_pid(struct fg_occupancy *o, int i)
{
    int32_t pid = __atomic_load_n(&o->proc[i].pid, __ATOMIC_ACQUIRE);
    uint64_t ns = __atomic_load_n(&o->pidns[i], __ATOMIC_ACQUIRE);
    if (ns == fg_pidns()) return pid;
    DIR *dir = opendir("/proc");
    if (!dir) return 0;
    int32_t found = 0;
    struct dirent *e;
    while (!found && (e = readdir(dir))) {
        char path[300], line[256];
        struct stat st;
        if (e->d_name[0] < '1' || e->d_name[0] > '9') continue;
        snprintf(path, sizeof path, "/proc/%s/ns/pid", e->d_name);
        if (stat(path, &st) != 0 || (uint64_t)st.st_ino != ns) continue;
        snprintf(path, sizeof path, "/proc/%s/status", e->d_name);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        while (fgets(line, sizeof line, f)) {
            if (strncmp(line, "NSpid:", 6)) continue;
            char *last = strrchr(line, '\t');
            if (last && atoi(last + 1) == pid) found = atoi(e->d_name);
            break;
        }
        fclose(f);
    }
    closedir(dir);
    return found;
}

static nvmlReturn_t procs(nvmlDevice_t d, unsigned *count, void *infos, int v2)
{
    CHECK_DEV(d);
    if (!count) return NVML_ERROR_INVALID_ARGUMENT;
    fg_proc_cleanup(0);
    struct fg_occupancy *o = fg_occupancy();
    unsigned n = 0, cap = infos ? *count : 0;
    for (int i = 0; o && i < FG_MAX_PROCS; i++) {
        if (o->proc[i].gpu != idx || !fg_slot_alive(o, i)) continue;
        int32_t pid = visible_pid(o, i);
        if (!pid) continue;
        if (n < cap) {
            unsigned long long mem = __atomic_load_n(&o->proc[i].mem, __ATOMIC_RELAXED);
            if (v2) ((nvmlProcessInfo_v2_t *)infos)[n] = (nvmlProcessInfo_v2_t){(unsigned)pid, mem, 0xFFFFFFFFu, 0xFFFFFFFFu};
            else ((nvmlProcessInfo_v1_t *)infos)[n] = (nvmlProcessInfo_v1_t){(unsigned)pid, mem};
        }
        n++;
    }
    *count = n;
    return n > cap ? NVML_ERROR_INSUFFICIENT_SIZE : NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t d, unsigned *n, void *i) { return procs(d, n, i, 1); }
API nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v2(nvmlDevice_t d, unsigned *n, void *i) { return procs(d, n, i, 1); }
API nvmlReturn_t nvmlDeviceGetComputeRunningProcesses(nvmlDevice_t d, unsigned *n, void *i) { return procs(d, n, i, 0); }

typedef struct {
    unsigned pid;
    unsigned long long timeStamp; /* microseconds */
    unsigned smUtil, memUtil, encUtil, decUtil;
} nvmlProcessUtilizationSample_t;

/* One sample per process on the GPU, taken now. Busy time is kept per GPU,
 * not per process, so the GPU's utilisation is split evenly between its
 * processes: exact for one process (a whole GPU), an even share when a GPU
 * is shared (GPU fractions). NULL asks for the count, as with the real one. */
API nvmlReturn_t nvmlDeviceGetProcessUtilization(nvmlDevice_t d, nvmlProcessUtilizationSample_t *samples,
                                                 unsigned *count, unsigned long long last_seen)
{
    CHECK_DEV(d);
    if (!count) return NVML_ERROR_INVALID_ARGUMENT;
    fg_proc_cleanup(0);
    struct fg_occupancy *o = fg_occupancy();
    unsigned long long now = fg_now_ns() / 1000;
    int32_t pids[FG_MAX_PROCS];
    unsigned n = 0;
    for (int i = 0; o && i < FG_MAX_PROCS; i++) {
        if (o->proc[i].gpu != idx || !fg_slot_alive(o, i)) continue;
        int32_t pid = visible_pid(o, i);
        if (pid) pids[n++] = pid;
    }
    if (!n || now <= last_seen) {
        *count = 0;
        return NVML_ERROR_NOT_FOUND;
    }
    if (!samples || *count < n) {
        *count = n;
        return NVML_ERROR_INSUFFICIENT_SIZE;
    }
    unsigned sm = utilization(idx) / n;
    for (unsigned k = 0; k < n; k++)
        samples[k] = (nvmlProcessUtilizationSample_t){(unsigned)pids[k], now, sm, sm * 2 / 3, 0, 0};
    *count = n;
    return NVML_SUCCESS;
}

/* Graphics and MPS clients never exist here. */
static nvmlReturn_t no_procs(nvmlDevice_t d, unsigned *count)
{
    CHECK_DEV(d);
    if (!count) return NVML_ERROR_INVALID_ARGUMENT;
    *count = 0;
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v3(nvmlDevice_t d, unsigned *n, void *i) { (void)i; return no_procs(d, n); }
API nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v2(nvmlDevice_t d, unsigned *n, void *i) { (void)i; return no_procs(d, n); }
API nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses(nvmlDevice_t d, unsigned *n, void *i) { (void)i; return no_procs(d, n); }
API nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v3(nvmlDevice_t d, unsigned *n, void *i) { (void)i; return no_procs(d, n); }

/* ---- affinity ---------------------------------------------------------- */

/* Every GPU is local to every CPU of the (container's) node. */
/* The cores of the GPU's own Grace CPU: the tray's CPUs (the container's,
 * as lxcfs shows them in /sys/devices/system/cpu/online) split evenly between
 * its Grace CPUs, the same split numactl -H reports (GB200: GPUs 0-1 on the
 * first Grace's cores, 2-3 on the second's). */
static nvmlReturn_t grace_cpus(nvmlDevice_t d, unsigned size, unsigned long *set)
{
    CHECK_DEV(d);
    if (!set || !size) return NVML_ERROR_INVALID_ARGUMENT;
    memset(set, 0, size * sizeof *set);
    int cpus[4096], n = 0;
    FILE *f = fopen("/sys/devices/system/cpu/online", "r");
    if (f) {
        int a, b;
        char sep;
        while (n < 4096 && fscanf(f, "%d", &a) == 1) {
            b = a;
            if (fscanf(f, "%c", &sep) == 1 && sep == '-') {
                if (fscanf(f, "%d", &b) != 1) break;
                if (fscanf(f, "%c", &sep) != 1) sep = '\n';
            }
            for (int c = a; c <= b && n < 4096; c++) cpus[n++] = c;
            if (sep != ',') break;
        }
        fclose(f);
    }
    if (!n) /* no sysfs: all configured CPUs */
        for (long c = 0; c < sysconf(_SC_NPROCESSORS_CONF) && n < 4096; c++) cpus[n++] = (int)c;
    int graces = (fg_count() + 1) / 2;
    if (graces < 1) graces = 1;
    int per = (n + graces - 1) / graces, g = fg_grace_of(idx);
    const int bits = (int)(8 * sizeof *set);
    for (int i = g * per; i < (g + 1) * per && i < n; i++)
        if (cpus[i] / bits < (int)size) set[cpus[i] / bits] |= 1UL << (cpus[i] % bits);
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetCpuAffinity(nvmlDevice_t d, unsigned size, unsigned long *set) { return grace_cpus(d, size, set); }

API nvmlReturn_t nvmlDeviceGetCpuAffinityWithinScope(nvmlDevice_t d, unsigned size, unsigned long *set, unsigned scope)
{
    (void)scope;
    return grace_cpus(d, size, set);
}

/* The nearest memory is the GPU's own Grace CPU's node (GB200: GPUs 0-1 on
 * node 0, 2-3 on node 1). */
API nvmlReturn_t nvmlDeviceGetMemoryAffinity(nvmlDevice_t d, unsigned size, unsigned long *set, unsigned scope)
{
    CHECK_DEV(d);
    (void)scope;
    if (!set || !size) return NVML_ERROR_INVALID_ARGUMENT;
    memset(set, 0, size * sizeof *set);
    int node = fg_grace_of(idx);
    const int bits = (int)(8 * sizeof *set);
    if (node / bits < (int)size) set[node / bits] = 1UL << (node % bits);
    return NVML_SUCCESS;
}

/* The GPU's own NUMA node: only where its memory is one (coherent memory
 * onlined by the OS, the default without CDMM). */
API nvmlReturn_t nvmlDeviceGetNumaNodeId(nvmlDevice_t d, unsigned *node)
{
    CHECK_DEV(d);
    if (!node) return NVML_ERROR_INVALID_ARGUMENT;
    int n = fg_gpu_numa_node(idx);
    if (n < 0) return NVML_ERROR_NOT_SUPPORTED;
    *node = (unsigned)n;
    return NVML_SUCCESS;
}

/* ---- NVLink / fabric --------------------------------------------------- */

#define NVML_TOPOLOGY_SYSTEM 50
/* The NVLink peer's PCI info. Unverified: 22A3 is the H100 baseboard
 * NVSwitch; GB200's switches sit in the switch trays, off the compute tray's
 * PCI bus (NVIDIA/k8s-test-infra#882), and what NVML reports for them there
 * is not public. */
#define NVSWITCH_PCI_BUS 0x05
#define NVSWITCH_PCI_DEVICE_ID 0x22A310DE

API nvmlReturn_t nvmlDeviceGetNvLinkState(nvmlDevice_t d, unsigned link, unsigned *active)
{
    CHECK_DEV(d);
    if (!active || link >= FG_NVLINKS) return NVML_ERROR_INVALID_ARGUMENT;
    *active = !fg_link_disabled(idx, link);
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetNvLinkVersion(nvmlDevice_t d, unsigned link, unsigned *v)
{
    if (link >= FG_NVLINKS) return NVML_ERROR_INVALID_ARGUMENT;
    return set_uint(d, v, 7); /* NVML_NVLINK_VERSION_5_0 */
}

API nvmlReturn_t nvmlDeviceGetNvLinkCapability(nvmlDevice_t d, unsigned link, int cap, unsigned *res)
{
    (void)cap;
    if (link >= FG_NVLINKS) return NVML_ERROR_INVALID_ARGUMENT;
    return set_uint(d, res, 1);
}

API nvmlReturn_t nvmlDeviceGetNvLinkRemoteDeviceType(nvmlDevice_t d, unsigned link, unsigned *type)
{
    if (link >= FG_NVLINKS) return NVML_ERROR_INVALID_ARGUMENT;
    return set_uint(d, type, 2); /* NVML_NVLINK_DEVICE_TYPE_SWITCH */
}

API nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo_v2(nvmlDevice_t d, unsigned link, nvmlPciInfo_t *p)
{
    CHECK_DEV(d);
    if (!p || link >= FG_NVLINKS) return NVML_ERROR_INVALID_ARGUMENT;
    fill_pci(p, 0, NVSWITCH_PCI_BUS, NVSWITCH_PCI_DEVICE_ID, FG_PCI_SUBSYSTEM_ID);
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo(nvmlDevice_t d, unsigned link, nvmlPciInfo_t *p)
{
    return nvmlDeviceGetNvLinkRemotePciInfo_v2(d, link, p);
}

/* PCIe-wise the GPUs only meet at the system level; they talk over NVLink. */
API nvmlReturn_t nvmlDeviceGetTopologyCommonAncestor(nvmlDevice_t a, nvmlDevice_t b, unsigned *level)
{
    if (dev_idx(a) < 0 || dev_idx(b) < 0 || !level) return NVML_ERROR_INVALID_ARGUMENT;
    *level = a == b ? 0 : NVML_TOPOLOGY_SYSTEM;
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetP2PStatus(nvmlDevice_t a, nvmlDevice_t b, int index, unsigned *status)
{
    (void)index;
    if (dev_idx(a) < 0 || dev_idx(b) < 0 || !status) return NVML_ERROR_INVALID_ARGUMENT;
    *status = 0; /* NVML_P2P_STATUS_OK */
    return NVML_SUCCESS;
}

typedef struct {
    unsigned char clusterUuid[16];
    nvmlReturn_t status;
    unsigned cliqueId;
    unsigned char state;
} nvmlGpuFabricInfo_t;

typedef struct {
    unsigned version;
    unsigned char clusterUuid[16];
    nvmlReturn_t status;
    unsigned cliqueId;
    unsigned char state;
    unsigned healthMask;
    unsigned char healthSummary; /* v3 only */
} nvmlGpuFabricInfoV_t;

/* ---- GPU reset and recovery action --------------------------------------- */

/* The clique the GPU took at its last reset (see struct fg_gpu_reset). A GPU
 * with no reset recorded takes the current partition's clique. */
static unsigned gpu_clique(int idx)
{
    struct fg_occupancy *o = fg_occupancy();
    unsigned partition = fg_partition_clique(idx);
    if (!o) return partition;
    uint32_t taken = __atomic_load_n(&o->reset[idx].clique, __ATOMIC_ACQUIRE);
    if (taken) return taken - 1;
    uint32_t expect = 0;
    if (__atomic_compare_exchange_n(&o->reset[idx].clique, &expect, partition + 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
        return partition;
    return expect - 1;
}

/* nvmlDeviceGpuRecoveryAction_t */
#define RECOVERY_ACTION_NONE 0
#define RECOVERY_ACTION_GPU_RESET 1

/* A partition change waits for a GPU reset, as on GB200. */
static unsigned recovery_action(int idx)
{
    return gpu_clique(idx) != fg_partition_clique(idx) ? RECOVERY_ACTION_GPU_RESET : RECOVERY_ACTION_NONE;
}

/* Lab entry point for nvidia-smi --gpu-reset (not an NVML API): resets
 * physical GPU idx, which takes its partition's clique and starts idle.
 * Returns 0, -1 while processes still use the GPU (the real reset refuses
 * too), -2 without the shared state. */
API int fakegpu_reset_gpu(unsigned idx)
{
    struct fg_occupancy *o = fg_occupancy();
    if (!o || idx >= FG_MAX_GPUS) return -2;
    fg_proc_cleanup(0);
    for (int i = 0; i < FG_MAX_PROCS; i++)
        if (__atomic_load_n(&o->proc[i].pid, __ATOMIC_ACQUIRE) && o->proc[i].gpu == (int32_t)idx && fg_slot_alive(o, i))
            return -1;
    struct fg_gpu_state *g = &o->gpu[idx];
    __atomic_store_n(&g->util, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&g->sample_ns, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&g->temp_mc, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&g->temp_ns, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&o->reset[idx].clique, fg_partition_clique(idx) + 1, __ATOMIC_RELEASE);
    __atomic_store_n(&o->reset[idx].reset_ns, fg_now_ns(), __ATOMIC_RELAXED);
    __atomic_add_fetch(&o->reset[idx].count, 1, __ATOMIC_RELAXED);
    mig_reset((int)idx); /* a pending MIG mode takes effect */
    return 0;
}

/* Lab entry point for the tray's boot (nvidia-smi --fakegpu-driver-load, not
 * an NVML API): the driver loads, so MIG instances are gone; MIG mode stays. */
API void fakegpu_driver_load(void)
{
    for (int i = 0; i < FG_MAX_GPUS; i++) mig_driver_load(i);
}

typedef union { double d; int si; unsigned ui; unsigned long ul; unsigned long long ull; long long sll; unsigned short us; } nvmlValue_t;
typedef struct {
    unsigned fieldId, scopeId;
    long long timestamp, latencyUsec;
    int valueType;
    nvmlReturn_t nvmlReturn;
    nvmlValue_t value;
} nvmlFieldValue_t;

#define NVML_FI_DEV_NVLINK_SPEED_MBPS_COMMON 90
#define NVML_FI_DEV_GET_GPU_RECOVERY_ACTION 230
#define NVML_VALUE_TYPE_UNSIGNED_INT 1

/* Answered: the recovery action and the NVLink speed (MB/s per link and
 * direction, from the GPU profile); other fields are left as the caller set
 * them, as before this was implemented. */
API nvmlReturn_t nvmlDeviceGetFieldValues(nvmlDevice_t d, int count, nvmlFieldValue_t *values)
{
    CHECK_DEV(d);
    if (!values || count < 0) return NVML_ERROR_INVALID_ARGUMENT;
    for (int i = 0; i < count; i++)
        if (values[i].fieldId == NVML_FI_DEV_GET_GPU_RECOVERY_ACTION) {
            values[i].valueType = NVML_VALUE_TYPE_UNSIGNED_INT;
            values[i].value.ui = recovery_action(idx);
            values[i].nvmlReturn = NVML_SUCCESS;
            values[i].timestamp = (long long)time(NULL) * 1000000;
            values[i].latencyUsec = 0;
        } else if (values[i].fieldId == NVML_FI_DEV_NVLINK_SPEED_MBPS_COMMON) {
            values[i].valueType = NVML_VALUE_TYPE_UNSIGNED_INT;
            values[i].value.ui = (unsigned)(fg_nvlink_bytes_per_ns() / fg_nvlinks() * 1000);
            values[i].nvmlReturn = NVML_SUCCESS;
            values[i].timestamp = (long long)time(NULL) * 1000000;
            values[i].latencyUsec = 0;
        }
    return NVML_SUCCESS;
}

#define FABRIC_STATE_COMPLETED 3
/* bandwidth not degraded, no route recovery, routes healthy, no access
 * timeout recovery, configuration correct */
#define FABRIC_HEALTHY_MASK ((2u << 0) | (2u << 2) | (2u << 4) | (2u << 6) | (1u << 8))

API nvmlReturn_t nvmlDeviceGetGpuFabricInfo(nvmlDevice_t d, nvmlGpuFabricInfo_t *f)
{
    CHECK_DEV(d);
    if (!f) return NVML_ERROR_INVALID_ARGUMENT;
    memcpy(f->clusterUuid, fg_cluster_uuid(), 16);
    f->status = NVML_SUCCESS;
    f->cliqueId = gpu_clique(idx);
    f->state = FABRIC_STATE_COMPLETED;
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetGpuFabricInfoV(nvmlDevice_t d, nvmlGpuFabricInfoV_t *f)
{
    CHECK_DEV(d);
    if (!f) return NVML_ERROR_INVALID_ARGUMENT;
    unsigned ver = f->version >> 24;
    memcpy(f->clusterUuid, fg_cluster_uuid(), 16);
    f->status = NVML_SUCCESS;
    f->cliqueId = gpu_clique(idx);
    f->state = FABRIC_STATE_COMPLETED;
    f->healthMask = FABRIC_HEALTHY_MASK;
    if (ver >= 3) f->healthSummary = 1; /* HEALTHY */
    return NVML_SUCCESS;
}

/* ---- events -------------------------------------------------------------- */

/* A healthy GPU raises no events: sets can be created and registered, and a
 * wait blocks for its timeout, then reports NVML_ERROR_TIMEOUT. (A no-op
 * "success" would read as a stream of events to monitoring tools.) */
#define NVML_ERROR_TIMEOUT 10
#define EVENT_TYPE_XID_CRITICAL 0x8ULL
#define EVENT_TYPES_SUPPORTED (EVENT_TYPE_XID_CRITICAL | 0x1ULL /* single-bit ECC */ | 0x2ULL /* double-bit ECC */)

static void sleep_ms(unsigned ms)
{
    struct timespec ts = {.tv_sec = ms / 1000, .tv_nsec = (long)(ms % 1000) * 1000000L};
    while (nanosleep(&ts, &ts) == -1 && errno == EINTR) {}
}

API nvmlReturn_t nvmlEventSetCreate(void **set)
{
    if (!set) return NVML_ERROR_INVALID_ARGUMENT;
    *set = calloc(1, 16);
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlEventSetFree(void *set) { free(set); return NVML_SUCCESS; }

API nvmlReturn_t nvmlDeviceGetSupportedEventTypes(nvmlDevice_t d, unsigned long long *types)
{
    CHECK_DEV(d);
    if (!types) return NVML_ERROR_INVALID_ARGUMENT;
    *types = EVENT_TYPES_SUPPORTED;
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceRegisterEvents(nvmlDevice_t d, unsigned long long types, void *set)
{
    CHECK_DEV(d);
    if (!set) return NVML_ERROR_INVALID_ARGUMENT;
    return types & ~EVENT_TYPES_SUPPORTED ? NVML_ERROR_NOT_SUPPORTED : NVML_SUCCESS;
}

API nvmlReturn_t nvmlEventSetWait_v2(void *set, void *data, unsigned timeoutms)
{
    (void)data;
    if (!set) return NVML_ERROR_INVALID_ARGUMENT;
    sleep_ms(timeoutms);
    return NVML_ERROR_TIMEOUT;
}

API nvmlReturn_t nvmlEventSetWait(void *set, void *data, unsigned timeoutms) { return nvmlEventSetWait_v2(set, data, timeoutms); }

/* nvmlSystemEventSetCreateRequest_v1_t: version, set (out). */
typedef struct { unsigned version; void *set; } nvmlSystemEventSetCreateRequest_t;
/* nvmlSystemEventSetWaitRequest_v1_t */
typedef struct { unsigned version, timeoutms; void *set; void *data; unsigned dataSize, numEvent; } nvmlSystemEventSetWaitRequest_t;

API nvmlReturn_t nvmlSystemEventSetCreate(nvmlSystemEventSetCreateRequest_t *r)
{
    if (!r) return NVML_ERROR_INVALID_ARGUMENT;
    r->set = calloc(1, 16);
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlSystemEventSetWait(nvmlSystemEventSetWaitRequest_t *r)
{
    if (!r) return NVML_ERROR_INVALID_ARGUMENT;
    r->numEvent = 0;
    sleep_ms(r->timeoutms);
    return NVML_ERROR_TIMEOUT;
}

/* ---- MIG (mig.h) ------------------------------------------------------------- */

typedef struct mig_handle *nvmlGpuInstance_t, *nvmlComputeInstance_t;

typedef struct {
    unsigned id, isP2pSupported, sliceCount, instanceCount, multiprocessorCount, copyEngineCount, decoderCount,
        encoderCount, jpegCount, ofaCount;
    unsigned long long memorySizeMB;
} nvmlGpuInstanceProfileInfo_t;

typedef struct {
    unsigned version, id, isP2pSupported, sliceCount, instanceCount, multiprocessorCount, copyEngineCount,
        decoderCount, encoderCount, jpegCount, ofaCount;
    unsigned long long memorySizeMB;
    char name[96];
} nvmlGpuInstanceProfileInfo_v2_t;

typedef struct {
    unsigned version, id, sliceCount, instanceCount, multiprocessorCount, copyEngineCount, decoderCount,
        encoderCount, jpegCount, ofaCount;
    unsigned long long memorySizeMB;
    char name[96];
    unsigned capabilities;
} nvmlGpuInstanceProfileInfo_v3_t;

typedef struct { unsigned start, size; } nvmlGpuInstancePlacement_t, nvmlComputeInstancePlacement_t;

typedef struct {
    nvmlDevice_t device;
    unsigned id, profileId;
    nvmlGpuInstancePlacement_t placement;
} nvmlGpuInstanceInfo_t;

typedef struct {
    unsigned id, sliceCount, instanceCount, multiprocessorCount, sharedCopyEngineCount, sharedDecoderCount,
        sharedEncoderCount, sharedJpegCount, sharedOfaCount;
} nvmlComputeInstanceProfileInfo_t;

typedef struct {
    unsigned version, id, sliceCount, instanceCount, multiprocessorCount, sharedCopyEngineCount,
        sharedDecoderCount, sharedEncoderCount, sharedJpegCount, sharedOfaCount;
    char name[96];
} nvmlComputeInstanceProfileInfo_v2_t;

typedef struct {
    unsigned version, id, sliceCount, instanceCount, multiprocessorCount, sharedCopyEngineCount,
        sharedDecoderCount, sharedEncoderCount, sharedJpegCount, sharedOfaCount;
    char name[96];
    unsigned capabilities;
} nvmlComputeInstanceProfileInfo_v3_t;

typedef struct {
    nvmlDevice_t device;
    nvmlGpuInstance_t gpuInstance;
    unsigned id, profileId;
    nvmlComputeInstancePlacement_t placement;
} nvmlComputeInstanceInfo_t;

typedef struct {
    unsigned multiprocessorCount, sharedCopyEngineCount, sharedDecoderCount, sharedEncoderCount,
        sharedJpegCount, sharedOfaCount, gpuInstanceSliceCount, computeInstanceSliceCount;
    unsigned long long memorySizeMB;
} nvmlDeviceAttributes_t;

#define STRUCT_VERSION(v) ((v) >> 24)

/* The visible index of a whole-GPU handle, or -1. */
static int vis_of(nvmlDevice_t d) { return d >= devs && d < devs + fg_count() ? (int)(d - devs) : -1; }

static nvmlGpuInstance_t gi_handle(int vis, int gi)
{
    struct mig_handle *h = &gi_handles[vis][gi];
    *h = (struct mig_handle){GI_MAGIC, vis, gi, -1};
    return h;
}

static nvmlComputeInstance_t ci_handle(int vis, int gi, int ci)
{
    struct mig_handle *h = &ci_handles[vis][gi][ci];
    *h = (struct mig_handle){CI_MAGIC, vis, gi, ci};
    return h;
}

static nvmlDevice_t mig_dev_handle(int vis, int gi, int ci)
{
    struct mig_handle *h = &mig_dev_handles[vis][gi][ci];
    *h = (struct mig_handle){MIGDEV_MAGIC, vis, gi, ci};
    return (nvmlDevice_t)h;
}

/* A GPU instance handle whose instance exists; its profile in *p. */
static struct mig_handle *gi_of(nvmlGpuInstance_t g, const struct mig_profile **p)
{
    if (!g || g < &gi_handles[0][0] || g >= &gi_handles[0][0] + FG_MAX_GPUS * FG_MIG_SLOTS) return NULL;
    if (g->magic != GI_MAGIC || !mig_gi_exists(fg_phys(g->vis), g->gi, p)) return NULL;
    return g;
}

static struct mig_handle *ci_of(nvmlComputeInstance_t c, const struct mig_profile **p,
                                const struct mig_ci_profile **cp)
{
    if (!c || c < &ci_handles[0][0][0] || c >= &ci_handles[0][0][0] + FG_MAX_GPUS * FG_MIG_SLOTS * FG_MIG_CIS)
        return NULL;
    if (c->magic != CI_MAGIC || !mig_ci_exists(fg_phys(c->vis), c->gi, c->ci, p, cp)) return NULL;
    return c;
}

static void ci_name_of(int phys, int gi, int ci, char *buf, size_t len)
{
    const struct mig_profile *p;
    const struct mig_ci_profile *c;
    if (mig_ci_exists(phys, gi, ci, &p, &c)) mig_ci_name(p, c, buf, len);
    else snprintf(buf, len, "?");
}

static nvmlReturn_t mig_dev_uuid(struct mig_handle *h, char *uuid, unsigned len)
{
    char u[64];
    mig_uuid_str(fg_phys(h->vis), h->gi, h->ci, u, sizeof u);
    return copy_str(uuid, len, u);
}

static nvmlReturn_t mig_dev_name(struct mig_handle *h, char *name, unsigned len)
{
    char ci[48], n[128];
    ci_name_of(fg_phys(h->vis), h->gi, h->ci, ci, sizeof ci);
    snprintf(n, sizeof n, "%s MIG %s", fg_name(), ci);
    return copy_str(name, len, n);
}

static unsigned long long mig_dev_mem(struct mig_handle *h)
{
    const struct mig_profile *p;
    return mig_gi_exists(fg_phys(h->vis), h->gi, &p) ? p->mib << 20 : 0;
}

static int mig_dev_index(struct mig_handle *h)
{
    struct mig_dev list[FG_MIG_SLOTS * FG_MIG_CIS];
    int n = mig_devices(fg_phys(h->vis), list);
    for (int i = 0; i < n; i++)
        if (list[i].gi == h->gi && list[i].ci == h->ci) return i;
    return -1;
}

static int mig_find_uuid(const char *uuid, nvmlDevice_t *d)
{
    char u[64];
    struct mig_dev list[FG_MIG_SLOTS * FG_MIG_CIS];
    for (int v = 0; v < fg_count(); v++) {
        int n = mig_devices(fg_phys(v), list);
        for (int i = 0; i < n; i++) {
            mig_uuid_str(fg_phys(v), list[i].gi, list[i].ci, u, sizeof u);
            if (!strcasecmp(u, uuid)) {
                *d = mig_dev_handle(v, list[i].gi, list[i].ci);
                return 1;
            }
        }
    }
    return 0;
}

/* ---- MIG mode ---- */

API nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t d, unsigned *cur, unsigned *pend)
{
    CHECK_DEV(d);
    if (!cur || !pend) return NVML_ERROR_INVALID_ARGUMENT;
    struct fg_gpu_mig *m = mig_state(idx);
    *cur = m ? __atomic_load_n(&m->current, __ATOMIC_ACQUIRE) : 0;
    *pend = m ? __atomic_load_n(&m->pending, __ATOMIC_ACQUIRE) : 0;
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceSetMigMode(nvmlDevice_t d, unsigned mode, nvmlReturn_t *activationStatus)
{
    CHECK_DEV(d);
    if (!activationStatus) return NVML_ERROR_INVALID_ARGUMENT;
    int act = MIG_OK;
    nvmlReturn_t rc = mig_set_mode(idx, mode, &act);
    *activationStatus = rc == NVML_SUCCESS ? act : rc;
    return rc;
}

API nvmlReturn_t nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t d, unsigned *n)
{
    if (mig_dev_of(d)) return NVML_ERROR_NOT_SUPPORTED;
    CHECK_DEV(d);
    if (!n) return NVML_ERROR_INVALID_ARGUMENT;
    *n = mig_enabled(idx) ? MIG_COMPUTE_SLICES : 0;
    return NVML_SUCCESS;
}

/* ---- GPU instance profiles and placements ---- */

static void profile_fill(const struct mig_profile *p, unsigned *id, unsigned *slices, unsigned *inst, unsigned *sms,
                         unsigned *ce, unsigned *dec, unsigned *enc, unsigned *jpeg, unsigned *ofa,
                         unsigned long long *mib)
{
    *id = p->id;
    *slices = p->slices;
    *inst = p->instances;
    *sms = p->sms;
    *ce = p->ce;
    *dec = p->dec;
    *enc = p->enc;
    *jpeg = p->jpeg;
    *ofa = p->ofa;
    *mib = p->mib;
}

/* Profiles exist whether MIG is on or off; an index GB200 lacks is not
 * supported, one beyond the enum invalid (callers enumerate until it). */
static nvmlReturn_t profile_lookup(nvmlDevice_t d, unsigned index, const struct mig_profile **p)
{
    if (dev_idx(d) < 0) return NVML_ERROR_INVALID_ARGUMENT;
    if (index >= MIG_PROFILE_COUNT) return NVML_ERROR_INVALID_ARGUMENT;
    *p = mig_profile_by_index(index);
    return *p ? NVML_SUCCESS : NVML_ERROR_NOT_SUPPORTED;
}

API nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfo(nvmlDevice_t d, unsigned index, nvmlGpuInstanceProfileInfo_t *i)
{
    const struct mig_profile *p;
    nvmlReturn_t rc = profile_lookup(d, index, &p);
    if (rc != NVML_SUCCESS) return rc;
    if (!i) return NVML_ERROR_INVALID_ARGUMENT;
    i->isP2pSupported = 0;
    profile_fill(p, &i->id, &i->sliceCount, &i->instanceCount, &i->multiprocessorCount, &i->copyEngineCount,
                 &i->decoderCount, &i->encoderCount, &i->jpegCount, &i->ofaCount, &i->memorySizeMB);
    return NVML_SUCCESS;
}

static nvmlReturn_t profile_info_v(const struct mig_profile *p, void *info)
{
    unsigned v = STRUCT_VERSION(*(unsigned *)info);
    if (v == 2) {
        nvmlGpuInstanceProfileInfo_v2_t *i = info;
        i->isP2pSupported = 0;
        profile_fill(p, &i->id, &i->sliceCount, &i->instanceCount, &i->multiprocessorCount, &i->copyEngineCount,
                     &i->decoderCount, &i->encoderCount, &i->jpegCount, &i->ofaCount, &i->memorySizeMB);
        snprintf(i->name, sizeof i->name, "MIG %s", p->name);
        return NVML_SUCCESS;
    }
    if (v == 3) {
        nvmlGpuInstanceProfileInfo_v3_t *i = info;
        profile_fill(p, &i->id, &i->sliceCount, &i->instanceCount, &i->multiprocessorCount, &i->copyEngineCount,
                     &i->decoderCount, &i->encoderCount, &i->jpegCount, &i->ofaCount, &i->memorySizeMB);
        snprintf(i->name, sizeof i->name, "MIG %s", p->name);
        i->capabilities = 0; /* no P2P, no graphics */
        return NVML_SUCCESS;
    }
    return 25; /* NVML_ERROR_ARGUMENT_VERSION_MISMATCH */
}

API nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfoV(nvmlDevice_t d, unsigned index, void *info)
{
    const struct mig_profile *p;
    nvmlReturn_t rc = profile_lookup(d, index, &p);
    if (rc != NVML_SUCCESS) return rc;
    return info ? profile_info_v(p, info) : NVML_ERROR_INVALID_ARGUMENT;
}

API nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfoByIdV(nvmlDevice_t d, unsigned id, void *info)
{
    if (dev_idx(d) < 0 || !info) return NVML_ERROR_INVALID_ARGUMENT;
    const struct mig_profile *p = mig_profile_by_id(id);
    return p ? profile_info_v(p, info) : NVML_ERROR_INVALID_ARGUMENT;
}

/* Placements of a profile id, whether MIG is on or off; NULL placements:
 * the count only (go-nvml's first call). */
API nvmlReturn_t nvmlDeviceGetGpuInstancePossiblePlacements_v2(nvmlDevice_t d, unsigned id,
                                                               nvmlGpuInstancePlacement_t *pl, unsigned *n)
{
    if (dev_idx(d) < 0 || !n) return NVML_ERROR_INVALID_ARGUMENT;
    const struct mig_profile *p = mig_profile_by_id(id);
    if (!p) return NVML_ERROR_INVALID_ARGUMENT;
    if (pl) {
        if (*n < p->nstarts) return NVML_ERROR_INSUFFICIENT_SIZE;
        for (unsigned i = 0; i < p->nstarts; i++) pl[i] = (nvmlGpuInstancePlacement_t){p->starts[i], p->mem_slices};
    }
    *n = p->nstarts;
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetGpuInstancePossiblePlacements(nvmlDevice_t d, unsigned id,
                                                            nvmlGpuInstancePlacement_t *pl, unsigned *n)
{
    return nvmlDeviceGetGpuInstancePossiblePlacements_v2(d, id, pl, n);
}

/* ---- GPU instances ---- */

/* MIG must be on for instances; *p the profile of a known id. */
static nvmlReturn_t gi_call(nvmlDevice_t d, unsigned id, int *idx, const struct mig_profile **p)
{
    *idx = dev_idx(d);
    if (*idx < 0) return NVML_ERROR_INVALID_ARGUMENT;
    if (!mig_enabled(*idx)) return NVML_ERROR_NOT_SUPPORTED;
    if (p && !(*p = mig_profile_by_id(id))) return NVML_ERROR_INVALID_ARGUMENT;
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetGpuInstanceRemainingCapacity(nvmlDevice_t d, unsigned id, unsigned *n)
{
    int idx;
    const struct mig_profile *p;
    nvmlReturn_t rc = gi_call(d, id, &idx, &p);
    if (rc != NVML_SUCCESS) return rc;
    if (!n) return NVML_ERROR_INVALID_ARGUMENT;
    struct fg_gpu_mig m = *mig_state(idx); /* a copy to fill */
    unsigned count = 0;
    int s;
    while ((s = mig_gi_fit(&m, p, -1)) >= 0) {
        struct fg_mig_gi *x = &m.gi[mig_slot((unsigned)s, p->mem_slices)];
        *x = (struct fg_mig_gi){.used = 1, .profile = (uint8_t)(p - mig_profiles), .start = (uint8_t)s,
                                .size = (uint8_t)p->mem_slices};
        count++;
    }
    *n = count;
    return NVML_SUCCESS;
}

static nvmlReturn_t create_gi(nvmlDevice_t d, unsigned id, int start, nvmlGpuInstance_t *g)
{
    int idx;
    const struct mig_profile *p;
    nvmlReturn_t rc = gi_call(d, id, &idx, &p);
    if (rc != NVML_SUCCESS) return rc;
    if (!g) return NVML_ERROR_INVALID_ARGUMENT;
    int slot;
    rc = mig_create_gi(idx, p, start, &slot);
    if (rc == NVML_SUCCESS) *g = gi_handle(vis_of(d), slot);
    return rc;
}

API nvmlReturn_t nvmlDeviceCreateGpuInstance(nvmlDevice_t d, unsigned id, nvmlGpuInstance_t *g)
{
    return create_gi(d, id, -1, g);
}

API nvmlReturn_t nvmlDeviceCreateGpuInstanceWithPlacement(nvmlDevice_t d, unsigned id,
                                                          const nvmlGpuInstancePlacement_t *pl, nvmlGpuInstance_t *g)
{
    const struct mig_profile *p = mig_profile_by_id(id);
    if (!pl || !p) return NVML_ERROR_INVALID_ARGUMENT;
    if (!mig_valid_start(p, pl->start, pl->size)) return NVML_ERROR_INVALID_ARGUMENT;
    return create_gi(d, id, (int)pl->start, g);
}

API nvmlReturn_t nvmlGpuInstanceDestroy(nvmlGpuInstance_t g)
{
    struct mig_handle *h = gi_of(g, NULL);
    if (!h) return NVML_ERROR_INVALID_ARGUMENT;
    return mig_destroy_gi(fg_phys(h->vis), h->gi);
}

API nvmlReturn_t nvmlDeviceGetGpuInstances(nvmlDevice_t d, unsigned id, nvmlGpuInstance_t *gs, unsigned *n)
{
    int idx;
    const struct mig_profile *p;
    nvmlReturn_t rc = gi_call(d, id, &idx, &p);
    if (rc != NVML_SUCCESS) return rc;
    if (!n) return NVML_ERROR_INVALID_ARGUMENT;
    struct fg_gpu_mig *m = mig_state(idx);
    unsigned count = 0;
    for (int g = 0; g < FG_MIG_SLOTS; g++) {
        if (!m->gi[g].used || &mig_profiles[m->gi[g].profile] != p) continue;
        if (gs) {
            if (count >= *n) return NVML_ERROR_INSUFFICIENT_SIZE;
            gs[count] = gi_handle(vis_of(d), g);
        }
        count++;
    }
    *n = count;
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetGpuInstanceById(nvmlDevice_t d, unsigned id, nvmlGpuInstance_t *g)
{
    int idx;
    nvmlReturn_t rc = gi_call(d, 0, &idx, NULL);
    if (rc != NVML_SUCCESS) return rc;
    if (!g) return NVML_ERROR_INVALID_ARGUMENT;
    if (id >= FG_MIG_SLOTS || !mig_gi_exists(idx, (int)id, NULL)) return NVML_ERROR_NOT_FOUND;
    *g = gi_handle(vis_of(d), (int)id);
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlGpuInstanceGetInfo(nvmlGpuInstance_t g, nvmlGpuInstanceInfo_t *info)
{
    const struct mig_profile *p;
    struct mig_handle *h = gi_of(g, &p);
    if (!h || !info) return NVML_ERROR_INVALID_ARGUMENT;
    const struct fg_mig_gi *x = &mig_state(fg_phys(h->vis))->gi[h->gi];
    info->device = &devs[h->vis];
    info->id = (unsigned)h->gi;
    info->profileId = p->id;
    info->placement = (nvmlGpuInstancePlacement_t){x->start, x->size};
    return NVML_SUCCESS;
}

/* ---- compute instances ---- */

static nvmlReturn_t ci_profile_lookup(nvmlGpuInstance_t g, unsigned index, unsigned engine,
                                      struct mig_handle **h, const struct mig_profile **p,
                                      const struct mig_ci_profile **c)
{
    *h = gi_of(g, p);
    if (!*h) return NVML_ERROR_INVALID_ARGUMENT;
    if (index >= MIG_CI_PROFILE_COUNT || engine != 0) return NVML_ERROR_INVALID_ARGUMENT;
    *c = mig_ci_profile(*p, index);
    return *c ? NVML_SUCCESS : NVML_ERROR_NOT_SUPPORTED;
}

API nvmlReturn_t nvmlGpuInstanceGetComputeInstanceProfileInfo(nvmlGpuInstance_t g, unsigned index, unsigned engine,
                                                              nvmlComputeInstanceProfileInfo_t *i)
{
    struct mig_handle *h;
    const struct mig_profile *p;
    const struct mig_ci_profile *c;
    nvmlReturn_t rc = ci_profile_lookup(g, index, engine, &h, &p, &c);
    if (rc != NVML_SUCCESS) return rc;
    if (!i) return NVML_ERROR_INVALID_ARGUMENT;
    *i = (nvmlComputeInstanceProfileInfo_t){(unsigned)c->index, c->slices, c->instances, c->sms, p->ce, p->dec,
                                            p->enc, p->jpeg, p->ofa};
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlGpuInstanceGetComputeInstanceProfileInfoV(nvmlGpuInstance_t g, unsigned index, unsigned engine,
                                                               void *info)
{
    struct mig_handle *h;
    const struct mig_profile *p;
    const struct mig_ci_profile *c;
    nvmlReturn_t rc = ci_profile_lookup(g, index, engine, &h, &p, &c);
    if (rc != NVML_SUCCESS) return rc;
    if (!info) return NVML_ERROR_INVALID_ARGUMENT;
    unsigned v = STRUCT_VERSION(*(unsigned *)info);
    if (v != 2 && v != 3) return 25; /* NVML_ERROR_ARGUMENT_VERSION_MISMATCH */
    nvmlComputeInstanceProfileInfo_v3_t *i = info; /* v2 is v3 without capabilities */
    i->id = (unsigned)c->index;
    i->sliceCount = c->slices;
    i->instanceCount = c->instances;
    i->multiprocessorCount = c->sms;
    i->sharedCopyEngineCount = p->ce;
    i->sharedDecoderCount = p->dec;
    i->sharedEncoderCount = p->enc;
    i->sharedJpegCount = p->jpeg;
    i->sharedOfaCount = p->ofa;
    char name[48];
    mig_ci_name(p, c, name, sizeof name);
    snprintf(i->name, sizeof i->name, "MIG %s", name);
    if (v == 3) i->capabilities = 0;
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlGpuInstanceGetComputeInstancePossiblePlacements(nvmlGpuInstance_t g, unsigned id,
                                                                     nvmlComputeInstancePlacement_t *pl, unsigned *n)
{
    struct mig_handle *h;
    const struct mig_profile *p;
    const struct mig_ci_profile *c;
    nvmlReturn_t rc = ci_profile_lookup(g, id, 0, &h, &p, &c);
    if (rc != NVML_SUCCESS) return rc;
    if (!n) return NVML_ERROR_INVALID_ARGUMENT;
    if (pl) {
        if (*n < c->instances) return NVML_ERROR_INSUFFICIENT_SIZE;
        for (unsigned k = 0; k < c->instances; k++) pl[k] = (nvmlComputeInstancePlacement_t){k * c->slices, c->slices};
    }
    *n = c->instances;
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlGpuInstanceGetComputeInstanceRemainingCapacity(nvmlGpuInstance_t g, unsigned id, unsigned *n)
{
    struct mig_handle *h;
    const struct mig_profile *p;
    const struct mig_ci_profile *c;
    nvmlReturn_t rc = ci_profile_lookup(g, id, 0, &h, &p, &c);
    if (rc != NVML_SUCCESS) return rc;
    if (!n) return NVML_ERROR_INVALID_ARGUMENT;
    struct fg_mig_gi x = mig_state(fg_phys(h->vis))->gi[h->gi];
    unsigned count = 0;
    int s;
    while ((s = mig_ci_fit(&x, c)) >= 0) {
        for (int i = 0; i < FG_MIG_CIS; i++)
            if (!x.ci[i].used) {
                x.ci[i] = (struct fg_mig_ci){1, (uint8_t)c->index, (uint8_t)s, (uint8_t)c->slices};
                break;
            }
        count++;
    }
    *n = count;
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlGpuInstanceCreateComputeInstance(nvmlGpuInstance_t g, unsigned id, nvmlComputeInstance_t *ci)
{
    struct mig_handle *h = gi_of(g, NULL);
    if (!h || !ci) return NVML_ERROR_INVALID_ARGUMENT;
    if (id >= MIG_CI_PROFILE_COUNT) return NVML_ERROR_INVALID_ARGUMENT;
    int c;
    nvmlReturn_t rc = mig_create_ci(fg_phys(h->vis), h->gi, id, &c);
    if (rc == NVML_SUCCESS) *ci = ci_handle(h->vis, h->gi, c);
    return rc;
}

API nvmlReturn_t nvmlGpuInstanceCreateComputeInstanceWithPlacement(nvmlGpuInstance_t g, unsigned id,
                                                                   const nvmlComputeInstancePlacement_t *pl,
                                                                   nvmlComputeInstance_t *ci)
{
    (void)pl; /* the lowest free placement: compute instances do not move memory */
    return nvmlGpuInstanceCreateComputeInstance(g, id, ci);
}

API nvmlReturn_t nvmlGpuInstanceGetComputeInstances(nvmlGpuInstance_t g, unsigned id, nvmlComputeInstance_t *cs,
                                                    unsigned *n)
{
    struct mig_handle *h = gi_of(g, NULL);
    if (!h || !n) return NVML_ERROR_INVALID_ARGUMENT;
    const struct fg_mig_gi *x = &mig_state(fg_phys(h->vis))->gi[h->gi];
    unsigned count = 0;
    for (int c = 0; c < FG_MIG_CIS; c++) {
        if (!x->ci[c].used || x->ci[c].profile != id) continue;
        if (cs) {
            if (count >= *n) return NVML_ERROR_INSUFFICIENT_SIZE;
            cs[count] = ci_handle(h->vis, h->gi, c);
        }
        count++;
    }
    *n = count;
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlGpuInstanceGetComputeInstanceById(nvmlGpuInstance_t g, unsigned id, nvmlComputeInstance_t *ci)
{
    struct mig_handle *h = gi_of(g, NULL);
    if (!h || !ci) return NVML_ERROR_INVALID_ARGUMENT;
    if (id >= FG_MIG_CIS || !mig_ci_exists(fg_phys(h->vis), h->gi, (int)id, NULL, NULL)) return NVML_ERROR_NOT_FOUND;
    *ci = ci_handle(h->vis, h->gi, (int)id);
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlComputeInstanceGetInfo_v2(nvmlComputeInstance_t c, nvmlComputeInstanceInfo_t *info)
{
    const struct mig_ci_profile *cp;
    struct mig_handle *h = ci_of(c, NULL, &cp);
    if (!h || !info) return NVML_ERROR_INVALID_ARGUMENT;
    const struct fg_mig_ci *y = &mig_state(fg_phys(h->vis))->gi[h->gi].ci[h->ci];
    info->device = &devs[h->vis];
    info->gpuInstance = gi_handle(h->vis, h->gi);
    info->id = (unsigned)h->ci;
    info->profileId = (unsigned)cp->index;
    info->placement = (nvmlComputeInstancePlacement_t){y->start, y->size};
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlComputeInstanceGetInfo(nvmlComputeInstance_t c, nvmlComputeInstanceInfo_t *info)
{
    return nvmlComputeInstanceGetInfo_v2(c, info);
}

API nvmlReturn_t nvmlComputeInstanceDestroy(nvmlComputeInstance_t c)
{
    struct mig_handle *h = ci_of(c, NULL, NULL);
    if (!h) return NVML_ERROR_INVALID_ARGUMENT;
    return mig_destroy_ci(fg_phys(h->vis), h->gi, h->ci);
}

/* ---- MIG devices ---- */

API nvmlReturn_t nvmlDeviceGetMigDeviceHandleByIndex(nvmlDevice_t d, unsigned index, nvmlDevice_t *mig)
{
    CHECK_DEV(d);
    if (!mig) return NVML_ERROR_INVALID_ARGUMENT;
    struct mig_dev list[FG_MIG_SLOTS * FG_MIG_CIS];
    int n = mig_devices(idx, list);
    if (index >= (unsigned)n) return NVML_ERROR_NOT_FOUND;
    *mig = mig_dev_handle(vis_of(d), list[index].gi, list[index].ci);
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceIsMigDeviceHandle(nvmlDevice_t d, unsigned *is)
{
    if (!is) return NVML_ERROR_INVALID_ARGUMENT;
    if (mig_dev_of(d)) {
        *is = 1;
        return NVML_SUCCESS;
    }
    CHECK_DEV(d);
    *is = 0;
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetDeviceHandleFromMigDeviceHandle(nvmlDevice_t mig, nvmlDevice_t *d)
{
    struct mig_handle *h = mig_dev_of(mig);
    if (!h || !d) return NVML_ERROR_INVALID_ARGUMENT;
    *d = &devs[h->vis];
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetGpuInstanceId(nvmlDevice_t d, unsigned *id)
{
    struct mig_handle *h = mig_dev_of(d);
    if (!id) return NVML_ERROR_INVALID_ARGUMENT;
    if (!h) return dev_idx(d) < 0 ? NVML_ERROR_INVALID_ARGUMENT : NVML_ERROR_NOT_SUPPORTED;
    *id = (unsigned)h->gi;
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetComputeInstanceId(nvmlDevice_t d, unsigned *id)
{
    struct mig_handle *h = mig_dev_of(d);
    if (!id) return NVML_ERROR_INVALID_ARGUMENT;
    if (!h) return dev_idx(d) < 0 ? NVML_ERROR_INVALID_ARGUMENT : NVML_ERROR_NOT_SUPPORTED;
    *id = (unsigned)h->ci;
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetAttributes_v2(nvmlDevice_t d, nvmlDeviceAttributes_t *a)
{
    const struct mig_profile *p;
    const struct mig_ci_profile *c;
    struct mig_handle *h = mig_dev_of(d);
    if (!a) return NVML_ERROR_INVALID_ARGUMENT;
    if (!h) return dev_idx(d) < 0 ? NVML_ERROR_INVALID_ARGUMENT : NVML_ERROR_NOT_SUPPORTED;
    if (!mig_ci_exists(fg_phys(h->vis), h->gi, h->ci, &p, &c)) return NVML_ERROR_INVALID_ARGUMENT;
    *a = (nvmlDeviceAttributes_t){c->sms, p->ce, p->dec, p->enc, p->jpeg, p->ofa, p->slices, c->slices, p->mib};
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetAttributes(nvmlDevice_t d, nvmlDeviceAttributes_t *a) { return nvmlDeviceGetAttributes_v2(d, a); }
