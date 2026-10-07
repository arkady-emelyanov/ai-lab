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
#include "occupancy.h"

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
    CHECK_DEV(d);
    if (!i) return NVML_ERROR_INVALID_ARGUMENT;
    *i = (unsigned)(d - devs);
    return NVML_SUCCESS;
}

/* The /dev/nvidiaN number (Slurm builds the device path from it): physical. */
API nvmlReturn_t nvmlDeviceGetMinorNumber(nvmlDevice_t d, unsigned *m)
{
    CHECK_DEV(d);
    if (!m) return NVML_ERROR_INVALID_ARGUMENT;
    *m = (unsigned)idx;
    return NVML_SUCCESS;
}

/* ---- identity ---------------------------------------------------------- */

API nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t d, char *name, unsigned len)
{
    CHECK_DEV(d);
    return copy_str(name, len, fg_name());
}

API nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t d, char *uuid, unsigned len)
{
    char u[64];
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

static void fill_pci(nvmlPciInfo_t *p, unsigned bus, unsigned devid)
{
    memset(p, 0, sizeof *p);
    p->bus = bus;
    p->pciDeviceId = devid;
    p->pciSubSystemId = 0x20E610DE;
    snprintf(p->busIdLegacy, sizeof p->busIdLegacy, "%04X:%02X:00.0", 0, bus);
    snprintf(p->busId, sizeof p->busId, "%08X:%02X:00.0", 0, bus);
}

API nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t d, nvmlPciInfo_t *p)
{
    CHECK_DEV(d);
    if (!p) return NVML_ERROR_INVALID_ARGUMENT;
    fill_pci(p, fg_pci_bus[idx], FG_PCI_DEVICE_ID);
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
    CHECK_DEV(d);
    if (!m) return NVML_ERROR_INVALID_ARGUMENT;
    m->total = fg_mem_bytes();
    m->used = RESERVED_BYTES + mem_used(idx);
    m->free = m->used < m->total ? m->total - m->used : 0;
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t d, nvmlMemory_v2_t *m)
{
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
API nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t d, unsigned *cur, unsigned *pend)
{
    CHECK_DEV(d);
    if (!cur || !pend) return NVML_ERROR_INVALID_ARGUMENT;
    *cur = *pend = 0;
    return NVML_SUCCESS;
}

API nvmlReturn_t nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t d, unsigned *n) { return set_uint(d, n, 0); }

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
    fill_pci(p, NVSWITCH_PCI_BUS, NVSWITCH_PCI_DEVICE_ID);
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
    return 0;
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
