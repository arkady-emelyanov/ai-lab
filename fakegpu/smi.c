/*
 * nvidia-smi look-alike backed by NVML (the fakegpu stub on emulated nodes).
 * Like NVIDIA's, a dynamically linked program that loads libnvidia-ml.so.1
 * at run time, so it runs wherever the driver's libraries do: on the tray,
 * in pods and in containers (no interpreter needed).
 *
 * Supported: default table, -L, topo -m, topo -gnid, nvlink -s, -q [-d ...],
 * dmon [-c N], --query-gpu=FIELDS / --query-compute-apps=FIELDS
 * --format=csv[,noheader][,nounits], -i IDS (indices or PCI bus ids),
 * -l SECONDS, -r / --gpu-reset [-i IDS] (root, GPUs without processes),
 * --version, MIG: -mig 0|1, mig -lgip | -lgipp | -lcip [-gi IDS] | -lgi |
 * -lci [-gi IDS] | -cgi PROFILES [-C] | -cci [PROFILES] [-gi IDS] |
 * -dci [-gi IDS] [-ci IDS] | -dgi [-gi IDS] (changes as root, as NVML
 * requires), MIG devices in the default table.
 */
#include <ctype.h>
#include <dlfcn.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* ---- NVML's types (nvml.h ABI), as the fake NVML implements them ---------- */

typedef void *dev_t_;
typedef struct {
    char busIdLegacy[16];
    unsigned domain, bus, device, pciDeviceId, pciSubSystemId;
    char busId[32];
} pci_t;
typedef struct { unsigned long long total, free, used; } mem_t;
typedef struct { unsigned gpu, memory; } util_t;
typedef struct {
    unsigned fieldId, scopeId;
    long long timestamp, latencyUsec;
    int valueType, nvmlReturn;
    unsigned long long value;
} field_t;
typedef struct { unsigned pid; unsigned long long usedGpuMemory; unsigned gpuInstanceId, computeInstanceId; } proc_t;
typedef struct {
    unsigned version;
    unsigned char clusterUuid[16];
    int status;
    unsigned cliqueId;
    unsigned char state;
    unsigned healthMask;
    unsigned char healthSummary;
} fabric_t;
typedef struct {
    unsigned version, id, isP2pSupported, sliceCount, instanceCount, multiprocessorCount, copyEngineCount,
        decoderCount, encoderCount, jpegCount, ofaCount;
    unsigned long long memorySizeMB;
    char name[96];
} giprof_t;
typedef struct { unsigned start, size; } place_t;
typedef struct { void *device; unsigned id, profileId; place_t placement; } giinfo_t;
typedef struct {
    unsigned version, id, sliceCount, instanceCount, multiprocessorCount, sharedCopyEngineCount,
        sharedDecoderCount, sharedEncoderCount, sharedJpegCount, sharedOfaCount;
    char name[96];
} ciprof_t;
typedef struct { void *device, *gpuInstance; unsigned id, profileId; place_t placement; } ciinfo_t;
typedef struct {
    unsigned multiprocessorCount, sharedCopyEngineCount, sharedDecoderCount, sharedEncoderCount,
        sharedJpegCount, sharedOfaCount, gpuInstanceSliceCount, computeInstanceSliceCount;
    unsigned long long memorySizeMB;
} attrs_t;

#define FI_NVLINK_SPEED_MBPS 90   /* NVML_FI_DEV_NVLINK_SPEED_MBPS_COMMON */
#define FI_GPU_RECOVERY_ACTION 230 /* NVML_FI_DEV_GET_GPU_RECOVERY_ACTION */
#define NVLINK_MAX_LINKS 64        /* links are enumerated until NVML rejects the index */
#define MIG_PROFILE_COUNT 18
#define MIG_CI_PROFILE_COUNT 9
#define IN_USE 19
#define NOT_FOUND 6
#define NONE_ID 0xFFFFFFFFu
#define MAX_GPUS 64
#define MAX_PROCS 64
#define MAX_MIG 64

/* ---- NVML, loaded at run time -------------------------------------------------- */

#define NVML_FUNCS(X)                                                                          \
    X(int, nvmlInit_v2, (void))                                                                \
    X(const char *, nvmlErrorString, (int))                                                    \
    X(int, nvmlSystemGetDriverVersion, (char *, unsigned))                                     \
    X(int, nvmlSystemGetNVMLVersion, (char *, unsigned))                                       \
    X(int, nvmlSystemGetCudaDriverVersion_v2, (int *))                                         \
    X(int, nvmlSystemGetProcessName, (unsigned, char *, unsigned))                             \
    X(int, nvmlDeviceGetCount_v2, (unsigned *))                                                \
    X(int, nvmlDeviceGetHandleByIndex_v2, (unsigned, dev_t_ *))                                \
    X(int, nvmlDeviceGetName, (dev_t_, char *, unsigned))                                      \
    X(int, nvmlDeviceGetUUID, (dev_t_, char *, unsigned))                                      \
    X(int, nvmlDeviceGetSerial, (dev_t_, char *, unsigned))                                    \
    X(int, nvmlDeviceGetVbiosVersion, (dev_t_, char *, unsigned))                              \
    X(int, nvmlDeviceGetPciInfo_v3, (dev_t_, pci_t *))                                         \
    X(int, nvmlDeviceGetMemoryInfo, (dev_t_, mem_t *))                                         \
    X(int, nvmlDeviceGetUtilizationRates, (dev_t_, util_t *))                                  \
    X(int, nvmlDeviceGetTemperature, (dev_t_, int, unsigned *))                                \
    X(int, nvmlDeviceGetPowerUsage, (dev_t_, unsigned *))                                      \
    X(int, nvmlDeviceGetPowerManagementLimit, (dev_t_, unsigned *))                            \
    X(int, nvmlDeviceGetPerformanceState, (dev_t_, unsigned *))                                \
    X(int, nvmlDeviceGetClockInfo, (dev_t_, int, unsigned *))                                  \
    X(int, nvmlDeviceGetGpuFabricInfoV, (dev_t_, fabric_t *))                                  \
    X(int, nvmlDeviceGetNvLinkState, (dev_t_, unsigned, unsigned *))                           \
    X(int, nvmlDeviceGetMinorNumber, (dev_t_, unsigned *))                                     \
    X(int, nvmlDeviceGetNumaNodeId, (dev_t_, unsigned *))                                      \
    X(int, nvmlDeviceGetMemoryAffinity, (dev_t_, unsigned, unsigned long *, int))              \
    X(int, nvmlDeviceGetCpuAffinity, (dev_t_, unsigned, unsigned long *))                      \
    X(int, nvmlDeviceGetFieldValues, (dev_t_, int, field_t *))                                 \
    X(int, nvmlDeviceGetComputeRunningProcesses_v3, (dev_t_, unsigned *, proc_t *))            \
    X(int, nvmlDeviceGetMigMode, (dev_t_, unsigned *, unsigned *))                             \
    X(int, nvmlDeviceSetMigMode, (dev_t_, unsigned, int *))                                    \
    X(int, nvmlDeviceGetMaxMigDeviceCount, (dev_t_, unsigned *))                               \
    X(int, nvmlDeviceGetMigDeviceHandleByIndex, (dev_t_, unsigned, dev_t_ *))                  \
    X(int, nvmlDeviceGetGpuInstanceId, (dev_t_, unsigned *))                                   \
    X(int, nvmlDeviceGetComputeInstanceId, (dev_t_, unsigned *))                               \
    X(int, nvmlDeviceGetAttributes_v2, (dev_t_, attrs_t *))                                    \
    X(int, nvmlDeviceGetGpuInstanceProfileInfoV, (dev_t_, unsigned, giprof_t *))               \
    X(int, nvmlDeviceGetGpuInstanceRemainingCapacity, (dev_t_, unsigned, unsigned *))          \
    X(int, nvmlDeviceGetGpuInstancePossiblePlacements_v2, (dev_t_, unsigned, place_t *, unsigned *)) \
    X(int, nvmlDeviceGetGpuInstances, (dev_t_, unsigned, void **, unsigned *))                 \
    X(int, nvmlDeviceCreateGpuInstance, (dev_t_, unsigned, void **))                           \
    X(int, nvmlGpuInstanceGetInfo, (void *, giinfo_t *))                                       \
    X(int, nvmlGpuInstanceDestroy, (void *))                                                   \
    X(int, nvmlGpuInstanceGetComputeInstanceProfileInfoV, (void *, unsigned, unsigned, ciprof_t *)) \
    X(int, nvmlGpuInstanceGetComputeInstanceRemainingCapacity, (void *, unsigned, unsigned *)) \
    X(int, nvmlGpuInstanceGetComputeInstances, (void *, unsigned, void **, unsigned *))        \
    X(int, nvmlGpuInstanceCreateComputeInstance, (void *, unsigned, void **))                  \
    X(int, nvmlComputeInstanceGetInfo_v2, (void *, ciinfo_t *))                                \
    X(int, nvmlComputeInstanceDestroy, (void *))

#define X(ret, name, args) static ret(*name) args;
NVML_FUNCS(X)
#undef X
/* The fake driver's own (lab only): a GPU reset, the driver's load at boot. */
static int (*fakegpu_reset_gpu)(unsigned);
static void (*fakegpu_driver_load)(void);

static void fail(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

static void load_nvml(void)
{
    void *lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!lib) {
        puts("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. "
             "Make sure that the latest NVIDIA driver is installed and running.\n");
        exit(9);
    }
#define X(ret, name, args)                                   \
    *(void **)&name = dlsym(lib, #name);                     \
    if (!name)                                               \
        fail("nvidia-smi: %s missing from NVML", #name);
    NVML_FUNCS(X)
#undef X
    *(void **)&fakegpu_reset_gpu = dlsym(lib, "fakegpu_reset_gpu");
    *(void **)&fakegpu_driver_load = dlsym(lib, "fakegpu_driver_load");
}

static void check(int ret)
{
    if (ret != 0)
        fail("NVML error: %s", nvmlErrorString(ret));
}

/* ---- the GPUs ------------------------------------------------------------------- */

typedef struct { unsigned pid; char name[256]; unsigned long long mib; unsigned gi, ci; } procrow_t;
typedef struct { unsigned index, gi, ci; char uuid[96], name[96]; attrs_t a; mem_t mem; } migdev_t;

typedef struct {
    unsigned index;
    dev_t_ h;
    char name[96], uuid[96], serial[32], vbios[32], bus_id[32];
    unsigned pci_device_id;
    unsigned long long mem_total, mem_used;
    unsigned util, mem_util, temp, pstate, sm_clock, mem_clock, minor, clique;
    double power;
    unsigned power_limit;
    char cluster_uuid[40];
    const char *fabric_state, *recovery;
    unsigned links[NVLINK_MAX_LINKS], nlinks, links_up;
    char gpu_numa[16], numa_affinity[512], cpu_affinity[512];
    double nvlink_gbs;
    procrow_t procs[MAX_PROCS];
    unsigned nprocs;
    unsigned mig_current, mig_pending;
    migdev_t mig[MAX_MIG];
    unsigned nmig;
} gpu_t;

static const char *recovery_name(unsigned v)
{
    static const char *names[] = {"None", "GPU_RESET", "NODE_REBOOT", "DRAIN_P2P", "DRAIN_AND_RESET",
                                  "RECOVER_IMEX_DOMAIN", "BUS_RESET", "SYSTEM_REBOOT"};
    return v < sizeof(names) / sizeof(*names) ? names[v] : "Unknown";
}

/* [0, 1, 2, 5] -> "0-2,5", as nvidia-smi prints CPU affinity. */
static void cpu_ranges(const unsigned long *mask, int words, char *out, size_t len)
{
    size_t n = 0;
    out[0] = 0;
    int start = -1;
    for (int c = 0; c <= words * 64; c++) {
        int set = c < words * 64 && (mask[c / 64] >> (c % 64) & 1);
        if (set && start < 0)
            start = c;
        if (!set && start >= 0) {
            n += snprintf(out + n, n < len ? len - n : 0, "%s%d", n ? "," : "", start);
            if (c - 1 > start)
                n += snprintf(out + n, n < len ? len - n : 0, "-%d", c - 1);
            start = -1;
        }
    }
}

static unsigned field(dev_t_ h, unsigned id, int *ok)
{
    field_t fv = {.fieldId = id};
    check(nvmlDeviceGetFieldValues(h, 1, &fv));
    *ok = fv.nvmlReturn == 0;
    return (unsigned)(fv.value & 0xFFFFFFFFu);
}

static void read_mig_devices(gpu_t *g)
{
    unsigned n = 0;
    check(nvmlDeviceGetMaxMigDeviceCount(g->h, &n));
    for (unsigned i = 0; i < n && g->nmig < MAX_MIG; i++) {
        dev_t_ m;
        if (nvmlDeviceGetMigDeviceHandleByIndex(g->h, i, &m) != 0)
            break;
        migdev_t *d = &g->mig[g->nmig++];
        d->index = i;
        check(nvmlDeviceGetGpuInstanceId(m, &d->gi));
        check(nvmlDeviceGetComputeInstanceId(m, &d->ci));
        check(nvmlDeviceGetAttributes_v2(m, &d->a));
        char name[96];
        check(nvmlDeviceGetName(m, name, sizeof name));
        const char *p = strstr(name, " MIG ");
        snprintf(d->name, sizeof d->name, "%s", p ? p + 5 : name);
        check(nvmlDeviceGetMemoryInfo(m, &d->mem));
        check(nvmlDeviceGetUUID(m, d->uuid, sizeof d->uuid));
    }
}

static void read_gpu(gpu_t *g, unsigned index)
{
    memset(g, 0, sizeof *g);
    g->index = index;
    check(nvmlDeviceGetHandleByIndex_v2(index, &g->h));
    dev_t_ h = g->h;
    check(nvmlDeviceGetName(h, g->name, sizeof g->name));
    check(nvmlDeviceGetUUID(h, g->uuid, sizeof g->uuid));
    check(nvmlDeviceGetSerial(h, g->serial, sizeof g->serial));
    check(nvmlDeviceGetVbiosVersion(h, g->vbios, sizeof g->vbios));
    pci_t pci;
    check(nvmlDeviceGetPciInfo_v3(h, &pci));
    snprintf(g->bus_id, sizeof g->bus_id, "%s", pci.busId);
    g->pci_device_id = pci.pciDeviceId;
    mem_t mem;
    check(nvmlDeviceGetMemoryInfo(h, &mem));
    g->mem_total = mem.total >> 20;
    g->mem_used = mem.used >> 20;
    util_t util;
    check(nvmlDeviceGetUtilizationRates(h, &util));
    g->util = util.gpu;
    g->mem_util = util.memory;
    check(nvmlDeviceGetTemperature(h, 0, &g->temp));
    unsigned mw;
    check(nvmlDeviceGetPowerUsage(h, &mw));
    g->power = mw / 1000.0; /* W, milliwatt resolution */
    check(nvmlDeviceGetPowerManagementLimit(h, &mw));
    g->power_limit = mw / 1000;
    check(nvmlDeviceGetPerformanceState(h, &g->pstate));
    check(nvmlDeviceGetClockInfo(h, 1, &g->sm_clock));
    check(nvmlDeviceGetClockInfo(h, 2, &g->mem_clock));
    fabric_t fab = {.version = (unsigned)sizeof(fabric_t) | (3u << 24)};
    check(nvmlDeviceGetGpuFabricInfoV(h, &fab));
    const unsigned char *u = fab.clusterUuid;
    snprintf(g->cluster_uuid, sizeof g->cluster_uuid,
             "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", u[0], u[1], u[2], u[3], u[4],
             u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
    g->clique = fab.cliqueId;
    static const char *states[] = {"N/A", "Not Started", "In Progress", "Completed"};
    g->fabric_state = fab.state < 4 ? states[fab.state] : "Unknown";
    for (unsigned link = 0; link < NVLINK_MAX_LINKS; link++) {
        unsigned v;
        if (nvmlDeviceGetNvLinkState(h, link, &v) != 0)
            break;
        g->links[g->nlinks++] = v;
        g->links_up += v;
    }
    check(nvmlDeviceGetMinorNumber(h, &g->minor));
    unsigned node;
    if (nvmlDeviceGetNumaNodeId(h, &node) == 0)
        snprintf(g->gpu_numa, sizeof g->gpu_numa, "%u", node);
    else
        strcpy(g->gpu_numa, "N/A");
    unsigned long numa[4] = {0};
    check(nvmlDeviceGetMemoryAffinity(h, 4, numa, 0));
    size_t n = 0;
    for (int b = 0; b < 4 * 64; b++)
        if (numa[b / 64] >> (b % 64) & 1)
            n += snprintf(g->numa_affinity + n, n < sizeof g->numa_affinity ? sizeof g->numa_affinity - n : 0,
                          "%s%d", n ? "," : "", b);
    unsigned long cpus[64] = {0};
    check(nvmlDeviceGetCpuAffinity(h, 64, cpus));
    cpu_ranges(cpus, 64, g->cpu_affinity, sizeof g->cpu_affinity);
    int ok;
    unsigned v = field(h, FI_GPU_RECOVERY_ACTION, &ok);
    g->recovery = ok ? recovery_name(v) : "N/A";
    v = field(h, FI_NVLINK_SPEED_MBPS, &ok);
    g->nvlink_gbs = ok ? v / 1000.0 : 0;
    proc_t infos[MAX_PROCS];
    unsigned count = MAX_PROCS;
    check(nvmlDeviceGetComputeRunningProcesses_v3(h, &count, infos));
    for (unsigned i = 0; i < count && i < MAX_PROCS; i++) {
        procrow_t *p = &g->procs[g->nprocs++];
        p->pid = infos[i].pid;
        if (nvmlSystemGetProcessName(p->pid, p->name, sizeof p->name) != 0)
            strcpy(p->name, "[N/A]");
        p->mib = infos[i].usedGpuMemory >> 20;
        p->gi = infos[i].gpuInstanceId;
        p->ci = infos[i].computeInstanceId;
    }
    check(nvmlDeviceGetMigMode(h, &g->mig_current, &g->mig_pending));
    read_mig_devices(g);
}

static unsigned gpu_count(void)
{
    unsigned n;
    check(nvmlDeviceGetCount_v2(&n));
    return n;
}

/* GPU selection from -i; NULL: every GPU. */
static int selected(const int *ids, unsigned i) { return !ids || (i < MAX_GPUS && ids[i]); }

static gpu_t *read_gpus(const int *ids, unsigned *ndevs)
{
    unsigned n = gpu_count();
    gpu_t *devs = calloc(n ? n : 1, sizeof *devs);
    *ndevs = 0;
    for (unsigned i = 0; i < n; i++)
        if (selected(ids, i))
            read_gpu(&devs[(*ndevs)++], i);
    return devs;
}

static const char *driver_version(void)
{
    static char buf[80];
    check(nvmlSystemGetDriverVersion(buf, sizeof buf));
    return buf;
}

static const char *cuda_version(void)
{
    static char buf[16];
    int v;
    check(nvmlSystemGetCudaDriverVersion_v2(&v));
    snprintf(buf, sizeof buf, "%d.%d", v / 1000, v % 1000 / 10);
    return buf;
}

static void now(const char *fmt, char *buf, size_t len)
{
    time_t t = time(NULL);
    strftime(buf, len, fmt, localtime(&t));
}

/* Python's round(): halves to even. */
static long long round_even(double x)
{
    double f = floor(x), r = x - f;
    long long v = (long long)f;
    return r > 0.5 || (r == 0.5 && (v & 1)) ? v + 1 : v;
}

/* ---- the default table ------------------------------------------------------------ */

#define LINE "+-----------------------------------------------------------------------------------------+"

/* The MIG devices section of the default output (R580's layout, as a GB200
 * prints it): one row per compute instance; those of one GPU instance share
 * its memory cell. BAR1: GB200's 1g.47gb (46592 MiB here) has 18225 MiB
 * (dra-driver-nvidia-gpu#1454); others in proportion to their memory. */
static void mig_table(gpu_t *devs, unsigned n)
{
    const char *sep = "+------------------+----------------------------------+-----------+-----------------------+";
    const char *shared = "+------------------+                                  +-----------+-----------------------+";
    puts(LINE);
    puts("| MIG devices:                                                                            |");
    puts(sep);
    puts("| GPU  GI  CI  MIG |              Shared Memory-Usage |        Vol|        Shared         |");
    puts("|      ID  ID  Dev |                Shared BAR1-Usage | SM     Unc| CE ENC  DEC  OFA  JPG |");
    puts("|                  |                                  |        ECC|                       |");
    puts("|==================+==================================+===========+=======================|");
    int k = 0, prev_gpu = -1, prev_gi = -1;
    for (unsigned j = 0; j < n; j++) {
        gpu_t *g = &devs[j];
        if (!g->mig_current)
            continue;
        for (unsigned m = 0; m < g->nmig; m++) {
            migdev_t *d = &g->mig[m];
            int first = k == 0 || prev_gpu != (int)g->index || prev_gi != (int)d->gi;
            if (k)
                puts(first ? sep : shared);
            char usage[128] = "", bar1[128] = "", a[32], b[32];
            if (first) {
                snprintf(a, sizeof a, "%lluMiB", d->mem.used >> 20);
                snprintf(b, sizeof b, "%lluMiB", d->a.memorySizeMB);
                snprintf(usage, sizeof usage, "%19s / %-12s", a, b);
                snprintf(b, sizeof b, "%lldMiB", round_even(d->a.memorySizeMB * (18225.0 / 46592.0)));
                snprintf(bar1, sizeof bar1, "%19s / %-12s", "0MiB", b);
            }
            printf("|%3u%5u%4u%4u  |%-34s| %-3u%6d |%3u%4u%5u%5u%5u |\n", g->index, d->gi, d->ci, d->index, usage,
                   d->a.multiprocessorCount, 0, d->a.sharedCopyEngineCount, d->a.sharedEncoderCount,
                   d->a.sharedDecoderCount, d->a.sharedOfaCount, d->a.sharedJpegCount);
            printf("|%18s|%-34s|%11s|%23s|\n", "", bar1, "", "");
            k++;
            prev_gpu = (int)g->index;
            prev_gi = (int)d->gi;
        }
    }
    if (!k) {
        puts("|  No MIG devices found                                                                   |");
        puts(LINE);
        return;
    }
    puts(sep);
}

static void table(gpu_t *devs, unsigned n)
{
    const char *sep = "+-----------------------------------------+------------------------+----------------------+";
    char buf[64];
    now("%a %b %d %H:%M:%S %Y", buf, sizeof buf);
    puts(buf);
    puts(LINE);
    printf("| NVIDIA-SMI %-18s Driver Version: %-18s CUDA Version: %-8s |\n", driver_version(), driver_version(),
           cuda_version());
    puts(sep);
    puts("| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |");
    puts("| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |");
    puts("|                                         |                        |               MIG M. |");
    puts("+=========================================+========================+======================+");
    int any_mig = 0;
    for (unsigned j = 0; j < n; j++) {
        gpu_t *g = &devs[j];
        char left1[64], left2[64], power[64], memuse[64], util[16];
        snprintf(left1, sizeof left1, " %3u  %-24.24s", g->index, g->name);
        snprintf(left2, sizeof left2, " N/A   %2uC    P%-2u", g->temp, g->pstate);
        snprintf(power, sizeof power, "%.0fW / %uW", g->power, g->power_limit);
        snprintf(memuse, sizeof memuse, "%lluMiB / %lluMiB", g->mem_used, g->mem_total);
        /* A MIG-enabled GPU: ECC shown as on, utilisation per MIG device only (N/A). */
        if (g->mig_current)
            snprintf(util, sizeof util, "N/A");
        else
            snprintf(util, sizeof util, "%u%%", g->util);
        printf("|%s%*s | %-16s   Off | %20s |\n", left1, (int)(40 - strlen(left1)), "On", g->bus_id + 4,
               g->mig_current ? "On" : "0");
        printf("|%s%*s | %22s | %7s      Default |\n", left2, (int)(40 - strlen(left2)), power, memuse, util);
        printf("|%41s|%24s|%21s |\n", "", "", g->mig_current ? "Enabled" : "Disabled");
        puts(sep);
        any_mig |= g->mig_current != 0;
    }
    putchar('\n');
    if (any_mig) {
        mig_table(devs, n);
        putchar('\n');
    }
    puts(LINE);
    puts("| Processes:                                                                              |");
    puts("|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |");
    puts("|        ID   ID                                                               Usage      |");
    puts("|=========================================================================================|");
    int rows = 0;
    for (unsigned j = 0; j < n; j++)
        for (unsigned p = 0; p < devs[j].nprocs; p++) {
            procrow_t *r = &devs[j].procs[p];
            char name[64], gi[16], ci[16];
            size_t len = strlen(r->name);
            if (len <= 36)
                snprintf(name, sizeof name, "%s", r->name);
            else
                snprintf(name, sizeof name, "...%s", r->name + len - 33);
            if (r->gi == NONE_ID) {
                strcpy(gi, "N/A");
                strcpy(ci, "N/A");
            } else {
                snprintf(gi, sizeof gi, "%u", r->gi);
                snprintf(ci, sizeof ci, "%u", r->ci);
            }
            printf("|  %3u  %3s  %3s  %14u      C   %-36s %8lluMiB |\n", devs[j].index, gi, ci, r->pid, name, r->mib);
            rows++;
        }
    if (!rows)
        puts("|  No running processes found                                                             |");
    puts(LINE);
}

/* ---- other views ------------------------------------------------------------------ */

static void list_gpus(gpu_t *devs, unsigned n)
{
    for (unsigned j = 0; j < n; j++) {
        gpu_t *g = &devs[j];
        printf("GPU %u: %s (UUID: %s)\n", g->index, g->name, g->uuid);
        for (unsigned m = 0; m < g->nmig; m++) {
            char label[128];
            snprintf(label, sizeof label, "MIG %s", g->mig[m].name);
            printf("  %-16sDevice %2u: (UUID: %s)\n", label, g->mig[m].index, g->mig[m].uuid);
        }
    }
}

static void topo(gpu_t *devs, unsigned n)
{
    for (unsigned j = 0; j < n; j++)
        printf("\tGPU%u", devs[j].index);
    puts("\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID");
    for (unsigned j = 0; j < n; j++) {
        gpu_t *g = &devs[j];
        printf("GPU%u", g->index);
        /* Through NVSwitch, a pair is bonded by the links both ends still have up. */
        for (unsigned o = 0; o < n; o++) {
            if (devs[o].index == g->index)
                printf("\tX");
            else
                printf("\tNV%u", g->links_up < devs[o].links_up ? g->links_up : devs[o].links_up);
        }
        printf("\t%s\t%s\t\t%s\n", g->cpu_affinity, g->numa_affinity, g->gpu_numa);
    }
    puts("\nLegend:\n\n"
         "  X    = Self\n"
         "  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)\n"
         "  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node\n"
         "  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)\n"
         "  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)\n"
         "  PIX  = Connection traversing at most a single PCIe bridge\n"
         "  NV#  = Connection traversing a bonded set of # NVLinks");
}

static void nvlink_status(gpu_t *devs, unsigned n)
{
    for (unsigned j = 0; j < n; j++) {
        gpu_t *g = &devs[j];
        printf("GPU %u: %s (UUID: %s)\n", g->index, g->name, g->uuid);
        for (unsigned l = 0; l < g->nlinks; l++) {
            if (g->links[l])
                printf("\t Link %u: %g GB/s\n", l, g->nvlink_gbs);
            else
                printf("\t Link %u: <inactive>\n", l);
        }
    }
}

static void query(gpu_t *devs, unsigned n)
{
    char ts[64];
    now("%a %b %e %H:%M:%S %Y", ts, sizeof ts);
    printf("\n==============NVSMI LOG==============\n\n");
    printf("Timestamp                                 : %s\n", ts);
    printf("Driver Version                            : %s\n", driver_version());
    printf("CUDA Version                              : %s\n\n", cuda_version());
    printf("Attached GPUs                             : %u\n", gpu_count());
    for (unsigned j = 0; j < n; j++) {
        gpu_t *g = &devs[j];
        printf("GPU %s\n", g->bus_id);
        printf("    Product Name                          : %s\n", g->name);
        printf("    Product Architecture                  : Blackwell\n");
        printf("    Serial Number                         : %s\n", g->serial);
        printf("    GPU UUID                              : %s\n", g->uuid);
        printf("    Minor Number                          : %u\n", g->index);
        printf("    VBIOS Version                         : %s\n", g->vbios);
        printf("    GPU Recovery Action                   : %s\n", g->recovery);
        printf("    Fabric\n");
        printf("        State                             : %s\n", g->fabric_state);
        printf("        Status                            : Success\n");
        printf("        CliqueId                          : %u\n", g->clique);
        printf("        ClusterUUID                       : %s\n", g->cluster_uuid);
        printf("        Health\n");
        printf("            Bandwidth                     : Full\n");
        printf("            Summary                       : Healthy\n");
        printf("    MIG Mode\n");
        printf("        Current                           : %s\n", g->mig_current ? "Enabled" : "Disabled");
        printf("        Pending                           : %s\n", g->mig_pending ? "Enabled" : "Disabled");
        for (unsigned m = 0; m < g->nmig; m++) {
            migdev_t *d = &g->mig[m];
            printf("    MIG Device\n");
            printf("        Index                             : %u\n", d->index);
            printf("        GPU Instance ID                   : %u\n", d->gi);
            printf("        Compute Instance ID               : %u\n", d->ci);
            printf("        Device Attributes\n");
            printf("            Shared\n");
            printf("                Multiprocessor count      : %u\n", d->a.multiprocessorCount);
            printf("                Copy Engine count         : %u\n", d->a.sharedCopyEngineCount);
            printf("                Encoder count             : %u\n", d->a.sharedEncoderCount);
            printf("                Decoder count             : %u\n", d->a.sharedDecoderCount);
            printf("                OFA count                 : %u\n", d->a.sharedOfaCount);
            printf("                JPG count                 : %u\n", d->a.sharedJpegCount);
            printf("        FB Memory Usage\n");
            printf("            Total                         : %llu MiB\n", d->a.memorySizeMB);
        }
        printf("    FB Memory Usage\n");
        printf("        Total                             : %llu MiB\n", g->mem_total);
        printf("        Used                              : %llu MiB\n", g->mem_used);
        printf("        Free                              : %llu MiB\n", g->mem_total - g->mem_used);
    }
}

/* ---- --query-gpu, --query-compute-apps ------------------------------------------------ */

static const char *alias(const char *f)
{
    static const char *map[][2] = {
        {"fabric.clusterUuid", "fabric.cluster_uuid"}, {"fabric.cliqueId", "fabric.clique_id"},
        {"gpu_name", "name"}, {"gpu_uuid", "uuid"}, {"gpu_serial", "serial"}, {"gpu_bus_id", "pci.bus_id"},
        {"memory_total", "memory.total"}, {"memory_used", "memory.used"}, {"memory_free", "memory.free"}};
    for (size_t i = 0; i < sizeof map / sizeof *map; i++)
        if (!strcmp(f, map[i][0]))
            return map[i][1];
    return f;
}

/* A field's value for GPU g (NULL: not a field) and its unit ("" for none). */
static const char *field_value(const char *f, gpu_t *g, char *buf, size_t len, const char **unit)
{
    *unit = "";
#define IS(name) (!strcmp(f, name))
#define OUT(...) (snprintf(buf, len, __VA_ARGS__), buf)
    if (IS("index")) return OUT("%u", g->index);
    if (IS("name")) return OUT("%s", g->name);
    if (IS("uuid")) return OUT("%s", g->uuid);
    if (IS("serial")) return OUT("%s", g->serial);
    if (IS("pci.bus_id")) return OUT("%s", g->bus_id);
    if (IS("driver_version")) return OUT("%s", driver_version());
    if (IS("vbios_version")) return OUT("%s", g->vbios);
    if (IS("memory.total")) return *unit = "MiB", OUT("%llu", g->mem_total);
    if (IS("memory.used")) return *unit = "MiB", OUT("%llu", g->mem_used);
    if (IS("memory.free")) return *unit = "MiB", OUT("%llu", g->mem_total - g->mem_used);
    if (IS("utilization.gpu")) return *unit = "%", OUT("%u", g->util);
    if (IS("utilization.memory")) return *unit = "%", OUT("%u", g->mem_util);
    if (IS("temperature.gpu")) return OUT("%u", g->temp);
    if (IS("temperature.memory")) return OUT("%u", g->temp + 4);
    if (IS("power.draw")) return *unit = "W", OUT("%.2f", g->power);
    if (IS("power.limit") || IS("power.max_limit") || IS("power.default_limit"))
        return *unit = "W", OUT("%.2f", (double)g->power_limit);
    if (IS("pstate")) return OUT("P%u", g->pstate);
    if (IS("clocks.sm")) return *unit = "MHz", OUT("%u", g->sm_clock);
    if (IS("clocks.mem")) return *unit = "MHz", OUT("%u", g->mem_clock);
    if (IS("clocks.max.sm")) return *unit = "MHz", OUT("1965");
    if (IS("clocks.max.memory")) return *unit = "MHz", OUT("3996");
    if (IS("compute_cap")) return OUT("10.0");
    if (IS("persistence_mode")) return OUT("Enabled");
    if (IS("mig.mode.current")) return OUT("%s", g->mig_current ? "Enabled" : "Disabled");
    if (IS("mig.mode.pending")) return OUT("%s", g->mig_pending ? "Enabled" : "Disabled");
    if (IS("fabric.state")) return OUT("%s", g->fabric_state);
    if (IS("fabric.clique_id")) return OUT("%u", g->clique);
    if (IS("fabric.cluster_uuid")) return OUT("%s", g->cluster_uuid);
    if (IS("count")) return OUT("%u", gpu_count());
    if (IS("fan.speed")) return OUT("[N/A]");
    if (IS("timestamp")) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        char day[32];
        strftime(day, sizeof day, "%Y/%m/%d %H:%M:%S", localtime(&ts.tv_sec));
        return OUT("%s.%03ld", day, ts.tv_nsec / 1000000);
    }
#undef IS
#undef OUT
    return NULL;
}

/* Comma-separated list into trimmed, non-empty items. */
static int split(const char *list, char items[][64], int max)
{
    int n = 0;
    const char *p = list;
    while (*p && n < max) {
        const char *e = strchr(p, ',');
        size_t len = e ? (size_t)(e - p) : strlen(p);
        while (len && isspace((unsigned char)*p))
            p++, len--;
        while (len && isspace((unsigned char)p[len - 1]))
            len--;
        if (len) {
            snprintf(items[n], 64, "%.*s", (int)len, p);
            n++;
        }
        if (!e)
            break;
        p = e + 1;
    }
    return n;
}

static int has_opt(const char *fmt, const char *opt)
{
    char items[16][64];
    int n = split(fmt, items, 16);
    for (int i = 0; i < n; i++)
        if (!strcmp(items[i], opt))
            return 1;
    return 0;
}

static void query_gpu(gpu_t *devs, unsigned n, const char *list, const char *fmt)
{
    char fields[64][64], buf[128];
    int nf = split(list, fields, 64);
    const char *unit;
    gpu_t probe = {0};
    for (int i = 0; i < nf; i++) {
        char name[64];
        snprintf(name, sizeof name, "%s", alias(fields[i]));
        memcpy(fields[i], name, sizeof name);
        if (!field_value(fields[i], n ? &devs[0] : &probe, buf, sizeof buf, &unit))
            fail("Field \"%s\" is not a valid field to query.", fields[i]);
    }
    int nounits = has_opt(fmt, "nounits");
    if (!has_opt(fmt, "noheader")) {
        for (int i = 0; i < nf; i++) {
            field_value(fields[i], n ? &devs[0] : &probe, buf, sizeof buf, &unit);
            printf("%s%s", i ? ", " : "", fields[i]);
            if (*unit && !nounits)
                printf(" [%s]", unit);
        }
        putchar('\n');
    }
    for (unsigned j = 0; j < n; j++) {
        for (int i = 0; i < nf; i++) {
            const char *v = field_value(fields[i], &devs[j], buf, sizeof buf, &unit);
            printf("%s%s", i ? ", " : "", v);
            if (*unit && !nounits)
                printf(" %s", unit);
        }
        putchar('\n');
    }
}

static void query_apps(gpu_t *devs, unsigned n, const char *list, const char *fmt)
{
    static const char *app_fields[] = {"pid", "process_name", "used_memory", "used_gpu_memory", "gpu_uuid",
                                       "gpu_bus_id", "gpu_name"};
    char fields[64][64];
    int nf = split(list, fields, 64);
    for (int i = 0; i < nf; i++) {
        int ok = 0;
        for (size_t k = 0; k < sizeof app_fields / sizeof *app_fields; k++)
            ok |= !strcmp(fields[i], app_fields[k]);
        if (!ok)
            fail("Field \"%s\" is not a valid field to query.", fields[i]);
    }
    int nounits = has_opt(fmt, "nounits");
    if (!has_opt(fmt, "noheader")) {
        for (int i = 0; i < nf; i++) {
            int mem = !strcmp(fields[i], "used_memory") || !strcmp(fields[i], "used_gpu_memory");
            printf("%s%s%s", i ? ", " : "", fields[i], mem && !nounits ? " [MiB]" : "");
        }
        putchar('\n');
    }
    for (unsigned j = 0; j < n; j++) {
        gpu_t *g = &devs[j];
        for (unsigned p = 0; p < g->nprocs; p++) {
            procrow_t *r = &g->procs[p];
            for (int i = 0; i < nf; i++) {
                const char *f = fields[i];
                printf("%s", i ? ", " : "");
                if (!strcmp(f, "pid"))
                    printf("%u", r->pid);
                else if (!strcmp(f, "process_name"))
                    printf("%s", r->name);
                else if (!strcmp(f, "gpu_uuid"))
                    printf("%s", g->uuid);
                else if (!strcmp(f, "gpu_bus_id"))
                    printf("%s", g->bus_id);
                else if (!strcmp(f, "gpu_name"))
                    printf("%s", g->name);
                else
                    printf(nounits ? "%llu" : "%llu MiB", r->mib);
            }
            putchar('\n');
        }
    }
}

static void dmon(const int *ids, int count)
{
    puts("# gpu    pwr  gtemp  mtemp     sm    mem    enc    dec   mclk   pclk");
    puts("# Idx      W      C      C      %      %      %      %    MHz    MHz");
    for (int i = 0; count < 0 || i < count; i++) {
        unsigned n;
        gpu_t *devs = read_gpus(ids, &n);
        for (unsigned j = 0; j < n; j++) {
            gpu_t *g = &devs[j];
            printf("  %3u %6.0f %6u %6u %6u %6u %6d %6d %6u %6u\n", g->index, g->power, g->temp, g->temp + 4, g->util,
                   g->mem_util, 0, 0, g->mem_clock, g->sm_clock);
            fflush(stdout);
        }
        free(devs);
        if (count < 0 || i + 1 < count)
            sleep(1);
    }
}

/* Reset GPUs like `nvidia-smi --gpu-reset`: root only, and refused for a GPU
 * any process still uses. The GPU takes its NVLink partition's clique. */
static int gpu_reset(const int *ids)
{
    if (geteuid() != 0)
        fail("Unable to reset GPU: Insufficient Permissions");
    if (!fakegpu_reset_gpu)
        fail("Unable to reset GPU: Not Supported");
    unsigned n;
    gpu_t *devs = read_gpus(ids, &n);
    int failed = 0;
    for (unsigned j = 0; j < n; j++) {
        int rc = fakegpu_reset_gpu(devs[j].minor);
        if (rc == 0) {
            printf("GPU %s was successfully reset.\n", devs[j].bus_id);
            continue;
        }
        failed = 1;
        fflush(stdout);
        if (rc == -1)
            fprintf(stderr, "Unable to reset GPU %s because it's being used by some other process (e.g. CUDA "
                            "application). Please first kill all processes using this GPU.\n", devs[j].bus_id);
        else
            fprintf(stderr, "Unable to reset GPU %s: no GPU state available.\n", devs[j].bus_id);
    }
    puts(failed ? "Some GPUs could not be reset." : "All done.");
    return failed;
}

/* GPU indices from -i: indices or PCI bus ids (as mig-parted passes them). */
static int *parse_ids(const char *arg)
{
    if (!arg)
        fail("nvidia-smi: -i needs a value");
    static int ids[MAX_GPUS];
    memset(ids, 0, sizeof ids);
    char items[MAX_GPUS][64];
    int ni = split(arg, items, MAX_GPUS);
    unsigned n = gpu_count();
    for (int k = 0; k < ni; k++) {
        const char *x = items[k];
        char *end;
        unsigned long v = strtoul(x, &end, 10);
        if (*x && !*end && isdigit((unsigned char)*x)) {
            if (v < MAX_GPUS)
                ids[v] = 1;
            continue;
        }
        const char *tail = strchr(x, ':') ? strchr(x, ':') + 1 : x;
        unsigned long dom = strtoul(x, &end, 16);
        int found = 0;
        for (unsigned i = 0; i < n && !found; i++) {
            dev_t_ h;
            pci_t pci;
            check(nvmlDeviceGetHandleByIndex_v2(i, &h));
            check(nvmlDeviceGetPciInfo_v3(h, &pci));
            size_t bl = strlen(pci.busId), tl = strlen(tail);
            if (end != x && *end == ':' && tl <= bl && !strcasecmp(pci.busId + bl - tl, tail) &&
                strtoul(pci.busId, NULL, 16) == dom) {
                ids[i] = 1;
                found = 1;
            }
        }
        if (!found)
            fail("No devices were found matching %s", x);
    }
    return ids;
}

/* ---- MIG ----------------------------------------------------------------------------- */

typedef struct { unsigned index; dev_t_ h; char bus[32]; } handle_t;
typedef struct { unsigned id; void *gi; giinfo_t info; giprof_t prof; } gi_t;
typedef struct { unsigned id; void *ci; ciinfo_t info; ciprof_t prof; } ci_t;

static handle_t *handles(const int *ids, unsigned *count)
{
    unsigned n = gpu_count();
    handle_t *out = calloc(n ? n : 1, sizeof *out);
    *count = 0;
    for (unsigned i = 0; i < n; i++) {
        if (!selected(ids, i))
            continue;
        handle_t *h = &out[(*count)++];
        pci_t pci;
        h->index = i;
        check(nvmlDeviceGetHandleByIndex_v2(i, &h->h));
        check(nvmlDeviceGetPciInfo_v3(h->h, &pci));
        snprintf(h->bus, sizeof h->bus, "%s", pci.busId);
    }
    return out;
}

static const char *error(int ret) { return nvmlErrorString(ret); }

/* nvidia-smi -mig 0|1: at once on an idle GPU (Hopper and later), else pending
 * until a GPU reset; disabling with instances is refused, and without root
 * NVML refuses (Insufficient Permissions, exit code 4). */
static void set_mig_mode(const int *ids, const char *arg)
{
    int mode = atoi(arg);
    const char *word = mode ? "enable" : "disable";
    unsigned n;
    handle_t *devs = handles(ids, &n);
    for (unsigned j = 0; j < n; j++) {
        int act = 0;
        int ret = nvmlDeviceSetMigMode(devs[j].h, (unsigned)mode, &act);
        if (ret != 0) {
            printf("Unable to %s MIG Mode for GPU %s: %s\n", word, devs[j].bus, error(ret));
            puts("Terminating early due to previous errors.");
            exit(ret);
        }
        if (act == 0)
            printf("%s MIG Mode for GPU %s\n", mode ? "Enabled" : "Disabled", devs[j].bus);
        else if (act == IN_USE) {
            printf("Warning: MIG mode is in pending %s state for GPU %s:%s\n", word, devs[j].bus, error(act));
            printf("%s is currently being used by one or more other processes (e.g. CUDA application or a "
                   "monitoring application such as another instance of nvidia-smi). Please first kill all processes "
                   "using the device and retry the command or reboot the system to make MIG mode effective.\n",
                   devs[j].bus);
        } else {
            printf("Warning: MIG mode is in pending %s state for GPU %s:%s\n", word, devs[j].bus, error(act));
            printf("Reboot the system or try nvidia-smi --gpu-reset to make MIG mode effective on GPU %s\n",
                   devs[j].bus);
        }
    }
    puts("All done.");
}

static int cmp_prof(const void *a, const void *b)
{
    const giprof_t *x = a, *y = b;
    if (x->sliceCount != y->sliceCount)
        return x->sliceCount < y->sliceCount ? -1 : 1;
    if (x->memorySizeMB != y->memorySizeMB)
        return x->memorySizeMB < y->memorySizeMB ? -1 : 1;
    return strcmp(x->name, y->name);
}

/* GPU instance profiles, as nvidia-smi lists them: by size (1g.23gb,
 * 1g.23gb+me, 1g.47gb, 2g.47gb, ... 7g.186gb). */
static int gi_profiles(dev_t_ h, giprof_t *out)
{
    int n = 0;
    for (unsigned i = 0; i < MIG_PROFILE_COUNT; i++) {
        giprof_t info = {.version = (unsigned)sizeof(giprof_t) | (2u << 24)};
        if (nvmlDeviceGetGpuInstanceProfileInfoV(h, i, &info) == 0)
            out[n++] = info;
    }
    qsort(out, (size_t)n, sizeof *out, cmp_prof);
    return n;
}

/* A profile given by id or name (1g.23gb, MIG 1g.23gb). */
static giprof_t *profile_arg(giprof_t *profs, int n, const char *arg)
{
    for (int i = 0; i < n; i++) {
        char id[16];
        snprintf(id, sizeof id, "%u", profs[i].id);
        if (!strcmp(arg, id) || !strcmp(arg, profs[i].name) ||
            (strlen(profs[i].name) >= 4 && !strcmp(arg, profs[i].name + 4)))
            return &profs[i];
    }
    return NULL;
}

static int cmp_gi(const void *a, const void *b)
{
    const gi_t *x = a, *y = b;
    return x->id < y->id ? -1 : x->id > y->id;
}

/* A GPU's GPU instances, by id. */
static int gpu_instances(dev_t_ h, gi_t *out)
{
    giprof_t profs[MIG_PROFILE_COUNT];
    int np = gi_profiles(h, profs), n = 0;
    for (int i = 0; i < np; i++) {
        unsigned cnt = profs[i].instanceCount;
        void *buf[MAX_MIG];
        if (nvmlDeviceGetGpuInstances(h, profs[i].id, buf, &cnt) != 0)
            continue;
        for (unsigned k = 0; k < cnt && n < MAX_MIG; k++) {
            gi_t *g = &out[n++];
            g->gi = buf[k];
            check(nvmlGpuInstanceGetInfo(g->gi, &g->info));
            g->id = g->info.id;
            g->prof = profs[i];
        }
    }
    qsort(out, (size_t)n, sizeof *out, cmp_gi);
    return n;
}

static int ci_profiles(void *gi, ciprof_t *out)
{
    int n = 0;
    for (unsigned j = 0; j < MIG_CI_PROFILE_COUNT; j++) {
        ciprof_t info = {.version = (unsigned)sizeof(ciprof_t) | (2u << 24)};
        if (nvmlGpuInstanceGetComputeInstanceProfileInfoV(gi, j, 0, &info) == 0)
            out[n++] = info;
    }
    return n;
}

/* The default compute instance profile (marked * by -lcip): the most SMs. */
static ciprof_t default_ci(void *gi)
{
    ciprof_t profs[MIG_CI_PROFILE_COUNT];
    int n = ci_profiles(gi, profs), best = 0;
    for (int i = 1; i < n; i++)
        if (profs[i].multiprocessorCount > profs[best].multiprocessorCount)
            best = i;
    return profs[best];
}

static int cmp_ci(const void *a, const void *b)
{
    const ci_t *x = a, *y = b;
    return x->id < y->id ? -1 : x->id > y->id;
}

static int compute_instances(void *gi, ci_t *out)
{
    ciprof_t profs[MIG_CI_PROFILE_COUNT];
    int np = ci_profiles(gi, profs), n = 0;
    for (int i = 0; i < np; i++) {
        unsigned cnt = profs[i].instanceCount;
        void *buf[MAX_MIG];
        if (nvmlGpuInstanceGetComputeInstances(gi, profs[i].id, buf, &cnt) != 0)
            continue;
        for (unsigned k = 0; k < cnt && n < MAX_MIG; k++) {
            ci_t *c = &out[n++];
            c->ci = buf[k];
            check(nvmlComputeInstanceGetInfo_v2(c->ci, &c->info));
            c->id = c->info.id;
            c->prof = profs[i];
        }
    }
    qsort(out, (size_t)n, sizeof *out, cmp_ci);
    return n;
}

/* Creates a compute instance; -cgi -C names its id, -cci does not. */
static int create_ci(unsigned i, unsigned gid, void *gi, ciprof_t *c, int with_id)
{
    void *ci;
    int ret = nvmlGpuInstanceCreateComputeInstance(gi, c->id, &ci);
    if (ret != 0) {
        printf("Unable to create a compute instance on GPU %2u GPU instance ID %2u using profile %u: %s\n", i, gid,
               c->id, error(ret));
        return ret;
    }
    ciinfo_t info;
    check(nvmlComputeInstanceGetInfo_v2(ci, &info));
    char id[32] = "";
    if (with_id)
        snprintf(id, sizeof id, " ID %2u", info.id);
    printf("Successfully created compute instance%s on GPU %2u GPU instance ID %2u using profile %s (ID %2u)\n", id, i,
           gid, c->name, c->id);
    return 0;
}

/* A box of rows as nvidia-smi mig prints them; each row is one or two lines. */
typedef struct { char line[2][160]; int nlines; } row_t;

static void box(const char *title, const char **header, int nheader, row_t *rows, int nrows, int width)
{
    char line[256];
    line[0] = '+';
    memset(line + 1, '-', (size_t)width);
    line[width + 1] = '+';
    line[width + 2] = 0;
    puts(line);
    printf("| %-*s|\n", width - 1, title);
    for (int i = 0; i < nheader; i++)
        printf("| %-*s|\n", width - 1, header[i]);
    putchar('|');
    for (int i = 0; i < width; i++)
        putchar('=');
    puts("|");
    for (int i = 0; i < nrows; i++) {
        for (int l = 0; l < rows[i].nlines; l++)
            printf("|%-*s|\n", width, rows[i].line[l]);
        puts(line);
    }
    if (!nrows)
        puts(line);
}

static void list_gi_profiles(handle_t *devs, unsigned n)
{
    row_t *rows = calloc(n * MIG_PROFILE_COUNT + 1, sizeof *rows);
    int nr = 0;
    for (unsigned j = 0; j < n; j++) {
        giprof_t profs[MIG_PROFILE_COUNT];
        int np = gi_profiles(devs[j].h, profs);
        for (int k = 0; k < np; k++) {
            giprof_t *p = &profs[k];
            unsigned free_ = p->instanceCount;
            if (nvmlDeviceGetGpuInstanceRemainingCapacity(devs[j].h, p->id, &free_) != 0)
                free_ = p->instanceCount;
            char inst[32], mem[32];
            snprintf(inst, sizeof inst, "%u/%u", free_, p->instanceCount);
            snprintf(mem, sizeof mem, "%.2f", p->memorySizeMB / 1024.0);
            row_t *r = &rows[nr++];
            r->nlines = 2;
            snprintf(r->line[0], sizeof r->line[0], "%4u  %-18s%4u     %-11s%-11s%-7s%-7u%-6u%-4u", devs[j].index,
                     p->name, p->id, inst, mem, "No", p->multiprocessorCount, p->decoderCount, p->encoderCount);
            snprintf(r->line[1], sizeof r->line[1], "%62s%2u     %-6u%-4u", "", p->copyEngineCount, p->jpegCount,
                     p->ofaCount);
        }
    }
    const char *header[] = {"GPU   Name               ID    Instances   Memory     P2P    SM    DEC   ENC",
                            "                               Free/Total   GiB              CE    JPEG  OFA"};
    box("GPU instance profiles:", header, 2, rows, nr, 79);
    free(rows);
}

static void list_placements(handle_t *devs, unsigned n)
{
    for (unsigned j = 0; j < n; j++) {
        giprof_t profs[MIG_PROFILE_COUNT];
        int np = gi_profiles(devs[j].h, profs);
        for (int k = 0; k < np; k++) {
            unsigned cnt = 0;
            check(nvmlDeviceGetGpuInstancePossiblePlacements_v2(devs[j].h, profs[k].id, NULL, &cnt));
            place_t buf[MAX_MIG];
            if (cnt > MAX_MIG)
                cnt = MAX_MIG;
            check(nvmlDeviceGetGpuInstancePossiblePlacements_v2(devs[j].h, profs[k].id, buf, &cnt));
            printf("GPU %2u Profile ID %2u %s {", devs[j].index, profs[k].id, cnt == 1 ? "Placement :" : "Placements:");
            for (unsigned i = 0; i < cnt; i++)
                printf("%s%u", i ? "," : "", buf[i].start);
            printf("}:%u\n", buf[0].size);
        }
    }
}

static int in_set(const int *set, unsigned v) { return !set || (v < MAX_MIG && set[v]); }

static void list_ci_profiles(handle_t *devs, unsigned n, const int *gis)
{
    row_t *rows = calloc(n * MAX_MIG * MIG_CI_PROFILE_COUNT + 1, sizeof *rows);
    int nr = 0;
    for (unsigned j = 0; j < n; j++) {
        gi_t g[MAX_MIG];
        int ng = gpu_instances(devs[j].h, g);
        for (int k = 0; k < ng; k++) {
            if (!in_set(gis, g[k].id))
                continue;
            ciprof_t best = default_ci(g[k].gi), profs[MIG_CI_PROFILE_COUNT];
            int np = ci_profiles(g[k].gi, profs);
            for (int c = 0; c < np; c++) {
                unsigned free_ = profs[c].instanceCount;
                nvmlGpuInstanceGetComputeInstanceRemainingCapacity(g[k].gi, profs[c].id, &free_);
                char inst[32];
                snprintf(inst, sizeof inst, "%u/%u", free_, profs[c].instanceCount);
                row_t *r = &rows[nr++];
                r->nlines = 2;
                snprintf(r->line[0], sizeof r->line[0], "%4u%7u       %-20s%2u%c     %-14s%-10u%-6u%-6u%-4u",
                         devs[j].index, g[k].id, profs[c].name, profs[c].id, profs[c].id == best.id ? '*' : ' ', inst,
                         profs[c].multiprocessorCount, profs[c].sharedDecoderCount, profs[c].sharedEncoderCount,
                         profs[c].sharedOfaCount);
                snprintf(r->line[1], sizeof r->line[1], "%70s%-6u%-10u", "", profs[c].sharedCopyEngineCount,
                         profs[c].sharedJpegCount);
            }
        }
    }
    const char *header[] = {"GPU     GPU       Name             Profile  Instances   Exclusive       Shared",
                            "      Instance                       ID     Free/Total     SM       DEC   ENC   OFA",
                            "        ID                                                          CE    JPEG"};
    box("Compute instance profiles:", header, 3, rows, nr, 86);
    free(rows);
}

static void list_gis(handle_t *devs, unsigned n)
{
    row_t *rows = calloc(n * MAX_MIG + 1, sizeof *rows);
    int nr = 0;
    for (unsigned j = 0; j < n; j++) { /* by profile, as profiles are listed, then by id */
        giprof_t profs[MIG_PROFILE_COUNT];
        int np = gi_profiles(devs[j].h, profs);
        gi_t g[MAX_MIG];
        int ng = gpu_instances(devs[j].h, g);
        for (int p = 0; p < np; p++)
            for (int k = 0; k < ng; k++) {
                if (g[k].prof.id != profs[p].id)
                    continue;
                row_t *r = &rows[nr++];
                r->nlines = 1;
                snprintf(r->line[0], sizeof r->line[0], "%4u  %-17s%2u%9u%11u:%-6u", devs[j].index, g[k].prof.name,
                         g[k].prof.id, g[k].id, g[k].info.placement.start, g[k].info.placement.size);
            }
    }
    if (!nr) {
        puts("No GPU instances found: Not Found");
        exit(NOT_FOUND);
    }
    const char *header[] = {"GPU   Name          Profile  Instance   Placement",
                            "                      ID       ID       Start:Size"};
    box("GPU instances:", header, 2, rows, nr, 52);
    free(rows);
}

static void list_cis(handle_t *devs, unsigned n, const int *gis)
{
    row_t *rows = calloc(n * MAX_MIG * MAX_MIG + 1, sizeof *rows);
    int nr = 0;
    for (unsigned j = 0; j < n; j++) {
        gi_t g[MAX_MIG];
        int ng = gpu_instances(devs[j].h, g);
        for (int k = 0; k < ng; k++) {
            if (!in_set(gis, g[k].id))
                continue;
            ci_t c[MAX_MIG];
            int nc = compute_instances(g[k].gi, c);
            for (int i = 0; i < nc; i++) {
                row_t *r = &rows[nr++];
                r->nlines = 1;
                snprintf(r->line[0], sizeof r->line[0], "  %2u     %2u       %-18s%3u       %3u       %3u:%-5u",
                         devs[j].index, g[k].id, c[i].prof.name, c[i].prof.id, c[i].id, c[i].info.placement.start,
                         c[i].info.placement.size);
            }
        }
    }
    if (!nr) {
        puts("No compute instances found: Not Found");
        exit(NOT_FOUND);
    }
    const char *header[] = {"GPU     GPU       Name             Profile   Instance   Placement",
                            "      Instance                       ID        ID       Start:Size",
                            "        ID"};
    box("Compute instances:", header, 3, rows, nr, 68);
    free(rows);
}

static void create_gis(handle_t *devs, unsigned n, const char *arg, int with_default)
{
    int failed = 0;
    char items[MAX_MIG][64];
    int ni = split(arg ? arg : "", items, MAX_MIG);
    for (unsigned j = 0; j < n; j++) {
        giprof_t profs[MIG_PROFILE_COUNT];
        int np = gi_profiles(devs[j].h, profs);
        for (int k = 0; k < ni; k++) {
            giprof_t *p = profile_arg(profs, np, items[k]);
            if (!p) {
                printf("Unable to create a GPU instance on GPU %2u using profile %s: Invalid Argument\n",
                       devs[j].index, items[k]);
                failed = failed ? failed : 2;
                continue;
            }
            void *gi;
            int ret = nvmlDeviceCreateGpuInstance(devs[j].h, p->id, &gi);
            if (ret != 0) {
                printf("Unable to create a GPU instance on GPU %2u using profile %s: %s\n", devs[j].index, items[k],
                       error(ret));
                failed = failed ? failed : ret;
                continue;
            }
            giinfo_t info;
            check(nvmlGpuInstanceGetInfo(gi, &info));
            printf("Successfully created GPU instance ID %2u on GPU %2u using profile %s (ID %2u)\n", info.id,
                   devs[j].index, p->name, p->id);
            if (with_default) {
                ciprof_t c = default_ci(gi);
                int r = create_ci(devs[j].index, info.id, gi, &c, 1);
                failed = r ? r : failed;
            }
        }
    }
    if (failed) {
        printf("Failed to create GPU instances: %s\n", error(failed));
        exit(failed);
    }
}

static void create_cis(handle_t *devs, unsigned n, const char *arg, const int *gis)
{
    int failed = 0;
    char items[MAX_MIG][64];
    int ni = arg ? split(arg, items, MAX_MIG) : 0;
    for (unsigned j = 0; j < n; j++) {
        gi_t g[MAX_MIG];
        int ng = gpu_instances(devs[j].h, g);
        for (int k = 0; k < ng; k++) {
            if (!in_set(gis, g[k].id))
                continue;
            ciprof_t profs[MIG_CI_PROFILE_COUNT];
            int np = ci_profiles(g[k].gi, profs);
            if (!ni) {
                ciprof_t c = default_ci(g[k].gi);
                int r = create_ci(devs[j].index, g[k].id, g[k].gi, &c, 0);
                failed = r ? r : failed;
                continue;
            }
            for (int w = 0; w < ni; w++) {
                ciprof_t *c = NULL;
                for (int p = 0; p < np && !c; p++) {
                    char id[16];
                    snprintf(id, sizeof id, "%u", profs[p].id);
                    if (!strcmp(items[w], id) || !strcmp(items[w], profs[p].name) ||
                        (strlen(profs[p].name) >= 4 && !strcmp(items[w], profs[p].name + 4)))
                        c = &profs[p];
                }
                if (!c) {
                    failed = failed ? failed : 2;
                    continue;
                }
                int r = create_ci(devs[j].index, g[k].id, g[k].gi, c, 0);
                failed = r ? r : failed;
            }
        }
    }
    if (failed) {
        printf("Failed to create compute instances: %s\n", error(failed));
        exit(failed);
    }
}

static void destroy_cis(handle_t *devs, unsigned n, const int *gis, const int *cis)
{
    int found = 0, failed = 0;
    for (unsigned j = 0; j < n; j++) {
        gi_t g[MAX_MIG];
        int ng = gpu_instances(devs[j].h, g);
        for (int k = 0; k < ng; k++) {
            if (!in_set(gis, g[k].id))
                continue;
            ci_t c[MAX_MIG];
            int nc = compute_instances(g[k].gi, c);
            for (int i = 0; i < nc; i++) {
                if (!in_set(cis, c[i].id))
                    continue;
                found++;
                int ret = nvmlComputeInstanceDestroy(c[i].ci);
                if (ret != 0) {
                    printf("Unable to destroy compute instance ID %2u from GPU %2u GPU instance ID %2u: %s\n", c[i].id,
                           devs[j].index, g[k].id, error(ret));
                    failed = failed ? failed : ret;
                } else
                    printf("Successfully destroyed compute instance ID %2u from GPU %2u GPU instance ID %2u\n",
                           c[i].id, devs[j].index, g[k].id);
            }
        }
    }
    if (!found) {
        puts("No compute instances found: Not Found");
        exit(NOT_FOUND);
    }
    if (failed) {
        printf("Failed to destroy compute instances: %s\n", error(failed));
        exit(failed);
    }
}

static void destroy_gis(handle_t *devs, unsigned n, const int *gis)
{
    int found = 0, failed = 0;
    for (unsigned j = 0; j < n; j++) {
        gi_t g[MAX_MIG];
        int ng = gpu_instances(devs[j].h, g);
        for (int k = 0; k < ng; k++) {
            if (!in_set(gis, g[k].id))
                continue;
            found++;
            int ret = nvmlGpuInstanceDestroy(g[k].gi);
            if (ret != 0) {
                printf("Unable to destroy GPU instance ID %2u from GPU %2u: %s\n", g[k].id, devs[j].index, error(ret));
                failed = failed ? failed : ret;
            } else
                printf("Successfully destroyed GPU instance ID %2u from GPU %2u\n", g[k].id, devs[j].index);
        }
    }
    if (!found) {
        puts("No GPU instances found: Not Found");
        exit(NOT_FOUND);
    }
    if (failed) {
        printf("Failed to destroy GPU instances: %s\n", error(failed));
        exit(failed);
    }
}

static int *int_set(const char *arg)
{
    if (!arg)
        fail("nvidia-smi: missing value");
    int *set = calloc(MAX_MIG, sizeof *set);
    char items[MAX_MIG][64];
    int n = split(arg, items, MAX_MIG);
    for (int i = 0; i < n; i++) {
        int v = atoi(items[i]);
        if (v >= 0 && v < MAX_MIG)
            set[v] = 1;
    }
    return set;
}

static void unsupported(const char *prefix, int argc, char **argv)
{
    fprintf(stderr, "nvidia-smi (fakegpu): unsupported arguments:%s", prefix);
    for (int i = 0; i < argc; i++)
        fprintf(stderr, " %s", argv[i]);
    fputc('\n', stderr);
    exit(1);
}

static int mig(int argc, char **argv)
{
    static const char *names[][2] = {
        {"--list-gpu-instance-profiles", "lgip"}, {"--list-gpu-instance-possible-placements", "lgipp"},
        {"--list-compute-instance-profiles", "lcip"}, {"--list-gpu-instances", "lgi"},
        {"--list-compute-instances", "lci"}, {"--destroy-compute-instance", "dci"},
        {"--destroy-gpu-instance", "dgi"}, {"--create-gpu-instance", "cgi"},
        {"--create-compute-instance", "cci"}};
    static const char *shorts[] = {"-lgip", "-lgipp", "-lcip", "-lgi", "-lci", "-dci", "-dgi", "-cgi", "-cci"};
    const int *ids = NULL, *gis = NULL, *cis = NULL;
    const char *cmd = NULL, *arg = NULL;
    int with_default = 0;
    for (int i = 0; i < argc; i++) {
        const char *a = argv[i], *value = i + 1 < argc ? argv[i + 1] : NULL;
        const char *c = NULL;
        for (size_t k = 0; k < sizeof shorts / sizeof *shorts; k++)
            if (!strcmp(a, shorts[k]))
                c = a + 1;
        for (size_t k = 0; k < sizeof names / sizeof *names; k++)
            if (!strcmp(a, names[k][0]))
                c = names[k][1];
        if (!strcmp(a, "-i") || !strcmp(a, "--id"))
            ids = parse_ids(value), i++;
        else if (!strcmp(a, "-gi") || !strcmp(a, "--gpu-instance-id"))
            gis = int_set(value), i++;
        else if (!strcmp(a, "-ci") || !strcmp(a, "--compute-instance-id"))
            cis = int_set(value), i++;
        else if (!strcmp(a, "-C") || !strcmp(a, "--default-compute-instance"))
            with_default = 1;
        else if (c) {
            cmd = c;
            /* profiles: required for -cgi, optional for -cci */
            if ((!strcmp(c, "cgi") || !strcmp(c, "cci")) && value && value[0] != '-')
                arg = value, i++;
        } else
            unsupported(" mig", argc, argv);
    }
    unsigned n;
    handle_t *devs = handles(ids, &n);
    if (!cmd)
        fail("nvidia-smi (fakegpu): mig: no command");
    if (!strcmp(cmd, "lgip"))
        list_gi_profiles(devs, n);
    else if (!strcmp(cmd, "lgipp"))
        list_placements(devs, n);
    else if (!strcmp(cmd, "lcip"))
        list_ci_profiles(devs, n, gis);
    else if (!strcmp(cmd, "lgi"))
        list_gis(devs, n);
    else if (!strcmp(cmd, "lci"))
        list_cis(devs, n, gis);
    else if (!strcmp(cmd, "cgi"))
        create_gis(devs, n, arg, with_default);
    else if (!strcmp(cmd, "cci"))
        create_cis(devs, n, arg, gis);
    else if (!strcmp(cmd, "dci"))
        destroy_cis(devs, n, gis, cis);
    else if (!strcmp(cmd, "dgi"))
        destroy_gis(devs, n, gis);
    return 0;
}

/* nvidia-smi --version, as NVIDIA's prints it (NVIDIA's DRA driver checks a
 * driver root with it). */
static int version(void)
{
    char nvml[80] = "";
    const char *driver = driver_version();
    nvmlSystemGetNVMLVersion(nvml, sizeof nvml);
    printf("NVIDIA-SMI version  : %s\n", driver);
    printf("NVML version        : %s\n", nvml[0] ? nvml : driver);
    printf("DRIVER version      : %s\n", driver);
    printf("CUDA Version        : %s\n", cuda_version());
    return 0;
}

static const char *need(int argc, char **argv, int *i)
{
    if (*i + 1 >= argc)
        fail("nvidia-smi: %s needs a value", argv[*i]);
    return argv[++*i];
}

int main(int argc, char **argv)
{
    load_nvml();
    check(nvmlInit_v2());
    argc--, argv++;
    if (argc && !strcmp(argv[0], "mig"))
        return mig(argc - 1, argv + 1);
    if (argc && !strcmp(argv[0], "--fakegpu-driver-load")) { /* lab-only: the tray's boot (MIG instances go) */
        if (fakegpu_driver_load)
            fakegpu_driver_load();
        return 0;
    }
    if (argc && !strcmp(argv[0], "--version"))
        return version();
    const int *ids = NULL;
    const char *fmt = "csv", *fields = NULL, *apps = NULL, *mig_mode = NULL;
    double loop = 0;
    int count = -1;
    char **rest = calloc((size_t)argc + 1, sizeof *rest);
    int nrest = 0;
    for (int i = 0; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-i") || !strcmp(a, "--id"))
            ids = parse_ids(need(argc, argv, &i));
        else if (!strncmp(a, "--id=", 5))
            ids = parse_ids(a + 5);
        else if (!strcmp(a, "-mig") || !strcmp(a, "--multi-instance-gpu"))
            mig_mode = need(argc, argv, &i);
        else if (!strncmp(a, "--query-gpu=", 12))
            fields = a + 12;
        else if (!strncmp(a, "--query-compute-apps=", 21))
            apps = a + 21;
        else if (!strcmp(a, "-l") || !strcmp(a, "--loop"))
            loop = atof(need(argc, argv, &i));
        else if (!strncmp(a, "--loop=", 7))
            loop = atof(a + 7);
        else if (!strcmp(a, "-c"))
            count = atoi(need(argc, argv, &i));
        else if (!strncmp(a, "--format=", 9))
            fmt = a + 9;
        else if (!strcmp(a, "-d") || !strcmp(a, "--display"))
            need(argc, argv, &i);
        else
            rest[nrest++] = argv[i];
    }
#define REST(k, s) (nrest > (k) && !strcmp(rest[k], s))
    if (REST(0, "dmon")) {
        dmon(ids, count);
        return 0;
    }
    if (REST(0, "-r") || REST(0, "--gpu-reset"))
        return gpu_reset(ids);
    if (mig_mode) {
        set_mig_mode(ids, mig_mode);
        return 0;
    }
    for (;;) {
        unsigned n;
        gpu_t *devs = read_gpus(ids, &n);
        if (apps)
            query_apps(devs, n, apps, fmt);
        else if (fields)
            query_gpu(devs, n, fields, fmt);
        else if (REST(0, "-L") || REST(0, "--list-gpus"))
            list_gpus(devs, n);
        else if (REST(0, "topo") && REST(1, "-m"))
            topo(devs, n);
        else if (REST(0, "topo") && REST(1, "-gnid")) /* the GPU's own NUMA node, N/A where its memory is not one */
            for (unsigned j = 0; j < n; j++)
                puts(devs[j].gpu_numa);
        else if (REST(0, "nvlink"))
            nvlink_status(devs, n);
        else if (REST(0, "-q") || REST(0, "--query"))
            query(devs, n);
        else if (!nrest)
            table(devs, n);
        else
            unsupported("", argc, argv);
        free(devs);
        fflush(stdout);
        if (loop <= 0)
            break;
        usleep((useconds_t)(loop * 1e6));
    }
#undef REST
    return 0;
}
