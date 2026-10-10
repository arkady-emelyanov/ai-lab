/*
 * Shared fake device model for the CUDA and NVML stubs.
 *
 * Both libraries derive device identity from the same inputs so that CUDA
 * ordinals, NVML handles and nvidia-smi output agree. Settings come from
 * /etc/fakegpu.conf (key = value, '#' comments; path overridable with
 * FAKEGPU_CONF), and each key can be overridden by env FAKEGPU_<KEY>:
 *   count         GPUs in this tray (default 4, max 8)
 *   name          marketing name (default "NVIDIA GB200")
 *   mem_mb        memory per GPU in MiB (default 189471)
 *   cluster_uuid  NVLink domain UUID, identical on every tray of the domain
 *   clique_id     NVLink partition (clique) id within the domain
 *   sideband_dir  directory shared with the tray's BMC; its nvlink-disabled
 *                 file lists "<gpu> <link>" pairs reported as inactive
 *   latency_scale multiplier for simulated operation times (default 1.0,
 *                 0 = no delays)
 *   copy_max_mb   copies larger than this are timed but not performed, so
 *                 big "device" buffers never touch host RAM (default 64)
 *   state_path    shared occupancy file (default /dev/shm/fakegpu)
 *   dev_dir       where the GPU device nodes are looked up (default /dev;
 *                 tests point it at a directory of plain files)
 *   nccl_dir      directory every rank of a job can reach (/shared in the
 *                 lab) where NCCL communicators exchange each rank's NVLink
 *                 partition at init; empty: no exchange, all ranks are
 *                 assumed to share one partition
 *   nccl_timeout_s  how long a rank waits for its peers there (default 60)
 *   fabric_dir    directory every tray of the NVLink domain can reach, where
 *                 memory exported as a fabric handle lives (cuMemCreate with
 *                 CU_MEM_HANDLE_TYPE_FABRIC); empty: no fabric handles
 *   ipc_dir       directory every GPU process on the tray can reach (also in
 *                 pods), where GPU memory shared through CUDA IPC lives
 *                 (default /dev/shm/fakegpu-ipc)
 *   ib_gbps       InfiniBand bandwidth per GPU (one NIC each, default 400)
 *   driver_version  driver version NVML and nvidia-smi report (580.95.05)
 *   cuda_version  highest CUDA version the driver supports, MAJOR.MINOR
 *                 (13.0): cuDriverGetVersion, NVML, nvidia-smi
 *   vbios_version GPU VBIOS version (97.00.82.00.0F)
 *   GPU profile (defaults: GB200):
 *   nvlinks       NVLinks per GPU (18, at most 64)
 *   nvlink_link_gbs  bandwidth per NVLink and direction, GB/s (50: 900 per GPU)
 *   power_limit_w power limit (1200); idle_power_w, max_power_w: draw idle
 *                 and at full utilisation (140, 1000)
 *   sm_count      streaming multiprocessors (148)
 *   tensor_tflops, tf32_tflops, fp8_tflops  dense tensor-core throughput: BF16/FP16, TF32, FP8 (2500, 1250, 5000)
 *   fp32_tflops, fp64_tflops  dense FP32 without tensor cores, FP64 (60, 40)
 *   host_link_gbs host <-> GPU copies, GB/s (400, NVLink-C2C)
 *   hbm_gbs       device-to-device copies, GB/s (4000)
 *   coherent_gpu_memory  os (default) or driver, like the driver option
 *                 NVreg_CoherentGPUMemoryMode: with os, each GPU's memory is
 *                 a NUMA node (GB200: GPUs on nodes 2, 10, 18, 26, after the
 *                 two Grace CPUs' nodes 0 and 1, each followed by 7 MIG
 *                 nodes); with driver (CDMM) it is not
 *   host          tray name GPU identities derive from (default: hostname;
 *                 set it so containers, whose hostname differs, see the
 *                 tray's GPUs)
 *
 * GPU UUIDs are a hash of the tray name + index: stable per tray, unique
 * across trays.
 */
#ifndef FAKEGPU_H
#define FAKEGPU_H

#include <ctype.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define FG_MAX_GPUS 8
#define FG_CC_MAJOR 10
#define FG_CC_MINOR 0
#define FG_PCI_DEVICE_ID 0x294110DE
#define FG_PCI_SUBSYSTEM_ID 0x204610DE

/* As on a GB200 compute tray (lspci in NVIDIA/mig-parted#227 and
 * NVIDIA/kubevirt-gpu-device-plugin#154): each GPU in a PCI domain of its
 * own, behind its root port at <domain>:00:00.0, as 0008:01:00.0,
 * 0009:01:00.0 (Grace 0), 0018:01:00.0, 0019:01:00.0 (Grace 1). A tray has
 * four GPUs; the domains of GPUs 4-7 continue the pattern and are the lab's. */
#define FG_PCI_BUS 0x01
static const unsigned fg_pci_domain[FG_MAX_GPUS] = {0x0008, 0x0009, 0x0018, 0x0019, 0x0028, 0x0029, 0x0038, 0x0039};

static struct {
    int count;
    char name[96];
    unsigned long long mem_mb;
    unsigned char cluster_uuid[16];
    char cluster_uuid_str[40];
    unsigned clique_id;
    char sideband_dir[200];
    double latency_scale;
    unsigned long long copy_max_mb;
    char state_path[200];
    char host[128];
    char dev_dir[200];
    char nccl_dir[200];
    char fabric_dir[200];
    char ipc_dir[200];
    double nccl_timeout_s;
    double ib_gbps;
    char driver_version[32];
    char cuda_version[16];
    char vbios_version[32];
    int nvlinks, sm_count;
    double nvlink_link_gbs, power_limit_w, idle_power_w, max_power_w;
    double tensor_tflops, tf32_tflops, fp8_tflops, fp32_tflops, fp64_tflops, host_link_gbs, hbm_gbs;
    char coherent_gpu_memory[16];
} fg_cfg;
static pthread_once_t fg_cfg_once = PTHREAD_ONCE_INIT;

static char *fg_trim(char *s)
{
    while (isspace((unsigned char)*s)) s++;
    char *e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1])) *--e = 0;
    if (e - s >= 2 && (*s == '"' || *s == '\'') && e[-1] == *s) { e[-1] = 0; s++; }
    return s;
}

static void fg_set(const char *key, const char *val)
{
    if (!strcmp(key, "count")) fg_cfg.count = atoi(val);
    else if (!strcmp(key, "name")) snprintf(fg_cfg.name, sizeof fg_cfg.name, "%s", val);
    else if (!strcmp(key, "mem_mb")) fg_cfg.mem_mb = strtoull(val, NULL, 10);
    else if (!strcmp(key, "cluster_uuid")) snprintf(fg_cfg.cluster_uuid_str, sizeof fg_cfg.cluster_uuid_str, "%s", val);
    else if (!strcmp(key, "clique_id")) fg_cfg.clique_id = (unsigned)strtoul(val, NULL, 0);
    else if (!strcmp(key, "sideband_dir")) snprintf(fg_cfg.sideband_dir, sizeof fg_cfg.sideband_dir, "%s", val);
    else if (!strcmp(key, "latency_scale")) fg_cfg.latency_scale = strtod(val, NULL);
    else if (!strcmp(key, "copy_max_mb")) fg_cfg.copy_max_mb = strtoull(val, NULL, 10);
    else if (!strcmp(key, "state_path")) snprintf(fg_cfg.state_path, sizeof fg_cfg.state_path, "%s", val);
    else if (!strcmp(key, "host")) snprintf(fg_cfg.host, sizeof fg_cfg.host, "%s", val);
    else if (!strcmp(key, "dev_dir")) snprintf(fg_cfg.dev_dir, sizeof fg_cfg.dev_dir, "%s", val);
    else if (!strcmp(key, "nccl_dir")) snprintf(fg_cfg.nccl_dir, sizeof fg_cfg.nccl_dir, "%s", val);
    else if (!strcmp(key, "fabric_dir")) snprintf(fg_cfg.fabric_dir, sizeof fg_cfg.fabric_dir, "%s", val);
    else if (!strcmp(key, "ipc_dir")) snprintf(fg_cfg.ipc_dir, sizeof fg_cfg.ipc_dir, "%s", val);
    else if (!strcmp(key, "nccl_timeout_s")) fg_cfg.nccl_timeout_s = strtod(val, NULL);
    else if (!strcmp(key, "ib_gbps")) fg_cfg.ib_gbps = strtod(val, NULL);
    else if (!strcmp(key, "driver_version")) snprintf(fg_cfg.driver_version, sizeof fg_cfg.driver_version, "%s", val);
    else if (!strcmp(key, "cuda_version")) snprintf(fg_cfg.cuda_version, sizeof fg_cfg.cuda_version, "%s", val);
    else if (!strcmp(key, "vbios_version")) snprintf(fg_cfg.vbios_version, sizeof fg_cfg.vbios_version, "%s", val);
    else if (!strcmp(key, "nvlinks")) fg_cfg.nvlinks = atoi(val);
    else if (!strcmp(key, "sm_count")) fg_cfg.sm_count = atoi(val);
    else if (!strcmp(key, "nvlink_link_gbs")) fg_cfg.nvlink_link_gbs = strtod(val, NULL);
    else if (!strcmp(key, "power_limit_w")) fg_cfg.power_limit_w = strtod(val, NULL);
    else if (!strcmp(key, "idle_power_w")) fg_cfg.idle_power_w = strtod(val, NULL);
    else if (!strcmp(key, "max_power_w")) fg_cfg.max_power_w = strtod(val, NULL);
    else if (!strcmp(key, "tensor_tflops")) fg_cfg.tensor_tflops = strtod(val, NULL);
    else if (!strcmp(key, "tf32_tflops")) fg_cfg.tf32_tflops = strtod(val, NULL);
    else if (!strcmp(key, "fp8_tflops")) fg_cfg.fp8_tflops = strtod(val, NULL);
    else if (!strcmp(key, "fp32_tflops")) fg_cfg.fp32_tflops = strtod(val, NULL);
    else if (!strcmp(key, "fp64_tflops")) fg_cfg.fp64_tflops = strtod(val, NULL);
    else if (!strcmp(key, "host_link_gbs")) fg_cfg.host_link_gbs = strtod(val, NULL);
    else if (!strcmp(key, "hbm_gbs")) fg_cfg.hbm_gbs = strtod(val, NULL);
    else if (!strcmp(key, "coherent_gpu_memory")) snprintf(fg_cfg.coherent_gpu_memory, sizeof fg_cfg.coherent_gpu_memory, "%s", val);
}

static void fg_load(void)
{
    static const char *keys[] = {"count", "name", "mem_mb", "cluster_uuid", "clique_id", "sideband_dir",
                                 "latency_scale", "copy_max_mb", "state_path", "host", "dev_dir", "nccl_dir", "fabric_dir", "ipc_dir",
                                 "nccl_timeout_s", "ib_gbps", "driver_version", "cuda_version", "vbios_version", "nvlinks", "sm_count",
                                 "nvlink_link_gbs", "power_limit_w", "idle_power_w", "max_power_w", "tensor_tflops",
                                 "tf32_tflops", "fp8_tflops", "fp32_tflops", "fp64_tflops", "host_link_gbs", "hbm_gbs", "coherent_gpu_memory"};
    fg_cfg.count = 4;
    snprintf(fg_cfg.name, sizeof fg_cfg.name, "NVIDIA GB200");
    fg_cfg.mem_mb = 189471;
    snprintf(fg_cfg.cluster_uuid_str, sizeof fg_cfg.cluster_uuid_str, "00000000-0000-0000-0000-000000000000");
    fg_cfg.clique_id = 1;
    fg_cfg.latency_scale = 1.0;
    fg_cfg.copy_max_mb = 64;
    snprintf(fg_cfg.state_path, sizeof fg_cfg.state_path, "/dev/shm/fakegpu");
    snprintf(fg_cfg.ipc_dir, sizeof fg_cfg.ipc_dir, "/dev/shm/fakegpu-ipc");
    snprintf(fg_cfg.dev_dir, sizeof fg_cfg.dev_dir, "/dev");
    fg_cfg.nccl_timeout_s = 60;
    fg_cfg.ib_gbps = 400;
    snprintf(fg_cfg.driver_version, sizeof fg_cfg.driver_version, "580.95.05");
    snprintf(fg_cfg.cuda_version, sizeof fg_cfg.cuda_version, "13.0");
    snprintf(fg_cfg.vbios_version, sizeof fg_cfg.vbios_version, "97.00.82.00.0F");
    fg_cfg.nvlinks = 18;            /* NVLink5 links per Blackwell GPU, all to NVSwitch */
    fg_cfg.sm_count = 148;
    fg_cfg.nvlink_link_gbs = 50;    /* 18 x 50 = 900 GB/s per direction */
    fg_cfg.power_limit_w = 1200;
    fg_cfg.idle_power_w = 140;
    fg_cfg.max_power_w = 1000;
    fg_cfg.tensor_tflops = 2500;    /* dense BF16/FP16, tensor cores */
    fg_cfg.tf32_tflops = 1250;
    fg_cfg.fp8_tflops = 5000;
    fg_cfg.fp32_tflops = 60;
    fg_cfg.fp64_tflops = 40;
    fg_cfg.host_link_gbs = 400;     /* NVLink-C2C */
    fg_cfg.hbm_gbs = 4000;          /* device-to-device: 8 TB/s read + write */
    snprintf(fg_cfg.coherent_gpu_memory, sizeof fg_cfg.coherent_gpu_memory, "os");

    const char *path = getenv("FAKEGPU_CONF");
    FILE *f = fopen(path ? path : "/etc/fakegpu.conf", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof line, f)) {
            char *hash = strchr(line, '#'), *eq = strchr(line, '=');
            if (hash) *hash = 0;
            if (!eq || (hash && eq > hash)) continue;
            *eq = 0;
            fg_set(fg_trim(line), fg_trim(eq + 1));
        }
        fclose(f);
    }
    for (size_t i = 0; i < sizeof keys / sizeof *keys; i++) {
        char env[64];
        snprintf(env, sizeof env, "FAKEGPU_%s", keys[i]);
        for (char *p = env; *p; p++) *p = (char)toupper((unsigned char)*p);
        const char *v = getenv(env);
        if (v) fg_set(keys[i], v);
    }

    if (fg_cfg.nvlinks < 0) fg_cfg.nvlinks = 0;
    if (fg_cfg.nvlinks > 64) fg_cfg.nvlinks = 64;
    if (fg_cfg.sm_count < 1) fg_cfg.sm_count = 1;
    if (fg_cfg.count < 0) fg_cfg.count = 0;
    if (fg_cfg.count > FG_MAX_GPUS) fg_cfg.count = FG_MAX_GPUS;
    const char *p = fg_cfg.cluster_uuid_str;
    for (int i = 0; i < 16 && *p; ) {
        if (*p == '-') { p++; continue; }
        unsigned b;
        if (sscanf(p, "%2x", &b) != 1) break;
        fg_cfg.cluster_uuid[i++] = (unsigned char)b;
        p += 2;
    }
}

static inline void fg_init(void) { pthread_once(&fg_cfg_once, fg_load); }

/* Like the real driver, a process sees only the GPUs whose device node
 * /dev/nvidia<n> exists in its mount namespace: all of them on the tray, the
 * allocated ones in a container (CDI injects one node per GPU). Visible
 * ordinals are dense; identity (UUID, PCI bus, minor number), sideband state
 * and telemetry stay keyed by the physical index. */
static int fg_present[FG_MAX_GPUS], fg_present_count;
static pthread_once_t fg_present_once = PTHREAD_ONCE_INIT;

static void fg_scan(void)
{
    fg_init();
    for (int i = 0; i < fg_cfg.count; i++) {
        char dev[256];
        snprintf(dev, sizeof dev, "%s/nvidia%d", fg_cfg.dev_dir, i);
        if (access(dev, F_OK) == 0) fg_present[fg_present_count++] = i;
    }
}

/* GPUs visible to this process, and the physical index of visible GPU i. */
static inline int fg_count(void) { pthread_once(&fg_present_once, fg_scan); return fg_present_count; }
static inline int fg_phys(int i) { return i >= 0 && i < fg_count() ? fg_present[i] : -1; }
static inline const char *fg_name(void) { fg_init(); return fg_cfg.name; }
static inline unsigned long long fg_mem_bytes(void) { fg_init(); return fg_cfg.mem_mb << 20; }
static inline const unsigned char *fg_cluster_uuid(void) { fg_init(); return fg_cfg.cluster_uuid; }
static inline unsigned fg_clique_id(void) { fg_init(); return fg_cfg.clique_id; }
/* GPU profile; rates in bytes or FLOP per nanosecond (GB/s, GFLOP/s). */
static inline int fg_nvlinks(void) { fg_init(); return fg_cfg.nvlinks; }
static inline int fg_sm_count(void) { fg_init(); return fg_cfg.sm_count; }
static inline double fg_nvlink_bytes_per_ns(void) { fg_init(); return fg_cfg.nvlinks * fg_cfg.nvlink_link_gbs; }
static inline double fg_power_limit_mw(void) { fg_init(); return fg_cfg.power_limit_w * 1000; }
static inline double fg_idle_power_mw(void) { fg_init(); return fg_cfg.idle_power_w * 1000; }
static inline double fg_max_power_mw(void) { fg_init(); return fg_cfg.max_power_w * 1000; }
static inline double fg_tensor_flop_per_ns(void) { fg_init(); return fg_cfg.tensor_tflops * 1000; }
static inline double fg_tf32_flop_per_ns(void) { fg_init(); return fg_cfg.tf32_tflops * 1000; }
static inline double fg_fp8_flop_per_ns(void) { fg_init(); return fg_cfg.fp8_tflops * 1000; }
static inline double fg_fp32_flop_per_ns(void) { fg_init(); return fg_cfg.fp32_tflops * 1000; }
static inline double fg_fp64_flop_per_ns(void) { fg_init(); return fg_cfg.fp64_tflops * 1000; }
static inline double fg_host_link_bytes_per_ns(void) { fg_init(); return fg_cfg.host_link_gbs; }
static inline double fg_hbm_bytes_per_ns(void) { fg_init(); return fg_cfg.hbm_gbs; }
#define FG_NVLINKS ((unsigned)fg_nvlinks())

/* NUMA layout of a GB200 tray (NVIDIA Grace Performance Tuning Guide): one
 * Grace CPU per two GPUs, its node first (0, 1, ...); then per GPU its own
 * memory node followed by 7 MIG instance nodes (2, 10, 18, 26). */
#define FG_MIG_NUMA_NODES 7
static inline int fg_grace_of(int phys) { return phys / 2; }
static inline int fg_gpu_numa_node(int phys)
{
    fg_init();
    if (strcmp(fg_cfg.coherent_gpu_memory, "os")) return -1; /* CDMM: not a NUMA node */
    int graces = (fg_cfg.count + 1) / 2;
    return graces + phys * (1 + FG_MIG_NUMA_NODES);
}
#define FG_SM_COUNT (fg_sm_count())

static inline const char *fg_driver_version(void) { fg_init(); return fg_cfg.driver_version; }
static inline const char *fg_vbios_version(void) { fg_init(); return fg_cfg.vbios_version; }
/* CUDA version as the driver API encodes it: 1000 x major + 10 x minor. */
static inline int fg_cuda_version(void)
{
    fg_init();
    int major = 0, minor = 0;
    sscanf(fg_cfg.cuda_version, "%d.%d", &major, &minor);
    return major * 1000 + minor * 10;
}
static inline double fg_latency_scale(void) { fg_init(); return fg_cfg.latency_scale < 0 ? 0 : fg_cfg.latency_scale; }
static inline size_t fg_copy_max(void) { fg_init(); return (size_t)fg_cfg.copy_max_mb << 20; }

/* Sideband files written by management services (read on every call, they
 * change while processes run):
 *   nvlink-disabled         "<gpu> <link>"  ports disabled by the tray BMC
 *   nvlink-disabled-switch  "<gpu> <link>"  switch ports disabled by the NVSwitch BMC
 *   fabric-clique           "<gpu> <clique>" NVLink partition per GPU (NMX-C) */
static inline int fg_sideband_pair(const char *file, int gpu, long *value)
{
    fg_init();
    if (!fg_cfg.sideband_dir[0]) return 0;
    char path[256], line[64];
    snprintf(path, sizeof path, "%s/%s", fg_cfg.sideband_dir, file);
    FILE *f = fopen(path, "r");
    if (!f) return -1; /* file absent: no information */
    int found = 0, g;
    long v;
    while (!found && fgets(line, sizeof line, f))
        if (line[0] != '#' && sscanf(line, "%d %ld", &g, &v) == 2 && g == gpu && (!value || *value < 0 || v == *value)) {
            found = 1;
            if (value) *value = v;
        }
    fclose(f);
    return found;
}

static inline int fg_link_disabled(int gpu, unsigned link)
{
    long l = link;
    if (fg_sideband_pair("nvlink-disabled", gpu, &l) == 1) return 1;
    l = link;
    return fg_sideband_pair("nvlink-disabled-switch", gpu, &l) == 1;
}

/* NVLink partition (clique) the partition controller assigns a GPU: from
 * NMX-C when it publishes one, else the configured default. 0 means the GPU
 * is in no partition. The GPU takes it at its next reset (nvml_stub.c). */
static inline unsigned fg_partition_clique(int gpu)
{
    long clique = -1;
    if (fg_sideband_pair("fabric-clique", gpu, &clique) == 1) return (unsigned)clique;
    return fg_clique_id();
}

static inline void fg_uuid(int idx, unsigned char out[16])
{
    char host[256] = "localhost";
    fg_init();
    if (fg_cfg.host[0]) snprintf(host, sizeof host, "%s", fg_cfg.host);
    else gethostname(host, sizeof host - 1);
    uint64_t h = 1469598103934665603ULL; /* FNV-1a */
    for (const char *p = host; *p; p++) h = (h ^ (unsigned char)*p) * 1099511628211ULL;
    for (int i = 0; i < 16; i++) {
        h = (h ^ (unsigned)(idx * 16 + i)) * 1099511628211ULL;
        out[i] = (unsigned char)(h >> 56);
    }
}

static inline void fg_uuid_str(int idx, char *buf, size_t len)
{
    unsigned char u[16];
    fg_uuid(idx, u);
    snprintf(buf, len,
             "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
             u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
}

static inline void fg_pci_str(int idx, char *buf, size_t len, int legacy)
{
    snprintf(buf, len, legacy ? "%04X:%02X:00.0" : "%08X:%02X:00.0", fg_pci_domain[idx], FG_PCI_BUS);
}

#endif
