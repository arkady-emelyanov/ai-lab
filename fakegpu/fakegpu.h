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
 *
 * GPU UUIDs are a hash of hostname + index: stable per node, unique across
 * nodes.
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
#define FG_NVLINKS 18           /* NVLink5 links per Blackwell GPU, all to NVSwitch */
#define FG_CC_MAJOR 10
#define FG_CC_MINOR 0
#define FG_SM_COUNT 148
#define FG_PCI_DEVICE_ID 0x294110DE
#define FG_DRIVER_VERSION "580.95.05"
#define FG_CUDA_VERSION 13000   /* 13.0 */

static const unsigned fg_pci_bus[FG_MAX_GPUS] = {0x18, 0x2A, 0x3A, 0x5D, 0x9A, 0xAB, 0xBA, 0xDB};

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
}

static void fg_load(void)
{
    static const char *keys[] = {"count", "name", "mem_mb", "cluster_uuid", "clique_id", "sideband_dir",
                                 "latency_scale", "copy_max_mb", "state_path"};
    fg_cfg.count = 4;
    snprintf(fg_cfg.name, sizeof fg_cfg.name, "NVIDIA GB200");
    fg_cfg.mem_mb = 189471;
    snprintf(fg_cfg.cluster_uuid_str, sizeof fg_cfg.cluster_uuid_str, "00000000-0000-0000-0000-000000000000");
    fg_cfg.clique_id = 1;
    fg_cfg.latency_scale = 1.0;
    fg_cfg.copy_max_mb = 64;
    snprintf(fg_cfg.state_path, sizeof fg_cfg.state_path, "/dev/shm/fakegpu");

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
static inline int fg_count(void) { fg_init(); return fg_cfg.count; }
static inline const char *fg_name(void) { fg_init(); return fg_cfg.name; }
static inline unsigned long long fg_mem_bytes(void) { fg_init(); return fg_cfg.mem_mb << 20; }
static inline const unsigned char *fg_cluster_uuid(void) { fg_init(); return fg_cfg.cluster_uuid; }
static inline unsigned fg_clique_id(void) { fg_init(); return fg_cfg.clique_id; }
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

/* NVLink partition (clique) of a GPU: from NMX-C when it publishes one, else
 * the configured default. 0 means the GPU is in no partition. */
static inline unsigned fg_gpu_clique(int gpu)
{
    long clique = -1;
    if (fg_sideband_pair("fabric-clique", gpu, &clique) == 1) return (unsigned)clique;
    return fg_clique_id();
}

static inline void fg_uuid(int idx, unsigned char out[16])
{
    char host[256] = "localhost";
    gethostname(host, sizeof host - 1);
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
    snprintf(buf, len, legacy ? "%04X:%02X:00.0" : "%08X:%02X:00.0", 0, fg_pci_bus[idx]);
}

#endif
