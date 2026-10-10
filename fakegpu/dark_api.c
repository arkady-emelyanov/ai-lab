/*
 * cuGetExportTable: the undocumented "dark API" tables libcudart requires
 * from the driver before cudaGetDeviceCount() & co. will work.
 *
 * Table UUIDs, slot layouts and the integrity-check hash are ported from
 * ZLUDA (https://github.com/vosen/ZLUDA, dark_api/src/lib.rs and
 * zluda/src/impl/driver.rs), Copyright ZLUDA contributors, Apache-2.0.
 *
 * Every table is an array of pointers; most start with their own size in
 * bytes. Slots ZLUDA leaves NULL get a success-returning no-op here.
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "fakegpu.h"

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;

#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_VALUE 1
#define CUDA_ERROR_INVALID_HANDLE 400
#define CUDA_ERROR_NOT_SUPPORTED 801

#define API __attribute__((visibility("default")))
#define SIZE_OF(n) ((void *)(uintptr_t)(sizeof(void *) * (n)))

/* Provided by cuda_stub.c */
CUresult cuDevicePrimaryCtxRetain(CUcontext *ctx, CUdevice dev);
CUresult cuCtxGetCurrent(CUcontext *ctx);
CUresult cuDeviceGetCount(int *n);
CUresult cuDeviceGetUuid_v2(void *uuid, CUdevice dev);
CUresult cuDeviceGetAttribute(int *val, int attr, CUdevice dev);
void *fg_fake_handle(void);

static int debug_enabled(void)
{
    const char *d = getenv("FAKEGPU_DEBUG");
    return d && *d && *d != '0';
}

/* ---- CUDART_INTERFACE {6BD5FB6C-5BF4-E74A-8987-D93912FD9DF9} ----------- */

/* Modules are never executed, so any fatbin "loads" into a dummy handle. */
static CUresult get_module_from_cubin(void **module, const void *fatbin)
{
    (void)fatbin;
    if (!module) return CUDA_ERROR_INVALID_VALUE;
    *module = fg_fake_handle();
    return CUDA_SUCCESS;
}

static CUresult cudart_get_primary_ctx(CUcontext *ctx, CUdevice dev) { return cuDevicePrimaryCtxRetain(ctx, dev); }

static CUresult get_module_from_cubin_ext1(void **module, const void *fatbin, void *a3, void *a4, unsigned a5)
{
    (void)a3; (void)a4; (void)a5;
    return get_module_from_cubin(module, fatbin);
}

static void cudart_fn7(size_t a) { (void)a; }

static CUresult get_module_from_cubin_ext2(const void *fatbin, void **module, void *a3, void *a4, unsigned a5)
{
    (void)a3; (void)a4; (void)a5;
    return get_module_from_cubin(module, fatbin);
}

static CUresult load_compilers(void) { return CUDA_SUCCESS; }

static void *cudart_interface[13] = {
    [0] = SIZE_OF(13),
    [1] = (void *)get_module_from_cubin,
    [2] = (void *)cudart_get_primary_ctx,
    [6] = (void *)get_module_from_cubin_ext1,
    [7] = (void *)cudart_fn7,
    [8] = (void *)get_module_from_cubin_ext2,
    [12] = (void *)load_compilers,
};

/* ---- TOOLS_TLS {42D85A81-23F6-CB47-8298-F6E78A3AECDC} ------------------ */

static void *tools_tls[4] = {[0] = SIZE_OF(4)};

/* ---- TOOLS_RUNTIME_CALLBACK_HOOKS {A094798C-2E74-2E74-93F2-0800200C0A66} */

static uint32_t unknown_buffer1[1024];
static uint32_t unknown_buffer2[14];

static void get_unknown_buffer1(void **p, size_t *n) { *p = unknown_buffer1; *n = 1024; }
static void get_unknown_buffer2(void **p, size_t *n) { *p = unknown_buffer2; *n = 14; }

static void *tools_runtime_callback_hooks[7] = {
    [0] = SIZE_OF(7),
    [2] = (void *)get_unknown_buffer1,
    [6] = (void *)get_unknown_buffer2,
};

/* ---- CONTEXT_LOCAL_STORAGE_INTERFACE_V0301 {C693336E-1121-DF11-A8C3-68F355D89593}
 * This table has no size header: slot 0 is a function. */

struct cls_entry {
    CUcontext ctx;
    void *key, *value;
    struct cls_entry *next;
};
static struct cls_entry *cls_head;
static pthread_mutex_t cls_lock = PTHREAD_MUTEX_INITIALIZER;

static CUcontext ctx_or_current(CUcontext ctx)
{
    if (!ctx) cuCtxGetCurrent(&ctx);
    return ctx;
}

static struct cls_entry **cls_find(CUcontext ctx, void *key)
{
    struct cls_entry **e = &cls_head;
    while (*e && ((*e)->ctx != ctx || (*e)->key != key)) e = &(*e)->next;
    return e;
}

static CUresult cls_put(CUcontext ctx, void *key, void *value, void *dtor)
{
    (void)dtor;
    ctx = ctx_or_current(ctx);
    pthread_mutex_lock(&cls_lock);
    struct cls_entry **e = cls_find(ctx, key);
    if (!*e) {
        *e = calloc(1, sizeof **e);
        (*e)->ctx = ctx;
        (*e)->key = key;
    }
    (*e)->value = value;
    pthread_mutex_unlock(&cls_lock);
    return CUDA_SUCCESS;
}

static CUresult cls_delete(CUcontext ctx, void *key)
{
    ctx = ctx_or_current(ctx);
    pthread_mutex_lock(&cls_lock);
    struct cls_entry **e = cls_find(ctx, key), *dead = *e;
    if (dead) *e = dead->next;
    pthread_mutex_unlock(&cls_lock);
    free(dead);
    return CUDA_SUCCESS;
}

static CUresult cls_get(void **value, CUcontext ctx, void *key)
{
    ctx = ctx_or_current(ctx);
    pthread_mutex_lock(&cls_lock);
    struct cls_entry *e = *cls_find(ctx, key);
    if (e) *value = e->value;
    pthread_mutex_unlock(&cls_lock);
    return e ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}

static void *context_local_storage[4] = {
    [0] = (void *)cls_put,
    [1] = (void *)cls_delete,
    [2] = (void *)cls_get,
};

/* ---- CTX_CREATE_BYPASS {0CA50B8C-1004-929A-89A7-D0DF10E77286} ---------- */

static CUresult ctx_create_bypass(CUcontext *ctx, unsigned flags, CUdevice dev)
{
    (void)ctx; (void)flags; (void)dev;
    return CUDA_ERROR_NOT_SUPPORTED;
}

static void *ctx_create_bypass_table[2] = {[0] = SIZE_OF(2), [1] = (void *)ctx_create_bypass};

/* ---- HEAP_ACCESS {195BCBF4-D67D-024A-ACC5-1D29CEA631AE} ----------------- */

static CUresult heap_alloc(const void **rec, size_t a2, size_t a3)
{
    (void)rec; (void)a2; (void)a3;
    return CUDA_ERROR_NOT_SUPPORTED;
}

static CUresult heap_free(const void *rec, size_t *a2)
{
    (void)rec; (void)a2;
    return CUDA_ERROR_NOT_SUPPORTED;
}

static void *heap_access[3] = {[0] = SIZE_OF(3), [1] = (void *)heap_alloc, [2] = (void *)heap_free};

/* ---- DEVICE_EXTENDED_RT {B10541E1-F7C7-C74A-9F64-F223BE99F1E2} --------- */

static CUresult device_get_attribute_ext(CUdevice dev, unsigned attr, int unknown, size_t result[2])
{
    (void)dev; (void)attr; (void)unknown; (void)result;
    return CUDA_ERROR_NOT_SUPPORTED;
}

static CUresult device_get_something(unsigned char *result, CUdevice dev)
{
    (void)dev;
    if (result) *result = 0;
    return CUDA_SUCCESS;
}

static void *device_extended_rt[26] = {
    [0] = SIZE_OF(26),
    [5] = (void *)device_get_attribute_ext,
    [13] = (void *)device_get_something,
};

/* ---- CONTEXT_CHECKS {263E8860-7CD2-6143-92F6-BBD5006DFA7E} -------------- */

static CUresult context_check(CUcontext ctx, uint32_t *r1, const void **r2)
{
    (void)ctx; (void)r2;
    if (r1) *r1 = 0;
    return CUDA_SUCCESS;
}

static uint32_t check_fn3(void) { return 0; }

static void *context_checks[4] = {[0] = SIZE_OF(4), [2] = (void *)context_check, [3] = (void *)check_fn3};

/* ---- INTEGRITY_CHECK {D4082055-BDE6-704B-8D34-BA123C66E1F2} -------------
 * libcudart verifies the driver by asking for a hash over process/thread
 * ids, table addresses, the time and every device's UUID and PCI location. */

static const uint8_t mixing_table[256] = {
    0x29, 0x2E, 0x43, 0xC9, 0xA2, 0xD8, 0x7C, 0x01, 0x3D, 0x36, 0x54, 0xA1, 0xEC, 0xF0, 0x06, 0x13,
    0x62, 0xA7, 0x05, 0xF3, 0xC0, 0xC7, 0x73, 0x8C, 0x98, 0x93, 0x2B, 0xD9, 0xBC, 0x4C, 0x82, 0xCA,
    0x1E, 0x9B, 0x57, 0x3C, 0xFD, 0xD4, 0xE0, 0x16, 0x67, 0x42, 0x6F, 0x18, 0x8A, 0x17, 0xE5, 0x12,
    0xBE, 0x4E, 0xC4, 0xD6, 0xDA, 0x9E, 0xDE, 0x49, 0xA0, 0xFB, 0xF5, 0x8E, 0xBB, 0x2F, 0xEE, 0x7A,
    0xA9, 0x68, 0x79, 0x91, 0x15, 0xB2, 0x07, 0x3F, 0x94, 0xC2, 0x10, 0x89, 0x0B, 0x22, 0x5F, 0x21,
    0x80, 0x7F, 0x5D, 0x9A, 0x5A, 0x90, 0x32, 0x27, 0x35, 0x3E, 0xCC, 0xE7, 0xBF, 0xF7, 0x97, 0x03,
    0xFF, 0x19, 0x30, 0xB3, 0x48, 0xA5, 0xB5, 0xD1, 0xD7, 0x5E, 0x92, 0x2A, 0xAC, 0x56, 0xAA, 0xC6,
    0x4F, 0xB8, 0x38, 0xD2, 0x96, 0xA4, 0x7D, 0xB6, 0x76, 0xFC, 0x6B, 0xE2, 0x9C, 0x74, 0x04, 0xF1,
    0x45, 0x9D, 0x70, 0x59, 0x64, 0x71, 0x87, 0x20, 0x86, 0x5B, 0xCF, 0x65, 0xE6, 0x2D, 0xA8, 0x02,
    0x1B, 0x60, 0x25, 0xAD, 0xAE, 0xB0, 0xB9, 0xF6, 0x1C, 0x46, 0x61, 0x69, 0x34, 0x40, 0x7E, 0x0F,
    0x55, 0x47, 0xA3, 0x23, 0xDD, 0x51, 0xAF, 0x3A, 0xC3, 0x5C, 0xF9, 0xCE, 0xBA, 0xC5, 0xEA, 0x26,
    0x2C, 0x53, 0x0D, 0x6E, 0x85, 0x28, 0x84, 0x09, 0xD3, 0xDF, 0xCD, 0xF4, 0x41, 0x81, 0x4D, 0x52,
    0x6A, 0xDC, 0x37, 0xC8, 0x6C, 0xC1, 0xAB, 0xFA, 0x24, 0xE1, 0x7B, 0x08, 0x0C, 0xBD, 0xB1, 0x4A,
    0x78, 0x88, 0x95, 0x8B, 0xE3, 0x63, 0xE8, 0x6D, 0xE9, 0xCB, 0xD5, 0xFE, 0x3B, 0x00, 0x1D, 0x39,
    0xF2, 0xEF, 0xB7, 0x0E, 0x66, 0x58, 0xD0, 0xE4, 0xA6, 0x77, 0x72, 0xF8, 0xEB, 0x75, 0x4B, 0x0A,
    0x31, 0x44, 0x50, 0xB4, 0x8F, 0xED, 0x1F, 0x1A, 0xDB, 0x99, 0x8D, 0x33, 0x9F, 0x11, 0x83, 0x14,
};

static void ic_single_pass(uint8_t a[66], uint8_t b)
{
    uint8_t t1 = a[0x40];
    a[t1 + 0x10] = b;
    uint8_t t3 = (uint8_t)((t1 + 1) & 0xf);
    a[t1 + 0x20] = a[t1] ^ b;
    uint8_t t4 = mixing_table[b ^ a[0x41]];
    uint8_t old = a[t1 + 0x30];
    a[t1 + 0x30] = t4 ^ old;
    a[0x41] = t4 ^ old;
    a[0x40] = t3;
    if (t3 != 0) return;

    uint8_t x = 0x29, round = 0;
    for (;;) {
        x ^= a[0];
        a[0] = x;
        for (int i = 1; i < 0x30; i++) {
            x = a[i] ^ mixing_table[x];
            a[i] = x;
        }
        x = (uint8_t)(x + round);
        round++;
        if (round == 0x12) break;
        x = mixing_table[x];
    }
}

static void ic_hash(uint8_t acc[66], const void *data, size_t len, uint8_t xor_mask)
{
    const uint8_t *p = data;
    for (size_t i = 0; i < len; i++) ic_single_pass(acc, p[i] ^ xor_mask);
}

static void ic_finish(uint8_t acc[66], uint64_t out[2])
{
    uint8_t pad = (uint8_t)(16 - acc[64]);
    for (uint8_t i = 0; i < pad; i++) ic_single_pass(acc, pad);
    for (int i = 0x30; i < 0x40; i++) ic_single_pass(acc, acc[i]);
    memcpy(out, acc, 16);
}

struct ic_pass3 {
    uint32_t driver_version, version, pid, tid;
    const void *cudart_table, *integrity_table, *fn_address;
    uint64_t unix_seconds;
};

struct ic_device {
    unsigned char uuid[16];
    int32_t pci_domain, pci_bus, pci_device;
};

static CUresult integrity_check(uint32_t version, uint64_t unix_seconds, uint64_t result[2]);
static void *integrity_check_table[3] = {[0] = SIZE_OF(3), [1] = (void *)integrity_check};

static CUresult integrity_check(uint32_t version, uint64_t unix_seconds, uint64_t result[2])
{
    switch (version % 10) {
    case 0: result[0] = 0x3341181C03CB675CULL; result[1] = 0x8ED383AA1F4CD1E8ULL; return CUDA_SUCCESS;
    case 1: result[0] = 0x1841181C03CB675CULL; result[1] = 0x8ED383AA1F4CD1E8ULL; return CUDA_SUCCESS;
    }
    static const uint8_t pass1[16] = {0x14, 0x6A, 0xDD, 0xAE, 0x53, 0xA9, 0xA7, 0x52,
                                      0xAA, 0x08, 0x41, 0x36, 0x0B, 0xF5, 0x5A, 0x9F};
    uint8_t acc[66] = {0};
    ic_hash(acc, pass1, sizeof pass1, 0x36);

    struct ic_pass3 in = {
        .driver_version = (uint32_t)fg_cuda_version(),
        .version = version,
        .pid = (uint32_t)getpid(),
        .tid = (uint32_t)pthread_self(),
        .cudart_table = cudart_interface,
        .integrity_table = integrity_check_table,
        .fn_address = integrity_check_table[1],
        .unix_seconds = unix_seconds,
    };
    ic_hash(acc, &in, sizeof in, 0);

    int n = 0;
    cuDeviceGetCount(&n);
    for (int d = 0; d < n; d++) {
        struct ic_device dev;
        memset(&dev, 0, sizeof dev);
        cuDeviceGetUuid_v2(dev.uuid, d);
        cuDeviceGetAttribute(&dev.pci_domain, 50, d);
        cuDeviceGetAttribute(&dev.pci_bus, 33, d);
        cuDeviceGetAttribute(&dev.pci_device, 34, d);
        ic_hash(acc, &dev, sizeof dev, 0);
    }

    uint64_t inner[2];
    ic_finish(acc, inner);
    memset(acc, 0, 16);
    memset(acc + 48, 0, 18);
    ic_hash(acc, pass1, sizeof pass1, 0x5c);
    ic_hash(acc, inner, sizeof inner, 0);
    ic_finish(acc, result);
    return CUDA_SUCCESS;
}

/* Required by CUDA 13.4 runtimes (cuda-bindings 13.4's static cudart):
 * without it cudaGetDeviceCount() and every later call fail with
 * cudaErrorInvalidValue. Not in ZLUDA. Driver 595.99.02 serves 94 slots;
 * the runtime only checks the table exists (no slot was called through
 * device, memory, copy, stream, event, IPC and reset calls), so all are
 * no-ops. */
static void *cudart13_table[95] = {[0] = SIZE_OF(95)};

/* ---- lookup ------------------------------------------------------------ */

static const struct {
    unsigned char uuid[16];
    void **table;
} tables[] = {
    {{0x6B, 0xD5, 0xFB, 0x6C, 0x5B, 0xF4, 0xE7, 0x4A, 0x89, 0x87, 0xD9, 0x39, 0x12, 0xFD, 0x9D, 0xF9}, cudart_interface},
    {{0x42, 0xD8, 0x5A, 0x81, 0x23, 0xF6, 0xCB, 0x47, 0x82, 0x98, 0xF6, 0xE7, 0x8A, 0x3A, 0xEC, 0xDC}, tools_tls},
    {{0xA0, 0x94, 0x79, 0x8C, 0x2E, 0x74, 0x2E, 0x74, 0x93, 0xF2, 0x08, 0x00, 0x20, 0x0C, 0x0A, 0x66}, tools_runtime_callback_hooks},
    {{0xC6, 0x93, 0x33, 0x6E, 0x11, 0x21, 0xDF, 0x11, 0xA8, 0xC3, 0x68, 0xF3, 0x55, 0xD8, 0x95, 0x93}, context_local_storage},
    {{0x0C, 0xA5, 0x0B, 0x8C, 0x10, 0x04, 0x92, 0x9A, 0x89, 0xA7, 0xD0, 0xDF, 0x10, 0xE7, 0x72, 0x86}, ctx_create_bypass_table},
    {{0x19, 0x5B, 0xCB, 0xF4, 0xD6, 0x7D, 0x02, 0x4A, 0xAC, 0xC5, 0x1D, 0x29, 0xCE, 0xA6, 0x31, 0xAE}, heap_access},
    {{0xB1, 0x05, 0x41, 0xE1, 0xF7, 0xC7, 0xC7, 0x4A, 0x9F, 0x64, 0xF2, 0x23, 0xBE, 0x99, 0xF1, 0xE2}, device_extended_rt},
    {{0xD4, 0x08, 0x20, 0x55, 0xBD, 0xE6, 0x70, 0x4B, 0x8D, 0x34, 0xBA, 0x12, 0x3C, 0x66, 0xE1, 0xF2}, integrity_check_table},
    {{0x21, 0x31, 0x8C, 0x60, 0x97, 0x14, 0x32, 0x48, 0x8C, 0xA6, 0x41, 0xFF, 0x73, 0x24, 0xC8, 0xF2}, cudart13_table},
    {{0x26, 0x3E, 0x88, 0x60, 0x7C, 0xD2, 0x61, 0x43, 0x92, 0xF6, 0xBB, 0xD5, 0x00, 0x6D, 0xFA, 0x7E}, context_checks},
};

static CUresult noop(void) { return CUDA_SUCCESS; }

static size_t table_len(void **t)
{
    if (t == context_local_storage) return 4; /* no size header */
    return (uintptr_t)t[0] / sizeof(void *);
}

static void fill_empty_slots(void)
{
    for (size_t i = 0; i < sizeof tables / sizeof *tables; i++)
        for (size_t j = 1; j < table_len(tables[i].table); j++)
            if (!tables[i].table[j]) tables[i].table[j] = (void *)noop;
}

API CUresult cuGetExportTable(const void **table, const unsigned char id[16])
{
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, fill_empty_slots);
    if (!table || !id) return CUDA_ERROR_INVALID_VALUE;
    if (debug_enabled()) {
        fprintf(stderr, "[fakegpu] export table ");
        for (int i = 0; i < 16; i++) fprintf(stderr, "%02x", id[i]);
        fputc('\n', stderr);
    }
    for (size_t i = 0; i < sizeof tables / sizeof *tables; i++) {
        if (!memcmp(tables[i].uuid, id, 16)) {
            *table = tables[i].table;
            return CUDA_SUCCESS;
        }
    }
    if (debug_enabled()) {
        fprintf(stderr, "[fakegpu] unknown export table ");
        for (int i = 0; i < 16; i++) fprintf(stderr, "%02x", id[i]);
        fputc('\n', stderr);
    }
    *table = NULL;
    return CUDA_ERROR_INVALID_VALUE;
}
