/*
 * MIG (Multi-Instance GPU) for the fake driver, as on GB200: the GPU's 7
 * compute slices and 8 memory slices split into GPU instances (GI), each
 * split into compute instances (CI); a CI is a MIG device, with a MIG- UUID,
 * its GI's memory and its own SMs. NVML (nvml_stub.c) creates and lists
 * them, CUDA (cuda_stub.c) shows a MIG-enabled GPU as its MIG devices.
 *
 * Profiles, placements, SMs, memory and engines are GB200's as nvidia-smi
 * mig -lgip prints them (NVIDIA/mig-parted#227); GI IDs are the hardware's
 * slots over the memory slices (a tree: the whole GPU is 0, halves 1-2,
 * quarters 3-6, eighths 7-14: the first 1g nvidia-smi creates on GB200 is
 * GI 13, at slot 6). An instance created without a placement takes the
 * highest free one. Sub-CIs' SM counts are not published for GB200: they
 * are the GI's split by slices.
 *
 * MIG state is driver state: the tray's shared state (occupancy.h), seen by
 * every process, changed under a lock. As on Hopper and later, MIG mode
 * switches at once on an idle GPU; on a GPU a process uses, the switch is
 * pending until the GPU's next reset (nvidia-smi -r).
 */
#ifndef FAKEGPU_MIG_H
#define FAKEGPU_MIG_H

#include <stddef.h>
#include "occupancy.h"

/* NVML return codes the MIG calls use. */
enum { MIG_OK = 0, MIG_INVALID = 2, MIG_NOT_SUPPORTED = 3, MIG_NOT_FOUND = 6, MIG_INSUFFICIENT_SIZE = 7,
       MIG_IN_USE = 19, MIG_INSUFFICIENT_RESOURCES = 23 };

#define MIG_PROFILE_COUNT 18     /* NVML_GPU_INSTANCE_PROFILE_COUNT */
#define MIG_CI_PROFILE_COUNT 9   /* NVML_COMPUTE_INSTANCE_PROFILE_COUNT */
#define MIG_COMPUTE_SLICES 7
#define MIG_MEMORY_SLICES 8
#define MIG_FULL_SMS 152         /* the 7g profile's: what a MIG device's SMs are relative to */

struct mig_ci_profile {
    int index;          /* NVML_COMPUTE_INSTANCE_PROFILE_* (= its id) */
    unsigned slices, instances, sms;
};

struct mig_profile {
    int index;          /* NVML_GPU_INSTANCE_PROFILE_* */
    unsigned id;        /* profile id (nvidia-smi -lgip) */
    const char *name;
    unsigned slices, mem_slices, instances, sms, ce, dec, enc, jpeg, ofa;
    unsigned long long mib;
    unsigned nstarts, starts[8];
    unsigned nci;
    struct mig_ci_profile ci[5];
};

static const struct mig_profile mig_profiles[] = {
    {0, 19, "1g.23gb", 1, 1, 7, 18, 2, 1, 0, 1, 0, 22528, 7, {0, 1, 2, 3, 4, 5, 6}, 1, {{0, 1, 1, 18}}},
    {7, 20, "1g.23gb+me", 1, 1, 1, 18, 2, 1, 0, 1, 1, 22528, 7, {0, 1, 2, 3, 4, 5, 6}, 1, {{0, 1, 1, 18}}},
    {9, 15, "1g.47gb", 1, 2, 4, 32, 2, 1, 0, 1, 0, 46592, 4, {0, 2, 4, 6}, 2, {{0, 1, 1, 18}, {7, 1, 1, 32}}},
    {1, 14, "2g.47gb", 2, 2, 3, 38, 4, 2, 0, 2, 0, 46592, 3, {0, 2, 4}, 2, {{0, 1, 2, 19}, {1, 2, 1, 38}}},
    {2, 9, "3g.93gb", 3, 4, 2, 72, 6, 3, 0, 3, 0, 93696, 2, {0, 4}, 3, {{0, 1, 3, 24}, {1, 2, 1, 48}, {2, 3, 1, 72}}},
    {3, 5, "4g.93gb", 4, 4, 1, 76, 8, 4, 0, 4, 1, 93696, 1, {0}, 3, {{0, 1, 4, 19}, {1, 2, 2, 38}, {3, 4, 1, 76}}},
    {4, 0, "7g.186gb", 7, 8, 1, 152, 16, 7, 0, 7, 2, 188416, 1, {0}, 5,
     {{0, 1, 7, 21}, {1, 2, 3, 43}, {2, 3, 2, 65}, {3, 4, 1, 86}, {4, 7, 1, 152}}},
};
#define MIG_NPROFILES ((int)(sizeof mig_profiles / sizeof *mig_profiles))

static inline const struct mig_profile *mig_profile_by_index(unsigned index)
{
    for (int i = 0; i < MIG_NPROFILES; i++)
        if ((unsigned)mig_profiles[i].index == index) return &mig_profiles[i];
    return NULL;
}

static inline const struct mig_profile *mig_profile_by_id(unsigned id)
{
    for (int i = 0; i < MIG_NPROFILES; i++)
        if (mig_profiles[i].id == id) return &mig_profiles[i];
    return NULL;
}

static inline const struct mig_ci_profile *mig_ci_profile(const struct mig_profile *p, unsigned index)
{
    for (unsigned i = 0; i < p->nci; i++)
        if ((unsigned)p->ci[i].index == index) return &p->ci[i];
    return NULL;
}

/* The full CI of a GI profile: the most SMs (1g.47gb: CI 7, 32 SMs). */
static inline const struct mig_ci_profile *mig_full_ci(const struct mig_profile *p)
{
    const struct mig_ci_profile *best = &p->ci[0];
    for (unsigned i = 1; i < p->nci; i++)
        if (p->ci[i].sms > best->sms) best = &p->ci[i];
    return best;
}

/* CI name as nvidia-smi and NVML give it: the GI's when it covers the GI,
 * else <c>c.<gi> (MIG 1c.3g.93gb). */
static inline void mig_ci_name(const struct mig_profile *p, const struct mig_ci_profile *c, char *buf, size_t len)
{
    if (c->slices == p->slices) snprintf(buf, len, "%s", p->name);
    else snprintf(buf, len, "%uc.%s", c->slices, p->name);
}

/* GI id of a placement: its slot in the tree over the memory slices. */
static inline int mig_slot(unsigned start, unsigned size)
{
    switch (size) {
    case 8: return 0;
    case 4: return 1 + (int)start / 4;
    case 2: return 3 + (int)start / 2;
    default: return 7 + (int)start;
    }
}

/* ---- state --------------------------------------------------------------------- */

static inline struct fg_gpu_mig *mig_state(int gpu)
{
    struct fg_occupancy *o = fg_occupancy();
    return o && gpu >= 0 && gpu < FG_MAX_GPUS ? &o->mig[gpu] : NULL;
}

static pthread_mutex_t mig_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Changes to a GPU's MIG state: under the process's mutex (threads share the
 * file lock) and an OFD lock on the GPU's MIG state in the shared file. */
static inline struct fg_gpu_mig *mig_lock(int gpu)
{
    struct fg_gpu_mig *m = mig_state(gpu);
    if (!m) return NULL;
    pthread_mutex_lock(&mig_mutex);
    struct flock fl = {.l_type = F_WRLCK, .l_whence = SEEK_SET,
                       .l_start = (off_t)((char *)m - (char *)fg_occupancy()), .l_len = 1};
    while (fcntl(fg_occ_fd, F_OFD_SETLKW, &fl) != 0 && errno == EINTR) {}
    return m;
}

static inline void mig_unlock(struct fg_gpu_mig *m)
{
    struct flock fl = {.l_type = F_UNLCK, .l_whence = SEEK_SET,
                       .l_start = (off_t)((char *)m - (char *)fg_occupancy()), .l_len = 1};
    fcntl(fg_occ_fd, F_OFD_SETLK, &fl);
    pthread_mutex_unlock(&mig_mutex);
}

static inline int mig_enabled(int gpu)
{
    struct fg_gpu_mig *m = mig_state(gpu);
    return m && __atomic_load_n(&m->current, __ATOMIC_ACQUIRE) == 1;
}

/* Whether a live process other than this one uses the GPU. */
static inline int mig_gpu_busy(int gpu)
{
    struct fg_occupancy *o = fg_occupancy();
    if (!o) return 0;
    fg_proc_cleanup(0);
    for (int i = 0; i < FG_MAX_PROCS; i++)
        if (o->proc[i].gpu == gpu && fg_slot_alive(o, i) && !fg_slot_mine(o, i)) return 1;
    return 0;
}

static inline int mig_has_gi(const struct fg_gpu_mig *m)
{
    for (int g = 0; g < FG_MIG_SLOTS; g++)
        if (m->gi[g].used) return 1;
    return 0;
}

/* nvmlDeviceSetMigMode: returns the call's result, *act the activation. */
static inline int mig_set_mode(int gpu, unsigned mode, int *act)
{
    if (mode > 1) return MIG_INVALID;
    struct fg_gpu_mig *m = mig_lock(gpu);
    if (!m) return MIG_NOT_SUPPORTED;
    int rc = MIG_OK;
    if (mode == 0 && mig_has_gi(m)) {
        rc = MIG_IN_USE;
    } else if (mode == m->current) {
        m->pending = mode;
        *act = MIG_OK;
    } else if (mig_gpu_busy(gpu)) {
        m->pending = mode;
        *act = MIG_IN_USE;
    } else {
        m->pending = m->current = mode;
        *act = MIG_OK;
    }
    mig_unlock(m);
    return rc;
}

/* A GPU reset (nvidia-smi -r) applies a pending MIG mode. */
static inline void mig_reset(int gpu)
{
    struct fg_gpu_mig *m = mig_lock(gpu);
    if (!m) return;
    if (m->pending != m->current) {
        m->current = m->pending;
        if (!m->current) memset(m->gi, 0, sizeof m->gi);
    }
    mig_unlock(m);
}

static inline unsigned mig_compute_used(const struct fg_gpu_mig *m)
{
    unsigned n = 0;
    for (int g = 0; g < FG_MIG_SLOTS; g++)
        if (m->gi[g].used) n += mig_profiles[m->gi[g].profile].slices;
    return n;
}

static inline int mig_overlaps(const struct fg_gpu_mig *m, unsigned start, unsigned size)
{
    for (int g = 0; g < FG_MIG_SLOTS; g++) {
        const struct fg_mig_gi *x = &m->gi[g];
        if (x->used && start < (unsigned)x->start + x->size && x->start < start + size) return 1;
    }
    return 0;
}

static inline unsigned mig_count_of(const struct fg_gpu_mig *m, const struct mig_profile *p)
{
    unsigned n = 0;
    for (int g = 0; g < FG_MIG_SLOTS; g++)
        if (m->gi[g].used && &mig_profiles[m->gi[g].profile] == p) n++;
    return n;
}

/* Whether a GI of profile p fits at start (-1: the highest free placement);
 * returns the start, or -1. */
static inline int mig_gi_fit(const struct fg_gpu_mig *m, const struct mig_profile *p, int start)
{
    if (mig_count_of(m, p) >= p->instances) return -1;
    if (mig_compute_used(m) + p->slices > MIG_COMPUTE_SLICES) return -1;
    for (int i = (int)p->nstarts - 1; i >= 0; i--) {
        unsigned s = p->starts[i];
        if (start >= 0 && (unsigned)start != s) continue;
        if (!mig_overlaps(m, s, p->mem_slices)) return (int)s;
    }
    return -1;
}

static inline int mig_valid_start(const struct mig_profile *p, unsigned start, unsigned size)
{
    if (size != p->mem_slices) return 0;
    for (unsigned i = 0; i < p->nstarts; i++)
        if (p->starts[i] == start) return 1;
    return 0;
}

/* nvmlDeviceCreateGpuInstance(WithPlacement): start -1 for any placement. */
static inline int mig_create_gi(int gpu, const struct mig_profile *p, int start, int *gi)
{
    struct fg_gpu_mig *m = mig_lock(gpu);
    if (!m) return MIG_NOT_SUPPORTED;
    int rc = MIG_OK;
    if (m->current != 1) {
        rc = MIG_NOT_SUPPORTED;
    } else {
        int s = mig_gi_fit(m, p, start);
        if (s < 0) {
            rc = MIG_INSUFFICIENT_RESOURCES;
        } else {
            int slot = mig_slot((unsigned)s, p->mem_slices);
            struct fg_mig_gi *x = &m->gi[slot];
            memset(x, 0, sizeof *x);
            x->profile = (uint8_t)(p - mig_profiles);
            x->start = (uint8_t)s;
            x->size = (uint8_t)p->mem_slices;
            x->serial = ++m->serial;
            __atomic_store_n(&x->used, 1, __ATOMIC_RELEASE);
            *gi = slot;
        }
    }
    mig_unlock(m);
    return rc;
}

/* Whether GI gi of a MIG-enabled GPU exists; its profile in *p. */
static inline int mig_gi_exists(int gpu, int gi, const struct mig_profile **p)
{
    struct fg_gpu_mig *m = mig_state(gpu);
    if (!m || gi < 0 || gi >= FG_MIG_SLOTS || !__atomic_load_n(&m->gi[gi].used, __ATOMIC_ACQUIRE)) return 0;
    if (p) *p = &mig_profiles[m->gi[gi].profile];
    return 1;
}

static inline int mig_destroy_gi(int gpu, int gi)
{
    struct fg_gpu_mig *m = mig_lock(gpu);
    if (!m) return MIG_NOT_SUPPORTED;
    int rc = MIG_OK;
    if (gi < 0 || gi >= FG_MIG_SLOTS || !m->gi[gi].used) {
        rc = MIG_INVALID;
    } else {
        for (int c = 0; c < FG_MIG_CIS; c++)
            if (m->gi[gi].ci[c].used) rc = MIG_IN_USE;
        if (rc == MIG_OK) memset(&m->gi[gi], 0, sizeof m->gi[gi]);
    }
    mig_unlock(m);
    return rc;
}

/* Where a CI of profile c fits in GI x (lowest free start), or -1. */
static inline int mig_ci_fit(const struct fg_mig_gi *x, const struct mig_ci_profile *c)
{
    unsigned n = 0;
    for (int i = 0; i < FG_MIG_CIS; i++)
        if (x->ci[i].used && x->ci[i].profile == c->index) n++;
    if (n >= c->instances) return -1;
    for (unsigned k = 0; k < c->instances; k++) {
        unsigned s = k * c->slices;
        int free = 1;
        for (int i = 0; i < FG_MIG_CIS; i++) {
            const struct fg_mig_ci *y = &x->ci[i];
            if (y->used && s < (unsigned)y->start + y->size && y->start < s + c->slices) free = 0;
        }
        if (free) return (int)s;
    }
    return -1;
}

static inline int mig_create_ci(int gpu, int gi, unsigned ci_index, int *ci)
{
    struct fg_gpu_mig *m = mig_lock(gpu);
    if (!m) return MIG_NOT_SUPPORTED;
    int rc = MIG_OK;
    struct fg_mig_gi *x = gi >= 0 && gi < FG_MIG_SLOTS ? &m->gi[gi] : NULL;
    const struct mig_ci_profile *c = x && x->used ? mig_ci_profile(&mig_profiles[x->profile], ci_index) : NULL;
    if (!x || !x->used) {
        rc = MIG_INVALID;
    } else if (!c) {
        rc = MIG_NOT_SUPPORTED;
    } else {
        int s = mig_ci_fit(x, c), id = -1;
        for (int i = 0; i < FG_MIG_CIS && id < 0; i++)
            if (!x->ci[i].used) id = i;
        if (s < 0 || id < 0) {
            rc = MIG_INSUFFICIENT_RESOURCES;
        } else {
            x->ci[id] = (struct fg_mig_ci){.used = 1, .profile = (uint8_t)c->index, .start = (uint8_t)s,
                                           .size = (uint8_t)c->slices};
            *ci = id;
        }
    }
    mig_unlock(m);
    return rc;
}

static inline int mig_ci_exists(int gpu, int gi, int ci, const struct mig_profile **p, const struct mig_ci_profile **c)
{
    const struct mig_profile *pp;
    if (!mig_gi_exists(gpu, gi, &pp) || ci < 0 || ci >= FG_MIG_CIS) return 0;
    const struct fg_mig_ci *y = &mig_state(gpu)->gi[gi].ci[ci];
    if (!__atomic_load_n(&y->used, __ATOMIC_ACQUIRE)) return 0;
    if (p) *p = pp;
    if (c) *c = mig_ci_profile(pp, y->profile);
    return 1;
}

static inline int mig_destroy_ci(int gpu, int gi, int ci)
{
    struct fg_gpu_mig *m = mig_lock(gpu);
    if (!m) return MIG_NOT_SUPPORTED;
    int rc = MIG_INVALID;
    if (gi >= 0 && gi < FG_MIG_SLOTS && ci >= 0 && ci < FG_MIG_CIS && m->gi[gi].used && m->gi[gi].ci[ci].used) {
        memset(&m->gi[gi].ci[ci], 0, sizeof m->gi[gi].ci[ci]);
        rc = MIG_OK;
    }
    mig_unlock(m);
    return rc;
}

/* The GPU's MIG devices (one per CI), ordered by GI id, then CI id. */
struct mig_dev { int gi, ci; };

static inline int mig_devices(int gpu, struct mig_dev *out)
{
    int n = 0;
    struct fg_gpu_mig *m = mig_state(gpu);
    if (!m || !mig_enabled(gpu)) return 0;
    for (int g = 0; g < FG_MIG_SLOTS; g++)
        for (int c = 0; c < FG_MIG_CIS && m->gi[g].used; c++)
            if (m->gi[g].ci[c].used) out[n++] = (struct mig_dev){g, c};
    return n;
}

/* A MIG device's UUID: a version-5 UUID from the GPU's identity, its GI and
 * CI, and when the GI was created (a new instance gets a new UUID). */
static inline void mig_uuid(int gpu, int gi, int ci, unsigned char out[16])
{
    struct fg_gpu_mig *m = mig_state(gpu);
    unsigned char base[16];
    fg_uuid(gpu, base);
    uint64_t h = 1469598103934665603ULL; /* FNV-1a */
    uint32_t serial = m ? m->gi[gi].serial : 0;
    unsigned char key[24];
    memcpy(key, base, 16);
    key[16] = (unsigned char)gi;
    key[17] = (unsigned char)ci;
    memcpy(key + 18, &serial, 4);
    key[22] = 'M';
    key[23] = 'G';
    for (int i = 0; i < 16; i++) {
        for (int k = 0; k < 24; k++) h = (h ^ key[k]) * 1099511628211ULL;
        h = (h ^ (unsigned)i) * 1099511628211ULL;
        out[i] = (unsigned char)(h >> 56);
    }
    out[6] = (unsigned char)((out[6] & 0x0f) | 0x50); /* version 5 */
    out[8] = (unsigned char)((out[8] & 0x3f) | 0x80); /* RFC 4122 variant */
}

static inline void mig_uuid_str(int gpu, int gi, int ci, char *buf, size_t len)
{
    unsigned char u[16];
    mig_uuid(gpu, gi, ci, u);
    snprintf(buf, len,
             "MIG-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7],
             u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
}

#endif
