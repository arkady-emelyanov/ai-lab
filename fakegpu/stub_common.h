/* Helpers shared by the CUDA library stubs (cuBLAS, cuBLASLt, NCCL, cuDNN). */
#ifndef STUB_COMMON_H
#define STUB_COMMON_H

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "fakegpu.h"

#define API __attribute__((visibility("default")))

/* Opaque handles: distinct, non-NULL, big enough that callers peeking at a
 * few fields read zeros. Never freed; destroy calls are no-ops. */
static inline void *stub_handle(void) { return calloc(1, 256); }

/* Appends simulated GPU work to a CUDA stream's timeline in the fake
 * libcuda (fakegpu_enqueue), so library calls take time like kernels do. */
/* libcudart loads libcuda with RTLD_LOCAL, so ask libcuda directly. */
static inline void *stub_libcuda_sym(const char *name)
{
    void *cuda = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (!cuda) cuda = dlopen("libcuda.so.1", RTLD_LAZY);
    return cuda ? dlsym(cuda, name) : NULL;
}

static inline void stub_gpu_work(void *stream, double ns)
{
    static uint64_t (*enqueue)(void *, uint64_t);
    static int looked_up;
    if (!looked_up) {
        enqueue = (uint64_t (*)(void *, uint64_t))stub_libcuda_sym("fakegpu_enqueue");
        looked_up = 1;
    }
    if (enqueue && ns > 0) enqueue(stream, (uint64_t)ns);
}

/* Counts NVLink bytes sent/received by the stream's GPU (fabric telemetry). */
static inline void stub_nvlink_traffic(void *stream, double tx, double rx)
{
    static void (*traffic)(void *, uint64_t, uint64_t);
    static int looked_up;
    if (!looked_up) {
        traffic = (void (*)(void *, uint64_t, uint64_t))stub_libcuda_sym("fakegpu_nvlink_traffic");
        looked_up = 1;
    }
    if (traffic && (tx > 0 || rx > 0)) traffic(stream, (uint64_t)tx, (uint64_t)rx);
}

/* Counts InfiniBand bytes sent/received by the stream's GPU's NIC. */
static inline void stub_ib_traffic(void *stream, double tx, double rx)
{
    static void (*traffic)(void *, uint64_t, uint64_t);
    static int looked_up;
    if (!looked_up) {
        traffic = (void (*)(void *, uint64_t, uint64_t))stub_libcuda_sym("fakegpu_ib_traffic");
        looked_up = 1;
    }
    if (traffic && (tx > 0 || rx > 0)) traffic(stream, (uint64_t)tx, (uint64_t)rx);
}

#define STUB_LAUNCH_NS 5000.0          /* library call overhead */
/* GPU profile from /etc/fakegpu.conf (fakegpu.h). */
#define STUB_TENSOR_FLOP_PER_NS (fg_tensor_flop_per_ns())
#define STUB_TF32_FLOP_PER_NS (fg_tf32_flop_per_ns())
#define STUB_FP8_FLOP_PER_NS (fg_fp8_flop_per_ns())
#define STUB_FP32_FLOP_PER_NS (fg_fp32_flop_per_ns())
#define STUB_FP64_FLOP_PER_NS (fg_fp64_flop_per_ns())
#define STUB_NVLINK_B_PER_NS (fg_nvlink_bytes_per_ns()) /* all of a GPU's NVLinks, per direction */

#endif
