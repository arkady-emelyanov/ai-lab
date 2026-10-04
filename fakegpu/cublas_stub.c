/*
 * Fake cuBLAS (libcublas.so.13 / .so.12) and cuBLASLt (libcublasLt.so.13 /
 * .so.12). Every exported symbol returns CUBLAS_STATUS_SUCCESS (generated
 * from symbols/cublas*.syms); the functions below hand out handles, report
 * versions, make the Lt heuristic query find an algorithm and charge GEMMs
 * their simulated time (2*m*n*k FLOPs) on the handle's stream.
 *
 * Built once per library: -DSTUB_CUBLASLT selects the Lt half.
 */
#include "stub_common.h"

typedef int cublasStatus_t;
#define CUBLAS_STATUS_SUCCESS 0
#define CUBLAS_STATUS_INVALID_VALUE 7

/* Defaults match the CUDA 13 wheels; the .so.12 builds override them. */
#ifndef CUBLAS_VER_MAJOR
#define CUBLAS_VER_MAJOR 13
#define CUBLAS_VER_MINOR 1
#define CUBLAS_VER_PATCH 1
#endif
#ifndef CUDART_VERSION
#define CUDART_VERSION 13000
#endif
#define CUBLAS_VERSION (CUBLAS_VER_MAJOR * 10000 + CUBLAS_VER_MINOR * 100 + CUBLAS_VER_PATCH)

static const char *status_name(cublasStatus_t s)
{
    switch (s) {
    case 0: return "CUBLAS_STATUS_SUCCESS";
    case 1: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case 3: return "CUBLAS_STATUS_ALLOC_FAILED";
    case 7: return "CUBLAS_STATUS_INVALID_VALUE";
    case 8: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case 11: return "CUBLAS_STATUS_MAPPING_ERROR";
    case 13: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case 14: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case 15: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case 16: return "CUBLAS_STATUS_LICENSE_ERROR";
    default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

static cublasStatus_t new_handle(void **h)
{
    if (!h) return CUBLAS_STATUS_INVALID_VALUE;
    *h = stub_handle();
    return CUBLAS_STATUS_SUCCESS;
}

static void gemm_work(void *stream, double m, double n, double k, double batch, double flop_per_ns)
{
    stub_gpu_work(stream, STUB_LAUNCH_NS + 2.0 * m * n * k * batch / flop_per_ns);
}

#define CUDA_R_32F 0
#define CUDA_R_64F 1
#define CUBLAS_COMPUTE_32F 68
#define CUBLAS_COMPUTE_64F 70

static double rate_for(int atype, int compute)
{
    if (atype == CUDA_R_64F || compute == CUBLAS_COMPUTE_64F) return STUB_FP64_FLOP_PER_NS;
    if (atype == CUDA_R_32F && compute == CUBLAS_COMPUTE_32F) return STUB_FP32_FLOP_PER_NS;
    return STUB_TENSOR_FLOP_PER_NS;
}

#ifndef STUB_CUBLASLT

/* ---- cuBLAS ------------------------------------------------------------ */

API const char *cublasGetStatusName(cublasStatus_t s) { return status_name(s); }
API const char *cublasGetStatusString(cublasStatus_t s) { return s ? "the requested functionality is not supported" : "success"; }
API size_t cublasGetCudartVersion(void) { return CUDART_VERSION; }
API void cublasXerbla(const char *name, int info) { (void)name; (void)info; }

API cublasStatus_t cublasCreate_v2(void **h) { return new_handle(h); }
API cublasStatus_t cublasXtCreate(void **h) { return new_handle(h); }

API cublasStatus_t cublasGetVersion_v2(void *h, int *v)
{
    (void)h;
    if (!v) return CUBLAS_STATUS_INVALID_VALUE;
    *v = CUBLAS_VERSION;
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasGetVersion(int *v) { return cublasGetVersion_v2(NULL, v); }

API cublasStatus_t cublasGetProperty(int type, int *v)
{
    static const int parts[] = {CUBLAS_VER_MAJOR, CUBLAS_VER_MINOR, CUBLAS_VER_PATCH};
    if (!v || type < 0 || type > 2) return CUBLAS_STATUS_INVALID_VALUE;
    *v = parts[type];
    return CUBLAS_STATUS_SUCCESS;
}

/* Getters return the defaults a fresh handle would have. */
static cublasStatus_t get_int(void *h, int *out, int v)
{
    (void)h;
    if (!out) return CUBLAS_STATUS_INVALID_VALUE;
    *out = v;
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasGetPointerMode_v2(void *h, int *m) { return get_int(h, m, 0); }
API cublasStatus_t cublasGetAtomicsMode(void *h, int *m) { return get_int(h, m, 0); }
API cublasStatus_t cublasGetMathMode(void *h, int *m) { return get_int(h, m, 0); }
API cublasStatus_t cublasGetSmCountTarget(void *h, int *n) { return get_int(h, n, 0); }
API cublasStatus_t cublasGetEmulationStrategy(void *h, int *s) { return get_int(h, s, 0); }

/* A cuBLAS handle remembers its stream in its first word. */
API cublasStatus_t cublasSetStream_v2(void *h, void *stream)
{
    if (!h) return CUBLAS_STATUS_INVALID_VALUE;
    *(void **)h = stream;
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasGetStream_v2(void *h, void **s)
{
    if (!h || !s) return CUBLAS_STATUS_INVALID_VALUE;
    *s = *(void **)h;
    return CUBLAS_STATUS_SUCCESS;
}

static void *stream_of(void *h) { return h ? *(void **)h : NULL; }

API cublasStatus_t cublasSgemm_v2(void *h, int ta, int tb, int m, int n, int k, const float *alpha, const float *A,
                                  int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
{
    (void)ta; (void)tb; (void)alpha; (void)A; (void)lda; (void)B; (void)ldb; (void)beta; (void)C; (void)ldc;
    gemm_work(stream_of(h), m, n, k, 1, STUB_FP32_FLOP_PER_NS);
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasDgemm_v2(void *h, int ta, int tb, int m, int n, int k, const double *alpha, const double *A,
                                  int lda, const double *B, int ldb, const double *beta, double *C, int ldc)
{
    (void)ta; (void)tb; (void)alpha; (void)A; (void)lda; (void)B; (void)ldb; (void)beta; (void)C; (void)ldc;
    gemm_work(stream_of(h), m, n, k, 1, STUB_FP64_FLOP_PER_NS);
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasHgemm(void *h, int ta, int tb, int m, int n, int k, const void *alpha, const void *A, int lda,
                               const void *B, int ldb, const void *beta, void *C, int ldc)
{
    (void)ta; (void)tb; (void)alpha; (void)A; (void)lda; (void)B; (void)ldb; (void)beta; (void)C; (void)ldc;
    gemm_work(stream_of(h), m, n, k, 1, STUB_TENSOR_FLOP_PER_NS);
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasSgemmStridedBatched(void *h, int ta, int tb, int m, int n, int k, const float *alpha,
                                             const float *A, int lda, long long sA, const float *B, int ldb,
                                             long long sB, const float *beta, float *C, int ldc, long long sC,
                                             int batch)
{
    (void)ta; (void)tb; (void)alpha; (void)A; (void)lda; (void)sA; (void)B; (void)ldb; (void)sB; (void)beta;
    (void)C; (void)ldc; (void)sC;
    gemm_work(stream_of(h), m, n, k, batch, STUB_FP32_FLOP_PER_NS);
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasGemmEx(void *h, int ta, int tb, int m, int n, int k, const void *alpha, const void *A,
                                int atype, int lda, const void *B, int btype, int ldb, const void *beta, void *C,
                                int ctype, int ldc, int compute, int algo)
{
    (void)ta; (void)tb; (void)alpha; (void)A; (void)lda; (void)B; (void)btype; (void)ldb; (void)beta; (void)C;
    (void)ctype; (void)ldc; (void)algo;
    gemm_work(stream_of(h), m, n, k, 1, rate_for(atype, compute));
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasGemmStridedBatchedEx(void *h, int ta, int tb, int m, int n, int k, const void *alpha,
                                              const void *A, int atype, int lda, long long sA, const void *B,
                                              int btype, int ldb, long long sB, const void *beta, void *C,
                                              int ctype, int ldc, long long sC, int batch, int compute, int algo)
{
    (void)ta; (void)tb; (void)alpha; (void)A; (void)lda; (void)sA; (void)B; (void)btype; (void)ldb; (void)sB;
    (void)beta; (void)C; (void)ctype; (void)ldc; (void)sC; (void)algo;
    gemm_work(stream_of(h), m, n, k, batch, rate_for(atype, compute));
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasGemmBatchedEx(void *h, int ta, int tb, int m, int n, int k, const void *alpha,
                                       const void *const A[], int atype, int lda, const void *const B[], int btype,
                                       int ldb, const void *beta, void *const C[], int ctype, int ldc, int batch,
                                       int compute, int algo)
{
    (void)ta; (void)tb; (void)alpha; (void)A; (void)lda; (void)B; (void)btype; (void)ldb; (void)beta; (void)C;
    (void)ctype; (void)ldc; (void)algo;
    gemm_work(stream_of(h), m, n, k, batch, rate_for(atype, compute));
    return CUBLAS_STATUS_SUCCESS;
}

#else

/* ---- cuBLASLt ---------------------------------------------------------- */

API const char *cublasLtGetStatusName(cublasStatus_t s) { return status_name(s); }
API const char *cublasLtGetStatusString(cublasStatus_t s) { return s ? "the requested functionality is not supported" : "success"; }
API size_t cublasLtGetVersion(void) { return CUBLAS_VERSION; }
API size_t cublasLtGetCudartVersion(void) { return CUDART_VERSION; }
API unsigned cublasLtDisableCpuInstructionsSetMask(unsigned mask) { (void)mask; return 0; }

API cublasStatus_t cublasLtGetProperty(int type, int *v)
{
    static const int parts[] = {CUBLAS_VER_MAJOR, CUBLAS_VER_MINOR, CUBLAS_VER_PATCH};
    if (!v || type < 0 || type > 2) return CUBLAS_STATUS_INVALID_VALUE;
    *v = parts[type];
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasLtCreate(void **h) { return new_handle(h); }
API cublasStatus_t cublasLtMatmulDescCreate(void **d, int compute, int scale) { (void)compute; (void)scale; return new_handle(d); }

/* Layouts remember their shape so cublasLtMatmul can size its work. */
struct lt_layout {
    uint64_t rows, cols;
    int32_t batch;
};

#define CUBLASLT_MATRIX_LAYOUT_ROWS 2
#define CUBLASLT_MATRIX_LAYOUT_COLS 3
#define CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT 5

API cublasStatus_t cublasLtMatrixLayoutCreate(void **l, int type, uint64_t rows, uint64_t cols, int64_t ld)
{
    (void)type; (void)ld;
    if (new_handle(l) != CUBLAS_STATUS_SUCCESS) return CUBLAS_STATUS_INVALID_VALUE;
    struct lt_layout *lay = *l;
    lay->rows = rows;
    lay->cols = cols;
    lay->batch = 1;
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasLtMatrixLayoutSetAttribute(void *l, int attr, const void *buf, size_t size)
{
    struct lt_layout *lay = l;
    if (!lay || !buf) return CUBLAS_STATUS_INVALID_VALUE;
    if (attr == CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT && size >= sizeof(int32_t)) lay->batch = *(const int32_t *)buf;
    else if (attr == CUBLASLT_MATRIX_LAYOUT_ROWS && size >= sizeof(uint64_t)) lay->rows = *(const uint64_t *)buf;
    else if (attr == CUBLASLT_MATRIX_LAYOUT_COLS && size >= sizeof(uint64_t)) lay->cols = *(const uint64_t *)buf;
    return CUBLAS_STATUS_SUCCESS;
}

/* D is m x n; A holds m x k elements whichever way it is transposed. */
API cublasStatus_t cublasLtMatmul(void *lt, void *desc, const void *alpha, const void *A, const void *Adesc,
                                  const void *B, const void *Bdesc, const void *beta, const void *C,
                                  const void *Cdesc, void *D, const void *Ddesc, const void *algo, void *ws,
                                  size_t ws_size, void *stream)
{
    (void)lt; (void)desc; (void)alpha; (void)A; (void)B; (void)Bdesc; (void)beta; (void)C; (void)Cdesc; (void)D;
    (void)algo; (void)ws; (void)ws_size;
    const struct lt_layout *a = Adesc, *d = Ddesc;
    if (!a || !d || !d->rows) return CUBLAS_STATUS_INVALID_VALUE;
    double m = (double)d->rows, n = (double)d->cols, k = (double)a->rows * (double)a->cols / m;
    gemm_work(stream, m, n, k, d->batch > 0 ? d->batch : 1, STUB_TENSOR_FLOP_PER_NS);
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasLtMatmulPreferenceCreate(void **p) { return new_handle(p); }
API cublasStatus_t cublasLtMatrixTransformDescCreate(void **d, int scale) { (void)scale; return new_handle(d); }

typedef struct {
    uint64_t data[8];
} cublasLtMatmulAlgo_t;

typedef struct {
    cublasLtMatmulAlgo_t algo;
    size_t workspaceSize;
    cublasStatus_t state;
    float wavesCount;
    int reserved[4];
} cublasLtMatmulHeuristicResult_t;

/* Always "finds" one algorithm that needs no workspace. */
API cublasStatus_t cublasLtMatmulAlgoGetHeuristic(void *h, void *desc, void *a, void *b, void *c, void *d,
                                                  void *pref, int requested,
                                                  cublasLtMatmulHeuristicResult_t *results, int *returned)
{
    (void)h; (void)desc; (void)a; (void)b; (void)c; (void)d; (void)pref;
    if (!results || !returned || requested < 1) return CUBLAS_STATUS_INVALID_VALUE;
    memset(results, 0, sizeof *results);
    results->wavesCount = 1.0f;
    *returned = 1;
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasLtMatmulAlgoGetIds(void *h, int compute, int scale, int a, int b, int c, int d,
                                            int requested, int *ids, int *returned)
{
    (void)h; (void)compute; (void)scale; (void)a; (void)b; (void)c; (void)d;
    if (!returned) return CUBLAS_STATUS_INVALID_VALUE;
    if (requested > 0 && ids) ids[0] = 0;
    *returned = requested > 0 ? 1 : 0;
    return CUBLAS_STATUS_SUCCESS;
}

/* Attribute getters: report a zero value of the requested size. */
static cublasStatus_t get_attr(void *buf, size_t size, size_t *written)
{
    if (buf && size) memset(buf, 0, size);
    if (written) *written = size;
    return CUBLAS_STATUS_SUCCESS;
}

API cublasStatus_t cublasLtMatmulDescGetAttribute(void *d, int attr, void *buf, size_t size, size_t *written)
{
    (void)d; (void)attr;
    return get_attr(buf, size, written);
}

API cublasStatus_t cublasLtMatrixLayoutGetAttribute(void *l, int attr, void *buf, size_t size, size_t *written)
{
    (void)l; (void)attr;
    return get_attr(buf, size, written);
}

API cublasStatus_t cublasLtMatmulPreferenceGetAttribute(void *p, int attr, void *buf, size_t size, size_t *written)
{
    (void)p; (void)attr;
    return get_attr(buf, size, written);
}

API cublasStatus_t cublasLtMatmulAlgoConfigGetAttribute(const void *a, int attr, void *buf, size_t size, size_t *written)
{
    (void)a; (void)attr;
    return get_attr(buf, size, written);
}

API cublasStatus_t cublasLtMatmulAlgoCapGetAttribute(const void *a, int attr, void *buf, size_t size, size_t *written)
{
    (void)a; (void)attr;
    return get_attr(buf, size, written);
}

#endif
