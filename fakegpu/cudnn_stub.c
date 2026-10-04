/*
 * Fake cuDNN (libcudnn.so.9). Every exported symbol returns
 * CUDNN_STATUS_SUCCESS (generated from symbols/cudnn.syms). Handles and
 * descriptors are distinct dummies; algorithm searches and the backend
 * (graph) API's heuristics always offer exactly one zero-workspace choice.
 * Executing operations costs a fixed simulated time on the handle's stream.
 */
#include "stub_common.h"

typedef int cudnnStatus_t;
#define CUDNN_STATUS_SUCCESS 0
#define CUDNN_STATUS_BAD_PARAM 2000

#define CUDNN_VERSION 92400
#define CUDNN_MAJOR 9
#define CUDNN_MINOR 24
#define CUDNN_PATCHLEVEL 0
#define CUDART_VERSION 13000

/* ---- version / errors -------------------------------------------------- */

API size_t cudnnGetVersion(void) { return CUDNN_VERSION; }
API size_t cudnnGetCudartVersion(void) { return CUDART_VERSION; }
API size_t cudnnGetMaxDeviceVersion(void) { return 1210; }

API cudnnStatus_t cudnnGetProperty(int type, int *v)
{
    static const int parts[] = {CUDNN_MAJOR, CUDNN_MINOR, CUDNN_PATCHLEVEL};
    if (!v || type < 0 || type > 2) return CUDNN_STATUS_BAD_PARAM;
    *v = parts[type];
    return CUDNN_STATUS_SUCCESS;
}

API const char *cudnnGetErrorString(cudnnStatus_t s) { return s ? "CUDNN_STATUS_NOT_SUPPORTED" : "CUDNN_STATUS_SUCCESS"; }

API void cudnnGetLastErrorString(char *msg, size_t max)
{
    if (msg && max) msg[0] = 0;
}

/* ---- handles and descriptors ------------------------------------------- */

static cudnnStatus_t new_handle(void **h)
{
    if (!h) return CUDNN_STATUS_BAD_PARAM;
    *h = stub_handle();
    return CUDNN_STATUS_SUCCESS;
}

API cudnnStatus_t cudnnCreate(void **h) { return new_handle(h); }

/* A cuDNN handle remembers its stream in its first word. */
API cudnnStatus_t cudnnSetStream(void *h, void *stream)
{
    if (!h) return CUDNN_STATUS_BAD_PARAM;
    *(void **)h = stream;
    return CUDNN_STATUS_SUCCESS;
}

API cudnnStatus_t cudnnGetStream(void *h, void **s)
{
    if (!h || !s) return CUDNN_STATUS_BAD_PARAM;
    *s = *(void **)h;
    return CUDNN_STATUS_SUCCESS;
}

#define CUDNN_OP_NS 40000.0 /* one convolution / fused graph */

static cudnnStatus_t op_work(void *h)
{
    stub_gpu_work(h ? *(void **)h : NULL, CUDNN_OP_NS);
    return CUDNN_STATUS_SUCCESS;
}

API cudnnStatus_t cudnnBackendExecute(void *h, void *plan, void *pack) { (void)plan; (void)pack; return op_work(h); }

#define CONV(name)                                                                                       \
    API cudnnStatus_t name(void *h, const void *a, const void *xd, const void *x, const void *wd,        \
                           const void *w, const void *cd, int algo, void *ws, size_t wss, const void *b, \
                           const void *yd, void *y)                                                      \
    {                                                                                                    \
        (void)a; (void)xd; (void)x; (void)wd; (void)w; (void)cd; (void)algo; (void)ws; (void)wss;        \
        (void)b; (void)yd; (void)y;                                                                      \
        return op_work(h);                                                                               \
    }
CONV(cudnnConvolutionForward)
CONV(cudnnConvolutionBackwardData)
CONV(cudnnConvolutionBackwardFilter)

#define CREATE(name) API cudnnStatus_t name(void **d) { return new_handle(d); }
CREATE(cudnnCreateActivationDescriptor)
CREATE(cudnnCreateAttnDescriptor)
CREATE(cudnnCreateConvolutionDescriptor)
CREATE(cudnnCreateCTCLossDescriptor)
CREATE(cudnnCreateDropoutDescriptor)
CREATE(cudnnCreateFilterDescriptor)
CREATE(cudnnCreateLRNDescriptor)
CREATE(cudnnCreateOpTensorDescriptor)
CREATE(cudnnCreatePoolingDescriptor)
CREATE(cudnnCreateReduceTensorDescriptor)
CREATE(cudnnCreateRNNDataDescriptor)
CREATE(cudnnCreateRNNDescriptor)
CREATE(cudnnCreateSeqDataDescriptor)
CREATE(cudnnCreateSpatialTransformerDescriptor)
CREATE(cudnnCreateTensorDescriptor)
CREATE(cudnnCreateTensorTransformDescriptor)

#define CREATE_OPS(name) API cudnnStatus_t name(void **d, int ops) { (void)ops; return new_handle(d); }
CREATE_OPS(cudnnCreateFusedOpsConstParamPack)
CREATE_OPS(cudnnCreateFusedOpsVariantParamPack)
CREATE_OPS(cudnnCreateFusedOpsPlan)

/* ---- legacy algorithm selection ---------------------------------------- */

typedef struct {
    int algo, status;
    float time;
    size_t memory;
    int determinism, mathType, reserved[3];
} cudnnAlgoPerf_t; /* layout shared by the Fwd / BwdData / BwdFilter variants */

static cudnnStatus_t one_algo(int requested, int *returned, cudnnAlgoPerf_t *perf)
{
    if (!returned || (requested > 0 && !perf)) return CUDNN_STATUS_BAD_PARAM;
    if (requested > 0) {
        memset(perf, 0, sizeof *perf);
        perf->determinism = 1; /* CUDNN_DETERMINISTIC */
    }
    *returned = requested > 0 ? 1 : 0;
    return CUDNN_STATUS_SUCCESS;
}

#define ALGO_QUERY(name)                                                                  \
    API cudnnStatus_t name(void *h, void *a, void *b, void *c, void *d, int req, int *ret, \
                           cudnnAlgoPerf_t *perf)                                          \
    {                                                                                      \
        (void)h; (void)a; (void)b; (void)c; (void)d;                                       \
        return one_algo(req, ret, perf);                                                   \
    }
ALGO_QUERY(cudnnGetConvolutionForwardAlgorithm_v7)
ALGO_QUERY(cudnnGetConvolutionBackwardDataAlgorithm_v7)
ALGO_QUERY(cudnnGetConvolutionBackwardFilterAlgorithm_v7)
ALGO_QUERY(cudnnFindConvolutionForwardAlgorithm)
ALGO_QUERY(cudnnFindConvolutionBackwardDataAlgorithm)
ALGO_QUERY(cudnnFindConvolutionBackwardFilterAlgorithm)

#define WORKSPACE_QUERY(name)                                                             \
    API cudnnStatus_t name(void *h, void *a, void *b, void *c, void *d, int algo, size_t *n) \
    {                                                                                      \
        (void)h; (void)a; (void)b; (void)c; (void)d; (void)algo;                           \
        if (!n) return CUDNN_STATUS_BAD_PARAM;                                             \
        *n = 0;                                                                            \
        return CUDNN_STATUS_SUCCESS;                                                       \
    }
WORKSPACE_QUERY(cudnnGetConvolutionForwardWorkspaceSize)
WORKSPACE_QUERY(cudnnGetConvolutionBackwardDataWorkspaceSize)
WORKSPACE_QUERY(cudnnGetConvolutionBackwardFilterWorkspaceSize)

/* ---- backend (graph) API, used by cudnn-frontend / PyTorch ------------- */

#define CUDNN_TYPE_BACKEND_DESCRIPTOR 15

API cudnnStatus_t cudnnBackendCreateDescriptor(int type, void **d)
{
    (void)type;
    return new_handle(d);
}

static size_t attr_size(int type)
{
    switch (type) {
    case 2: case 24: return 1;          /* BOOLEAN, CHAR */
    case 3: case 5: case 6: return 8;   /* INT64, DOUBLE, VOID_PTR */
    case 0: return sizeof(void *);      /* HANDLE */
    case 26: return 16;                 /* FRACTION */
    default: return 4;                  /* enums, FLOAT, INT32 */
    }
}

/* A count query (requested == 0) reports one element, so heuristics and
 * engine lists are never empty. Descriptor arrays are pre-created by the
 * caller and left alone; scalar outputs are zeroed. */
API cudnnStatus_t cudnnBackendGetAttribute(void *const desc, int name, int type, int64_t requested,
                                           int64_t *count, void *array)
{
    (void)desc; (void)name;
    if (count) *count = 1;
    if (array && requested > 0 && type != CUDNN_TYPE_BACKEND_DESCRIPTOR)
        memset(array, 0, attr_size(type));
    return CUDNN_STATUS_SUCCESS;
}
