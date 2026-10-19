// ============================================================================
// dynlib.cpp — 全项目唯一的动态加载实现
// (LoadLibrary/dlopen + GetProcAddress/dlsym + available_xxx)
// ============================================================================
#include "dynlib.hpp"
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
  #include <windows.h>
  #define ZNN_DLOPEN(name)   LoadLibraryA(name)
  #define ZNN_DLSYM(h, sym)  GetProcAddress(h, sym)
#else
  #include <dlfcn.h>
  #define ZNN_DLOPEN(name)   dlopen(name, RTLD_LAZY)
  #define ZNN_DLSYM(h, sym)  dlsym(h, sym)
#endif

namespace znn {
namespace dynlib {

#if defined(ZNN_CUDA)

cublasHandle_t cublas_handle = nullptr;

decltype(&cublasCreate)    cublas_create     = nullptr;
decltype(&cublasDestroy)   cublas_destroy    = nullptr;
decltype(&cublasSgemm)     cublas_sgemm      = nullptr;
decltype(&cublasSetStream) cublas_set_stream = nullptr;
CublasGemmExFn cublas_gemm_ex    = nullptr;
decltype(&cublasSgemmStridedBatched) cublas_sgemm_batched = nullptr;
CublasGemmStridedBatchedExFn cublas_gemm_batched_ex = nullptr;

bool available_cublas() {
    static bool done = false;
    static bool ok = false;
    if (!done) {
        #ifdef _WIN32
        HMODULE h = LoadLibraryA("cublas64_13.dll");
        if (!h) h = LoadLibraryA("cublas64_12.dll");
        if (!h) h = LoadLibraryA("cublas64_11.dll");
        #else
        void* h = dlopen("libcublas.so.13", RTLD_LAZY);
        if (!h) h = dlopen("libcublas.so.12", RTLD_LAZY);
        if (!h) h = dlopen("libcublas.so", RTLD_LAZY);
        #endif
        if (h) {
            cublas_create = reinterpret_cast<decltype(cublas_create)>(
                ZNN_DLSYM(h, "cublasCreate_v2"));
            cublas_destroy = reinterpret_cast<decltype(cublas_destroy)>(
                ZNN_DLSYM(h, "cublasDestroy_v2"));
            cublas_sgemm = reinterpret_cast<decltype(cublas_sgemm)>(
                ZNN_DLSYM(h, "cublasSgemm_v2"));
            cublas_set_stream = reinterpret_cast<decltype(cublas_set_stream)>(
                ZNN_DLSYM(h, "cublasSetStream_v2"));
            cublas_gemm_ex = reinterpret_cast<CublasGemmExFn>(
                ZNN_DLSYM(h, "cublasGemmEx"));
            cublas_sgemm_batched =
                reinterpret_cast<decltype(cublas_sgemm_batched)>(
                    ZNN_DLSYM(h, "cublasSgemmStridedBatched"));
            cublas_gemm_batched_ex =
                reinterpret_cast<CublasGemmStridedBatchedExFn>(
                    ZNN_DLSYM(h, "cublasGemmStridedBatchedEx"));
            if (cublas_create && cublas_sgemm) {
                if (cublas_create(&cublas_handle) == CUBLAS_STATUS_SUCCESS) {
                    ok = true;
                }
            }
        }
        done = true;
    }
    return ok;
}


#endif // ZNN_CUDA


// ============================================================ oneDNN
// dnnl.dll 动态加载(C API 无重载, decltype 原生签名); 缺失时全部空指针
namespace onednn {

decltype(&dnnl_engine_create)  engine_create  = nullptr;
decltype(&dnnl_stream_create)  stream_create  = nullptr;
decltype(&dnnl_engine_destroy) engine_destroy = nullptr;
decltype(&dnnl_stream_destroy) stream_destroy = nullptr;
decltype(&dnnl_memory_destroy)  memory_destroy_ext = nullptr;
decltype(&dnnl_stream_wait)    stream_wait    = nullptr;
decltype(&dnnl_memory_desc_create_with_strides) md_create = nullptr;
decltype(&dnnl_memory_desc_destroy)              md_destroy = nullptr;
decltype(&dnnl_memory_create)    memory_create  = nullptr;
decltype(&dnnl_memory_destroy)   memory_destroy = nullptr;
decltype(&dnnl_primitive_create) primitive_create = nullptr;
decltype(&dnnl_primitive_destroy) primitive_destroy = nullptr;
decltype(&dnnl_primitive_desc_destroy) pd_destroy = nullptr;
decltype(&dnnl_primitive_execute)      execute = nullptr;
decltype(&dnnl_primitive_desc_query_md) pd_query_md = nullptr;
decltype(&dnnl_sgemm) sgemm = nullptr;
decltype(&dnnl_eltwise_forward_primitive_desc_create) eltwise_pd = nullptr;
decltype(&dnnl_binary_primitive_desc_create)          binary_pd = nullptr;
decltype(&dnnl_softmax_forward_primitive_desc_create) softmax_pd = nullptr;
decltype(&dnnl_reduction_primitive_desc_create)       reduction_pd = nullptr;
decltype(&dnnl_convolution_forward_primitive_desc_create) conv_pd = nullptr;
decltype(&dnnl_pooling_forward_primitive_desc_create)     pool_pd = nullptr;
decltype(&dnnl_layer_normalization_forward_primitive_desc_create) ln_pd = nullptr;
decltype(&dnnl_batch_normalization_forward_primitive_desc_create)  bn_pd = nullptr;

#define ZNN_DLSYM_DNNL(field, name)                                             onednn::field = reinterpret_cast<decltype(onednn::field)>(                      ZNN_DLSYM(h, name))

namespace {
// ZNN_NO_ONEDNN 一次解析: 空串=全部禁用; 逗号列表=按家族禁用
bool env_family_off(const char* family) {
    static const char* env = std::getenv("ZNN_NO_ONEDNN");
    static const bool any = env != nullptr;
    if (!any) return false;
    if (env[0] == 0) return true;          // 空串 = 全禁
    const size_t fl = strlen(family);
    const char* p = env;
    while ((p = strstr(p, family)) != nullptr) {
        const bool left_ok = p == env || p[-1] == ',';
        const bool right_ok = p[fl] == 0 || p[fl] == ',';
        if (left_ok && right_ok) return true;
        ++p;
    }
    return false;
}
} // namespace

bool family_off(const char* family) {
    return env_family_off(family);
}

dnnl_engine_t cached_engine = nullptr;
dnnl_stream_t cached_stream = nullptr;

bool available_onednn() {
    static bool done = false;
    static bool ok = false;
    if (!done) {
        #ifdef _WIN32
        HMODULE h = LoadLibraryA("dnnl.dll");
        if (!h) h = LoadLibraryA("dnnl_v3.13.dll");
        #else
        void* h = dlopen("libdnnl.so.3", RTLD_LAZY);
        if (!h) h = dlopen("libdnnl.so", RTLD_LAZY);
        if (!h) h = dlopen("libdnnl.3.dylib", RTLD_LAZY);   // macOS(homebrew: lib/libdnnl.3.dylib)
        if (!h) h = dlopen("/opt/homebrew/lib/libdnnl.3.dylib", RTLD_LAZY);
        if (!h) h = dlopen("/usr/local/lib/libdnnl.3.dylib", RTLD_LAZY);   // Intel mac
        #endif
        if (h) {
            ZNN_DLSYM_DNNL(engine_create, "dnnl_engine_create");
            ZNN_DLSYM_DNNL(stream_create, "dnnl_stream_create");
            ZNN_DLSYM_DNNL(engine_destroy, "dnnl_engine_destroy");
            ZNN_DLSYM_DNNL(stream_destroy, "dnnl_stream_destroy");
            ZNN_DLSYM_DNNL(memory_destroy_ext, "dnnl_memory_destroy");
            ZNN_DLSYM_DNNL(stream_wait, "dnnl_stream_wait");
            ZNN_DLSYM_DNNL(md_create, "dnnl_memory_desc_create_with_strides");
            ZNN_DLSYM_DNNL(md_destroy, "dnnl_memory_desc_destroy");
            ZNN_DLSYM_DNNL(memory_create, "dnnl_memory_create");
            ZNN_DLSYM_DNNL(memory_destroy, "dnnl_memory_destroy");
            ZNN_DLSYM_DNNL(primitive_create, "dnnl_primitive_create");
            ZNN_DLSYM_DNNL(primitive_destroy, "dnnl_primitive_destroy");
            ZNN_DLSYM_DNNL(pd_destroy, "dnnl_primitive_desc_destroy");
            ZNN_DLSYM_DNNL(execute, "dnnl_primitive_execute");
            ZNN_DLSYM_DNNL(pd_query_md, "dnnl_primitive_desc_query_md");
            ZNN_DLSYM_DNNL(sgemm, "dnnl_sgemm");
            ZNN_DLSYM_DNNL(eltwise_pd, "dnnl_eltwise_forward_primitive_desc_create");
            ZNN_DLSYM_DNNL(binary_pd, "dnnl_binary_primitive_desc_create");
            ZNN_DLSYM_DNNL(softmax_pd, "dnnl_softmax_forward_primitive_desc_create");
            ZNN_DLSYM_DNNL(reduction_pd, "dnnl_reduction_primitive_desc_create");
            ZNN_DLSYM_DNNL(conv_pd, "dnnl_convolution_forward_primitive_desc_create");
            ZNN_DLSYM_DNNL(pool_pd, "dnnl_pooling_forward_primitive_desc_create");
            ZNN_DLSYM_DNNL(ln_pd, "dnnl_layer_normalization_forward_primitive_desc_create");
            ZNN_DLSYM_DNNL(bn_pd, "dnnl_batch_normalization_forward_primitive_desc_create");
            ok = engine_create && stream_create && execute && sgemm && eltwise_pd
                 && conv_pd && md_create && memory_create;
            if (ok) {   // 一次性创建并常驻(小算子高频调用的开销大头)
                if (engine_create(&cached_engine, dnnl_cpu, 0) != dnnl_success
                    || stream_create(&cached_stream, cached_engine,
                                     dnnl_stream_default_flags) != dnnl_success) {
                    cached_engine = nullptr;
                    cached_stream = nullptr;
                    ok = false;
                }
            }
        }
        done = true;
    }
    return ok;
}

} // namespace onednn

} // namespace dynlib
} // namespace znn
