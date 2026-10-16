// ============================================================================
// dynlib.cpp — 全项目唯一的动态加载实现
// (LoadLibrary/dlopen + GetProcAddress/dlsym + available_xxx)
// ============================================================================
#include "dynlib.hpp"

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

} // namespace dynlib
} // namespace znn
