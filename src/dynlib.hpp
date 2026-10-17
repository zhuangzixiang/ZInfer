// ============================================================================
// dynlib.hpp — 全项目唯一的动态加载接口
//
// 职责边界(仅此三件事):
//   1. LoadLibrary / dlopen(打开库)
//   2. GetProcAddress / dlsym(取函数指针)
//   3. available_xxx()(惰性一次性检测)
//
// 不做: 类型定义 / wrapper 函数 / 业务逻辑
// 函数指针签名 = 库原生类型(头文件在此 include, 使用方传递获得)
// ============================================================================
#pragma once

#if defined(ZNN_CUDA)
#include <cublas_v2.h>
#endif

namespace znn {
namespace dynlib {

// ---- CUDA cuBLAS(GEMM 委托, CUDA Toolkit 自带 cublas64_13.dll) ----
#if defined(ZNN_CUDA)
bool available_cublas();

// 句柄(available() 成功后非空)
extern cublasHandle_t cublas_handle;

// 函数指针(库原生签名)
extern decltype(&cublasCreate)       cublas_create;
extern decltype(&cublasDestroy)      cublas_destroy;
extern decltype(&cublasSgemm)        cublas_sgemm;
extern decltype(&cublasSetStream)    cublas_set_stream;
// cublasGemmEx 的显式签名(decltype 不可用: CUDA 生成了重载)
// 签名必须与 cublas_api.h 的 C 声明逐参一致(19 参, 无 D 矩阵;
// 多传 C/D 重复参数会错位 computeType → INVALID_VALUE)
using CublasGemmExFn = cublasStatus_t (*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const void*,
    const void*, cudaDataType, int,
    const void*, cudaDataType, int,
    const void*,
    void*, cudaDataType, int,
    cublasComputeType_t, cublasGemmAlgo_t);
extern CublasGemmExFn cublas_gemm_ex;

// 批量 GEMM(attention 按头并行): SgemmStridedBatched 无重载, 用 decltype
extern decltype(&cublasSgemmStridedBatched) cublas_sgemm_batched;
// GemmStridedBatchedEx 同样有重载, 显式签名(注意: 无 D 矩阵, 与 GemmEx 不同)
using CublasGemmStridedBatchedExFn = cublasStatus_t (*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const void*, const void*, cudaDataType, int, long long,
    const void*, cudaDataType, int, long long,
    const void*,
    void*, cudaDataType, int, long long,
    int, cublasComputeType_t, cublasGemmAlgo_t);
extern CublasGemmStridedBatchedExFn cublas_gemm_batched_ex;
#endif

// ---- 后续新库在此添加 ----
// bool available_onednn();

} // namespace dynlib
} // namespace znn
