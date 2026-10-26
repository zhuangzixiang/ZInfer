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
#include "dnnl.h"   // oneDNN 纯 C 头(vendored, 仅取类型; 符号经动态加载)

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

// ---- oneDNN(CPU 原语委托, third_party/onednn 头 + dnnl.dll 动态加载) ----
// C API 无重载 → 全部 decltype 原生签名
namespace onednn {
bool available_onednn();
extern dnnl_engine_t cached_engine;   // available() 成功后一次性创建, 常驻
extern dnnl_stream_t cached_stream;
// ZNN_NO_ONEDNN="eltwise,gemm,...": 按家族禁用(排查/对拍用); 空串=全禁
bool family_off(const char* family);
extern decltype(&dnnl_engine_create)  engine_create;
extern decltype(&dnnl_stream_create)  stream_create;
extern decltype(&dnnl_engine_destroy) engine_destroy;
extern decltype(&dnnl_stream_destroy) stream_destroy;
extern decltype(&dnnl_memory_destroy)  memory_destroy_ext;
extern decltype(&dnnl_stream_wait)    stream_wait;
extern decltype(&dnnl_memory_desc_create_with_strides) md_create;
extern decltype(&dnnl_memory_desc_create_with_tag) md_tag;
extern decltype(&dnnl_reorder_primitive_desc_create) reorder_pd;
extern decltype(&dnnl_memory_set_data_handle) memory_set_handle;
extern decltype(&dnnl_memory_desc_get_size) md_size;
extern decltype(&dnnl_memory_desc_equal) md_equal;
extern decltype(&dnnl_memory_desc_destroy)              md_destroy;
extern decltype(&dnnl_memory_create)    memory_create;
extern decltype(&dnnl_memory_destroy)   memory_destroy;
extern decltype(&dnnl_primitive_create) primitive_create;
extern decltype(&dnnl_primitive_destroy) primitive_destroy;
extern decltype(&dnnl_primitive_desc_destroy) pd_destroy;
extern decltype(&dnnl_primitive_execute)      execute;
extern decltype(&dnnl_primitive_desc_query_md) pd_query_md;
extern decltype(&dnnl_sgemm) sgemm;
extern decltype(&dnnl_eltwise_forward_primitive_desc_create) eltwise_pd;
extern decltype(&dnnl_binary_primitive_desc_create)          binary_pd;
extern decltype(&dnnl_softmax_forward_primitive_desc_create) softmax_pd;
extern decltype(&dnnl_reduction_primitive_desc_create)       reduction_pd;
extern decltype(&dnnl_convolution_forward_primitive_desc_create) conv_pd;
extern decltype(&dnnl_pooling_forward_primitive_desc_create)     pool_pd;
extern decltype(&dnnl_layer_normalization_forward_primitive_desc_create) ln_pd;
extern decltype(&dnnl_batch_normalization_forward_primitive_desc_create)  bn_pd;
} // namespace onednn

// ---- 后续新库在此添加 ----
// bool available_onednn();

} // namespace dynlib
} // namespace znn
