// ============================================================================
// cuda_backend.hpp — CUDA 后端内部接口(znn_internal 经此调用, 仅 CUDA 构建生效)
//
// 职责: 设备内存分配/释放/双向拷贝 + 算子内核注册入口。
// 非CUDA 构建时由 cuda_stub.cpp 提供同名"不可用"实现, 调用方得到明确报错。
// 链路位置: Storage::create/release 与 Tensor::to_device 经此分发。
// ============================================================================
#pragma once
#include <cstddef>

namespace znn {
namespace cuda_backend {

bool   available();                                   // 是否编入且设备可用
void*  alloc(size_t bytes);                           // cudaMalloc
void   free(void* p);                                 // cudaFree
void   copy_host_to_device(void* dst, const void* src, size_t bytes);
void   copy_device_to_host(void* dst, const void* src, size_t bytes);

// 已实现的 CUDA 算子内核(在 .cu 中定义, 注册进各算子的 kernels[CUDA] 槽位)
struct KernelContext;
struct Tensor;
struct Span_;

} // namespace cuda_backend
} // namespace znn
