// ============================================================================
// cuda_backend.hpp — CUDA 后端内部接口(znn_internal 经此调用)
//
// 职责: 设备内存分配/释放/拷贝/清零 + CUDA 算子内核工厂。
// 非CUDA 构建时由 cuda_stub.cpp 提供同名实现(明确报错/返回空内核),
// CUDA 构建时由 cuda_device.cu / cuda_kernels.cu 提供真实实现。
// 链路位置: Storage 与 Tensor::to_device 经此分发;
// 各算子文件经 shared.hpp 的内核工厂填充 kernels[CUDA] 槽位。
// ============================================================================
#pragma once
#include <cstddef>

namespace znn {

class Tensor;
struct KernelContext;
template <class T> struct Span;
using KernelFnHost = void (*)(KernelContext&, Span<const Tensor*>,
                              Span<Tensor*>, const void*);

namespace cuda_backend {

// ---- 设备管理 ----
bool   available();                                   // 是否编入且设备可用
void*  alloc(size_t bytes);                           // cudaMalloc
void   free(void* p);                                 // cudaFree
void   memset0(void* p, size_t bytes);                // cudaMemset(0)
void   copy_host_to_device(void* dst, const void* src, size_t bytes);
void   copy_device_to_host(void* dst, const void* src, size_t bytes);
void   copy_device_to_device(void* dst, const void* src, size_t bytes);

// ---- CUDA 算子内核工厂(非CUDA 构建返回 nullptr, 派发器走 CPU 兜底) ----
// 逐元素二元
void add_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void sub_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void mul_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void div_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
// 逐元素一元
void neg_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void exp_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void log_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void sqrt_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void relu_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void gelu_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void sigmoid_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void tanh_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void silu_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
// 第二批(softmax/规约/损失/卷积/池化)
void softmax_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void sum_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void mean_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void cross_entropy_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void mse_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void conv2d_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void max_pool2d_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void avg_pool2d_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
// 矩阵
void matmul_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);

} // namespace cuda_backend
} // namespace znn
