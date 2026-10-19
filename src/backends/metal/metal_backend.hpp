// ============================================================================
// metal_backend.hpp — Metal 后端内部接口(znn_internal 经此调用)
//
// 职责: 设备/命令队列/缓冲管理 + 管线(MSL 运行时编译) + 算子内核工厂。
// 未编入(ZNN_METAL 未定义)时算子槽位为 nullptr, 派发器透明回落 CPU。
// 实现: metal_device.mm / metal_kernels.mm(Objective-C++, 仅 APPLE 编译;
// MSL 着色器以源码字符串内嵌, 由 Metal 驱动运行时编译 —— 与 Vulkan 的
// 构建期 SPIR-V 不同, 这是 Metal 官方推荐路径, 无离线工具依赖)。
// ============================================================================
#pragma once
#include <cstddef>

namespace znn {
class Tensor;
struct KernelContext;
template <class T> struct Span;

namespace metal_backend {

// ---- 设备管理 ----
bool   available();
void*  alloc(size_t bytes);
void   free(void* p);
void   memset0(void* p, size_t bytes);
void   copy_host_to_device(void* dst, const void* src, size_t bytes);
void   copy_device_to_host(void* dst, const void* src, size_t bytes);
void   copy_device_to_device(void* dst, const void* src, size_t bytes);

// ---- 算子内核(MSL 见 metal_kernels.mm 内嵌字符串) ----
void add_metal(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void sub_metal(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void mul_metal(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void div_metal(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void neg_metal(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void exp_metal(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void log_metal(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void sqrt_metal(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void relu_metal(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void gelu_metal(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void sigmoid_metal(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void tanh_metal(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void silu_metal(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void matmul_metal(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void linear_metal(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void softmax_metal(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void sum_metal(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void mean_metal(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);

} // namespace metal_backend
} // namespace znn
