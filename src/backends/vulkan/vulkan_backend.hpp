// ============================================================================
// vulkan_backend.hpp — Vulkan 后端内部接口(znn_internal 经此调用)
//
// 职责: 设备/队列/内存管理 + 计算管线缓存 + 算子内核工厂。
// 未编入(ZNN_VULKAN 未定义)时算子槽位为 nullptr, 派发器透明回落;
// Vulkan 构建时由 vulkan_device.cpp + vulkan_kernels.cpp 提供真实实现。
// SPIR-V: 着色器(src/backends/vulkan/shaders/*.comp)构建期经 glslang
// 编译为字节码头文件嵌入(third_party/glslang/bin)。
// ============================================================================
#pragma once
#include <cstddef>

// 前置声明: .cpp 文件必须先 include operator.hpp 再 include 本文件
namespace znn {
class Tensor;
struct KernelContext;
template <class T> struct Span;

namespace vulkan_backend {

// ---- 设备管理 ----
bool   available();                                   // 是否编入且设备可用
void*  alloc(size_t bytes);                           // VkDeviceMemory+Buffer
void   free(void* p);
void   memset0(void* p, size_t bytes);
void   copy_host_to_device(void* dst, const void* src, size_t bytes);
void   copy_device_to_host(void* dst, const void* src, size_t bytes);
void   copy_device_to_device(void* dst, const void* src, size_t bytes);

// ---- 算子内核(kernels[Vulkan] 槽位; 着色器见 shaders/) ----
void unary_cuda_style_elementwise(KernelContext&, Span<const Tensor*> in,
                                  Span<Tensor*> out, const void*);
void matmul_vulkan(KernelContext&, Span<const Tensor*> in,
                   Span<Tensor*> out, const void*);
void linear_vulkan(KernelContext&, Span<const Tensor*> in,
                   Span<Tensor*> out, const void*);
void softmax_vulkan(KernelContext&, Span<const Tensor*> in,
                    Span<Tensor*> out, const void*);
void sum_vulkan(KernelContext&, Span<const Tensor*> in,
                Span<Tensor*> out, const void*);
void mean_vulkan(KernelContext&, Span<const Tensor*> in,
                 Span<Tensor*> out, const void*);

// 逐元素族(与 CUDA 同名模式): 由 op 文件的内核工厂引用
void add_vulkan(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void sub_vulkan(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void mul_vulkan(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void div_vulkan(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void neg_vulkan(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void exp_vulkan(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void log_vulkan(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void sqrt_vulkan(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void relu_vulkan(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void gelu_vulkan(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void sigmoid_vulkan(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void tanh_vulkan(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);
void silu_vulkan(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);

// ---- 计算桥(内核文件使用; 实现于 vulkan_device.cpp, 隐藏 Vk* 类型) ----
// 获取/派发/缓冲句柄经不透明指针流转, 内核文件不感知 Vulkan 结构
namespace device_bridge {
// 管线获取: code/bytes 为 SPIR-V, c0/c1 为特化常量, nbind 为存储缓冲
// 绑定数(默认 3); 成功返回不透明句柄
void* get_pipeline(const void* code, size_t bytes, unsigned c0, unsigned c1,
                   unsigned nbind = 3);
// 派发: 3 输入缓冲(裸 Storage 句柄) + 推送常量 + 网格维度; true=成功
bool dispatch(void* pipe, void* a, void* b, void* y,
              const void* push, size_t push_size,
              unsigned gx, unsigned gy, unsigned gz);
// 派发: N 缓冲(优化器/多输入内核用)
bool dispatch_bufs(void* pipe, void* const* bufs, int nbuf,
                   const void* push, size_t push_size,
                   unsigned gx, unsigned gy, unsigned gz);
} // namespace device_bridge

// ---- 直连内核(梯度规则/内部拷贝/优化器经 backend_slots 宏调用) ----
void unary_bwd_vulkan(KernelContext&, Span<const Tensor*>, Span<Tensor*>,
                      const void*);
void grad_bcast_vulkan(const float* g, float* out, long long total, int rank,
                       const long long* dims, const long long* eff, float scale);
void strided_copy_vulkan(const float* src, float* dst, int rank,
                         const long long* ss, const long long* ds,
                         const long long* sh, long long total);
void sgd_step_vulkan(float* p, const float* g, float* vel, long long n,
                     float lr, float momentum, float wd);
void adam_step_vulkan(float* p, const float* g, float* m, float* v,
                      long long n, float lr, float b1, float b2, float eps,
                      float wd, float bias1, float bias2);

} // namespace vulkan_backend
} // namespace znn
