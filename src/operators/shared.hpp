// ============================================================================
// shared.hpp — 算子共用基础设施 (仅 operators/ 目录内使用)
//
// 提供逐元素类算子的: 广播形状推导 / CPU 参考内核 / 梯度规则帮手。
// 每个 KernelFunction 契约不变, 这里只是复用实现。
// ============================================================================
#pragma once
#include "operator.hpp"
#include "backends/onednn/onednn.hpp"   // kernels[CPU] 槽位: oneDNN 优先
#include "znn_internal.hpp"

namespace znn {

// PyTorch 规则的两形状广播: 右对齐, 逐维相等或为 1 取大
Shape broadcast_shapes(const Shape& a, const Shape& b);

// ---- 形状推导帮手 ----
void infer_broadcast(Span<const TensorMeta*> inputs, const void*, ShapeList& outs);   // 二元
void infer_same(Span<const TensorMeta*> inputs, const void*, ShapeList& outs);        // 一元同形

// ---- CPU 参考内核帮手(输出连续; 输入可为广播/strided 视图) ----
void unary_kernel(Span<const Tensor*> in, Span<Tensor*> out, float (*fn)(float));
void binary_kernel(Span<const Tensor*> in, Span<Tensor*> out, float (*fn)(float, float));

// ---- CPU 快速内核(kernels[CPU] 槽位用): 连续+无广播走 Eigen SIMD,
//      否则内部回退到朴素路径 ----
void unary_kernel_fast(Span<const Tensor*> in, Span<Tensor*> out, float (*fn)(float));
void binary_kernel_fast(Span<const Tensor*> in, Span<Tensor*> out, float (*fn)(float, float));

// 一元反向内部算子: gi = g × d(x, y)。d 按枚举查表,
// 经派发器原生执行于任意后端(CPU 循环 / CUDA 内核), 无公共 API
enum class UnaryBwd : uint8_t { Neg, Exp, Log, Sqrt, ReLU, GELU, Sigmoid, Tanh, Silu };
extern const Operator g_unary_bwd_operator;

// 可微深拷贝算子(Tensor::clone 经派发器走此: 建图 + 全后端原生拷贝)
extern const Operator g_clone_operator;

// ---- 梯度规则帮手(输入连续假设, MNIST 路径成立; 非连续留待 contiguous 处理) ----
// 二元逐元素: 两输入梯度 = grad_out 规约回各自形状
void binary_grad(BackwardContext& context);
// 一元逐元素: 梯度 = grad_out * d(前向输入 x, 前向输出 y)
void unary_grad(BackwardContext& context, UnaryBwd fn);

// ---- 规约共用(sum/mean/autograd 复用) ----
struct ReduceAttrs {                 // sum/mean 的调用选项
    int64_t dim    = -1;             // 规约维(负索引)
    bool    keepdim = false;         // 保留长度 1 的维
};

// 沿 dim 求和写入 out(out 形状已按 keepdim 约定分配)
void sum_into(const Tensor& x, int64_t dim, bool keepdim, Tensor& out);
// 便利封装: 自行分配输出
Tensor sum_along(const Tensor& x, int64_t dim, bool keepdim);
// sum 的梯度: 把 g 沿 dim 广播回 in_shape 写入 out(mean 梯度再除以 n)
void sum_grad_into(const Tensor& g, int64_t dim, bool keepdim,
                   const Shape& in_shape, Tensor& out);

// 设备无关内核派发(GPU 原生 / CPU 宿主循环; eff 由调用方按语义预计算)
// 梯度广播/置换: out[i] = scale × g[Σ idx·eff]
void grad_bcast_dispatch(const Tensor& g, Tensor& out, long long total, int rank,
                         const long long* dims, const long long* eff, float scale);
// 双步长拷贝(dst 可散射): dp[Σ idx·ds] = sp[Σ idx·ss]
void strided_copy_dispatch(const Tensor& src, Tensor& dst, int rank,
                           const long long* ss, const long long* ds,
                           const long long* sh, long long total);
// Impl 级形态(基址+元素偏移; Storage 拷贝用。CUDA 偏移并入设备指针,
// Metal/Vulkan 偏移须为 0 —— 句柄不可算术, 子缓冲绑定待绑定层支持)
void strided_copy_raw(Device dev, const void* src, int64_t soff, void* dst,
                      int64_t doff, int rank, const long long* ss,
                      const long long* ds, const long long* sh, long long total);
// 任意 GPU 设备的裸句柄(CPU 返回 nullptr) —— device_* 原语/d2d 等用
inline void* gpu_handle(const Tensor& t) {
    switch (t.device()) {
    case Device::CUDA:   return t.data<Tensor::CudaDataPtr>();
    case Device::Vulkan: return t.data<Tensor::VulkanDataPtr>();
    case Device::Metal:  return t.data<Tensor::MetalDataPtr>();
    default:             return nullptr;
    }
}

// CUDA 显存地址(经标记类型取回; 非 CUDA 张量返回 nullptr)
inline float* cuda_ptr(const Tensor& t) {
    return reinterpret_cast<float*>(t.data<Tensor::CudaDataPtr>());
}
inline const float* cuda_ptr_const(const Tensor& t) {
    return reinterpret_cast<const float*>(t.data<Tensor::CudaDataPtr>());
}

// conv2d 朴素参考(对拍与回退共用)
void conv2d_naive(KernelContext& ctx, Span<const Tensor*> in,
                  Span<Tensor*> out, const void* options);
// matmul 朴素参考(对拍与回退共用)
void matmul_naive(KernelContext& ctx, Span<const Tensor*> in,
                  Span<Tensor*> out, const void* options);

// 归一化算子入口(层经此调用, 声明集中于此避免头交叉)
Tensor layer_norm_apply(const Tensor& x, const Tensor& w, const Tensor& b, double eps);
Tensor batch_norm2d_apply(const Tensor& x, const Tensor& w, const Tensor& b,
                          Tensor& running_mean, Tensor& running_var,
                          double eps, double momentum, bool training);

// 训练态形状变换(材料化+逆梯度); 纯推理态各视图方法直接返回零拷贝视图
Tensor reshape_for_autograd(const Tensor& x, Shape target);
// 训练态维度重排/切片(材料化+逆变换梯度); 纯推理态为零拷贝视图
Tensor permute_for_autograd(const Tensor& x, std::initializer_list<int64_t> dims);
Tensor slice_for_autograd(const Tensor& x, int64_t dim, int64_t begin, int64_t end);
// 训练态转置(材料化+逆梯度); 纯推理态 Tensor::transpose 直接返回零拷贝视图
Tensor transpose_for_autograd(const Tensor& x, int64_t d0, int64_t d1);

// 池化调用的完整选项(公共 Pool2dOptions 不含核尺寸, 核是前端独立参数)
struct PoolCallOptions {
    Pool2dOptions pool;
    Shape         kernel;
};

// 规约类(sum/mean)共用形状推导: ReduceAttrs 选项
void infer_reduce(Span<const TensorMeta*> inputs, const void* options, ShapeList& outs);

} // namespace znn
