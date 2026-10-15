// ============================================================================
// shared.hpp — 算子共用基础设施 (仅 operators/ 目录内使用)
//
// 提供逐元素类算子的: 广播形状推导 / CPU 参考内核 / 梯度规则帮手。
// 每个 KernelFunction 契约不变, 这里只是复用实现。
// ============================================================================
#pragma once
#include "operator.hpp"
#include "znn_internal.hpp"

namespace znn {

// PyTorch 规则的两形状广播: 右对齐, 逐维相等或为 1 取大
Shape broadcast_shapes(const Shape& a, const Shape& b);

// ---- 形状推导帮手 ----
void infer_broadcast(Span<const TensorMeta*> inputs, const void*, std::vector<Shape>& outs);   // 二元
void infer_same(Span<const TensorMeta*> inputs, const void*, std::vector<Shape>& outs);        // 一元同形

// ---- CPU 参考内核帮手(输出连续; 输入可为广播/strided 视图) ----
void unary_kernel(Span<const Tensor*> in, Span<Tensor*> out, float (*fn)(float));
void binary_kernel(Span<const Tensor*> in, Span<Tensor*> out, float (*fn)(float, float));

// ---- 梯度规则帮手(输入连续假设, MNIST 路径成立; 非连续留待 contiguous 处理) ----
// 二元逐元素: 两输入梯度 = grad_out 规约回各自形状
void binary_grad(BackwardContext& context);
// 一元逐元素: 梯度 = grad_out * d(前向输入 x, 前向输出 y)
void unary_grad(BackwardContext& context, float (*d)(float x, float y));

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

// 归一化算子入口(层经此调用, 声明集中于此避免头交叉)
Tensor layer_norm_apply(const Tensor& x, const Tensor& w, const Tensor& b, double eps);
Tensor batch_norm2d_apply(const Tensor& x, const Tensor& w, const Tensor& b,
                          Tensor& running_mean, Tensor& running_var,
                          double eps, double momentum, bool training);

// 训练态形状变换(材料化+逆梯度); 纯推理态各视图方法直接返回零拷贝视图
Tensor reshape_for_autograd(const Tensor& x, Shape target);
// 训练态转置(材料化+逆梯度); 纯推理态 Tensor::transpose 直接返回零拷贝视图
Tensor transpose_for_autograd(const Tensor& x, int64_t d0, int64_t d1);

// 池化调用的完整选项(公共 Pool2dOptions 不含核尺寸, 核是前端独立参数)
struct PoolCallOptions {
    Pool2dOptions pool;
    Shape         kernel;
};

// 规约类(sum/mean)共用形状推导: ReduceAttrs 选项
void infer_reduce(Span<const TensorMeta*> inputs, const void* options, std::vector<Shape>& outs);

} // namespace znn
