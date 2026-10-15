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

} // namespace znn
