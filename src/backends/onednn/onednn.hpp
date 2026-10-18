// ============================================================================
// onednn.hpp — oneDNN CPU 原语委托(内部接口, 仅 CPU 算子槽位使用)
//
// 契约: 每个封装"尽力执行"——DLL 缺失 / 形状不支持 / 创建失败时返回
// false, 调用方回落 Eigen 快速路径或朴素 ref(全算子覆盖由调用方保证)。
// 性能依赖 oneDNN 内建原语缓存(同描述符重复创建命中缓存)。
// ============================================================================
#pragma once
#include "operator.hpp"
#include "dynlib.hpp"   // dnnl.h 类型与动态加载指针

namespace znn {
namespace onednn {

bool available();   // = dynlib::available_onednn()

// 逐元素(9 一元共用, alg 见实现): 输入连续同形
bool eltwise(Span<const Tensor*> in, Span<Tensor*> out, int alg,
             float alpha = 0.f, float beta = 0.f);   // linear(如 neg)用 alpha

// 二元(4 算子共用): 支持 PyTorch 广播(右对齐, 1 维步长 0)
bool binary(Span<const Tensor*> in, Span<Tensor*> out, int alg);

bool softmax(Span<const Tensor*> in, Span<Tensor*> out, int64_t dim);

// 规约: sum/mean 沿 dim(keepdim 由 out 形状体现)
bool reduce(Span<const Tensor*> in, Span<Tensor*> out, int64_t dim, bool mean);

// GEMM: C[M,N] = A[M,K] × op(B); trans_b=true 时 B 为 [N,K] 行主序(× w^T)
bool gemm(const float* A, int lda, const float* B, int ldb,
          float* C, int ldc, int M, int N, int K, bool trans_b);

// 卷积(NCHW/OIHW, 分组不支持→false)
bool conv2d(const Tensor& x, const Tensor& w, Tensor& y,
            const Conv2dOptions& o);

// 池化(kernel {kh,kw}; avg 除以有效元素数)
bool pool2d(const Tensor& x, Tensor& y, Shape kernel, const Pool2dOptions& o,
            bool max_pool);

// 层归一(末维, w/b 可训练参数走图层)
bool layer_norm(const Tensor& x, const Tensor& w, const Tensor& b,
                Tensor& y, double eps);

// 批归一(推理语义: 用给定统计量)
bool batch_norm(const Tensor& x, const Tensor& w, const Tensor& b,
                const Tensor& mean, const Tensor& var, Tensor& y, double eps);

} // namespace onednn
} // namespace znn
