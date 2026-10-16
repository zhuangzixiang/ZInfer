// linear 的 Eigen 融合内核(kernels[CPU]): GEMM + bias epilogue 一次完成
#include "shared.hpp"
#include <Eigen/Dense>

namespace znn {

struct LinearOptions {
    bool has_bias;
};

void linear_naive(KernelContext&, Span<const Tensor*>, Span<Tensor*>, const void*);

static KernelFunction linear_eigen_impl(KernelContext&, Span<const Tensor*> in,
                                        Span<Tensor*> out, const void* options) {
    const bool has_bias = static_cast<const LinearOptions*>(options)->has_bias;
    const Tensor& x = *in[0];
    const Tensor& w = *in[1];
    const Tensor* b = has_bias ? in[2] : nullptr;
    Tensor& y = *out[0];
    const bool fast = x.is_contiguous() && w.is_contiguous() && y.is_contiguous()
                      && (!b || b->is_contiguous());
    if (!fast) {
        linear_naive(KernelContext{Device::CPU, nullptr}, in, out, options);
        return nullptr;                                    // 不该到这, 仅防呆
    }
    using Vec = Eigen::VectorXf;
    using Mat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    // y = x @ w.T = x * w^T; Eigen 按行主序直接算
    Eigen::Map<const Mat> xm(x.data<float>(), x.size(0), x.size(1));
    Eigen::Map<const Mat> wm(w.data<float>(), w.size(0), w.size(1));   // [OUT, IN]
    Eigen::Map<Mat> ym(y.data<float>(), y.size(0), y.size(1));
    if (b) {
        Eigen::Map<const Vec> bv(b->data<float>(), b->numel());
        ym.noalias() = (xm * wm.transpose()).rowwise() + bv.transpose();
    } else {
        ym.noalias() = xm * wm.transpose();
    }
    return nullptr;
}

// 签名适配: KernelFunction 期望 void, 这里返回 nullptr 不可能——
// 直接改为 void 包装
static void linear_eigen(KernelContext& ctx, Span<const Tensor*> in,
                         Span<Tensor*> out, const void* options) {
    const bool has_bias = static_cast<const LinearOptions*>(options)->has_bias;
    const Tensor& x = *in[0];
    const Tensor& w = *in[1];
    const Tensor* b = has_bias ? in[2] : nullptr;
    Tensor& y = *out[0];
    const bool fast = x.is_contiguous() && w.is_contiguous() && y.is_contiguous()
                      && (!b || b->is_contiguous());
    if (!fast) {
        linear_naive(ctx, in, out, options);
        return;
    }
    using Vec = Eigen::VectorXf;
    using Mat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const Mat> xm(x.data<float>(), x.size(0), x.size(1));
    Eigen::Map<const Mat> wm(w.data<float>(), w.size(0), w.size(1));
    Eigen::Map<Mat> ym(y.data<float>(), y.size(0), y.size(1));
    if (b) {
        Eigen::Map<const Vec> bv(b->data<float>(), b->numel());
        ym.noalias() = (xm * wm.transpose()).rowwise() + bv.transpose();
    } else {
        ym.noalias() = xm * wm.transpose();
    }
}

KernelFunction linear_cpu_eigen() {
    return &linear_eigen;
}

} // namespace znn
