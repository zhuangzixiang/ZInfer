// conv2d 的 Eigen 快速内核(kernels[CPU] 槽位): im2col + Eigen GEMM;
// 与 ref 的数值一致性由对拍测试保证(朴素路径见 conv2d.cpp)
#include "shared.hpp"
#include <Eigen/Dense>
#include <vector>

namespace znn {

// conv2d.cpp 中的朴素参考实现(回退与对拍共用)
void conv2d_naive(KernelContext& ctx, Span<const Tensor*> in,
                  Span<Tensor*> out, const void* options);

static void conv2d_eigen(KernelContext&, Span<const Tensor*> in,
                         Span<Tensor*> out, const void* options) {
    const Conv2dOptions* o = static_cast<const Conv2dOptions*>(options);
    const Tensor& x = *in[0];
    const Tensor& w = *in[1];
    Tensor& y = *out[0];
    const bool fast = x.is_contiguous() && w.is_contiguous() && y.is_contiguous();
    if (fast && onednn::conv2d(x, w, y,
                               *static_cast<const Conv2dOptions*>(options))) {
        return;   // oneDNN 直接卷积(缺失/分组/膨胀回落 im2col+Eigen)
    }
    if (!fast) {
        conv2d_naive(KernelContext{Device::CPU, nullptr}, in, out, options);
        return;
    }
    const int64_t N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    const int64_t O = w.size(0), KH = w.size(2), KW = w.size(3);
    const int64_t HO = y.size(2), WO = y.size(3);
    const int64_t G = o->groups;
    const int64_t Cg = C / G;                             // 每组输入通道
    const int64_t Og = O / G;                             // 每组输出通道
    using Mat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    const float* xp = x.data<float>();
    const float* wp = w.data<float>();
    float* yp = y.data<float>();
    std::vector<float> col_buf(static_cast<size_t>(Cg * KH * KW * HO * WO));
    for (int64_t g = 0; g < G; ++g) {
    Eigen::Map<const Mat> w_row(wp + g * Og * Cg * KH * KW, Og, Cg * KH * KW);
    for (int64_t n = 0; n < N; ++n) {
        Eigen::Map<Mat> x_col(col_buf.data(), Cg * KH * KW, HO * WO);
        x_col.setZero();
        for (int64_t c = g * Cg; c < (g + 1) * Cg; ++c) {
            for (int64_t ki = 0; ki < KH; ++ki) {
                for (int64_t kj = 0; kj < KW; ++kj) {
                    const int64_t row = ((c - g * Cg) * KH + ki) * KW + kj;
                    for (int64_t ho = 0; ho < HO; ++ho) {
                        const int64_t i = ho * o->stride[0] + ki * o->dilation[0] - o->pad[0];
                        if (i < 0 || i >= H) continue;
                        for (int64_t wo = 0; wo < WO; ++wo) {
                            const int64_t j = wo * o->stride[1] + kj * o->dilation[1] - o->pad[1];
                            if (j < 0 || j >= W) continue;
                            x_col(row, ho * WO + wo) = xp[((n * C + c) * H + i) * W + j];
                        }
                    }
                }
            }
        }
        Eigen::Map<Mat> y_out(yp + n * O * HO * WO + g * Og * HO * WO, Og, HO * WO);
        y_out.noalias() = w_row * x_col;
    }
    }
}

KernelFunction conv2d_cpu_kernel() {
    return &conv2d_eigen;
}

} // namespace znn
