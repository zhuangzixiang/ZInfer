// matmul 的 Eigen 快速内核(kernels[CPU] 槽位): 连续输入走 Eigen GEMM,
// 非连续视图回退朴素路径; 与 ref 的数值一致性由对拍测试保证
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include <Eigen/Dense>

namespace znn {

// 共享给对拍测试: 朴素参考实现(matmul.cpp 内定义)
void matmul_naive(KernelContext& ctx, Span<const Tensor*> in,
                  Span<Tensor*> out, const void* options);

static void matmul_eigen(KernelContext&, Span<const Tensor*> in,
                         Span<Tensor*> out, const void*) {
    const Tensor& a = *in[0];
    const Tensor& b = *in[1];
    Tensor& y = *out[0];
    const bool fast = a.is_contiguous() && b.is_contiguous() && y.is_contiguous();
    if (!fast) {
        KernelContext c{Device::CPU, nullptr};
        matmul_naive(c, in, out, nullptr);
        return;
    }
    if (onednn::gemm(a.data<float>(), (int)a.size(1), b.data<float>(),
                     (int)b.size(1), y.data<float>(), (int)y.size(1),
                     (int)a.size(0), (int)y.size(1), (int)a.size(1),
                     /*trans_b=*/false)) {
        return;   // oneDNN(缺失/失败自动落到 Eigen)
    }
    using Mat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const Mat> ma(a.data<float>(), a.size(0), a.size(1));
    Eigen::Map<const Mat> mb(b.data<float>(), b.size(0), b.size(1));
    Eigen::Map<Mat> my(y.data<float>(), y.size(0), y.size(1));
    my.noalias() = ma * mb;
}

// 注册进 matmul 的 CPU 槽位(matmul.cpp 读取此函数指针)
KernelFunction matmul_cpu_kernel() {
    return &matmul_eigen;
}

} // namespace znn
