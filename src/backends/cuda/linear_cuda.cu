#include "operator.hpp"
#include "backends/cuda/cuda_backend.hpp"
// linear 的 CUDA 内核: cuBLAS GEMM + bias 行加(朴素兜底: 每线程一个输出元素)
#include <cuda_runtime.h>

namespace znn {
namespace cuda_backend {

struct LinearOptions {
    bool has_bias;
};

namespace {
constexpr int kBlk = 256;
}

// bias 行加: y[i] += b[i % OUT](GEMM 出口后追加, 免第三矩阵乘)
__global__ void bias_row_k(float* y, const float* b, long long n, int cols) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] += b[i % cols];
}

// 朴素兜底内核: 融合 GEMM + bias(每线程一个输出元素)
__global__ void linear_k(const float* x, const float* w, const float* b, float* y,
                         int N, int IN, int OUT, int has_bias) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)N * OUT;
    if (i >= total) return;
    const int o = i % OUT;
    const int n = i / OUT;
    const float* xn = x + n * IN;
    const float* wo = w + o * IN;
    float acc = 0.f;
    for (int k = 0; k < IN; ++k) {
        acc += xn[k] * wo[k];
    }
    y[i] = has_bias ? acc + b[o] : acc;
}

void linear_cuda(KernelContext& ctx, Span<const Tensor*> in, Span<Tensor*> out,
                 const void* options) {
    const bool has_bias = static_cast<const LinearOptions*>(options)->has_bias;
    const Tensor& x = *in[0];
    const Tensor& w = *in[1];
    Tensor& y = *out[0];
    const int N = x.size(0), IN = x.size(1), OUT = w.size(0);

    // 快速路径: GEMM 原语(w 按 [OUT,IN] 原布局直传, b_transposed 即 × w^T)
    if (gemm_cuda(dptr(x), IN, dptr(w), IN,
                  dptr(y), OUT, N, OUT, IN,
                  /*b_transposed=*/true, ctx.precision)) {
        if (has_bias) {
            const long long total = (long long)N * OUT;
            bias_row_k<<<(int)((total + kBlk - 1) / kBlk), kBlk>>>(
                dptr(y), dptr(*in[2]), total, OUT);
        }
        return;
    }

    const int64_t total = y.numel();
    linear_k<<<(int)((total + kBlk - 1) / kBlk), kBlk>>>(
        dptr(x), dptr(w),
        has_bias ? dptr(*in[2]) : nullptr,
        dptr(y), x.size(0), x.size(1), w.size(0), has_bias ? 1 : 0);
}

} // namespace cuda_backend
} // namespace znn
