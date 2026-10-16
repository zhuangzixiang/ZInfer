#include "operator.hpp"
#include "backends/cuda/cuda_backend.hpp"
// linear 的 CUDA 内核: 融合 GEMM + bias(朴素, 每线程一个输出元素)
#include <cuda_runtime.h>

namespace znn {
namespace cuda_backend {

struct LinearOptions {
    bool has_bias;
};

namespace {
constexpr int kBlk = 256;
}

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

void linear_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                 const void* options) {
    const bool has_bias = static_cast<const LinearOptions*>(options)->has_bias;
    const Tensor& x = *in[0];
    const Tensor& w = *in[1];
    Tensor& y = *out[0];
    const int64_t total = y.numel();
    linear_k<<<(int)((total + kBlk - 1) / kBlk), kBlk>>>(
        x.data<float>(), w.data<float>(),
        has_bias ? in[2]->data<float>() : nullptr,
        y.data<float>(), x.size(0), x.size(1), w.size(0), has_bias ? 1 : 0);
}

} // namespace cuda_backend
} // namespace znn
