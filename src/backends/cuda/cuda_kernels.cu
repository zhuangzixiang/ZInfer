// ============================================================================
// cuda_kernels.cu — CUDA 算子内核(逐元素 + GEMM)
//
// 模式: host 包装函数(匹配 KernelFunction 签名)校验连续性 → 启动内核;
// 设备指针经 Tensor::data<float>() 直接获得(Storage 持显存)。
// 扩展: 新算子 = 在此加内核 + cuda_backend.hpp 声明 + 算子文件填槽位。
// ============================================================================
#include "backends/cuda/cuda_backend.hpp"
#include "operator.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

namespace znn {
namespace cuda_backend {

namespace {

constexpr int kBlock = 256;

int blocks(long long n) {
    return static_cast<int>((n + kBlock - 1) / kBlock);
}

// ---- 逐元素设备内核 ----
#define ZNN_BINARY_KERNEL(fname, op)                                          \
    __global__ void fname(const float* a, const float* b, float* y,           \
                          long long n) {                                      \
        const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; \
        if (i < n) y[i] = a[i] op b[i];                                       \
    }

ZNN_BINARY_KERNEL(add_k, +)
ZNN_BINARY_KERNEL(sub_k, -)
ZNN_BINARY_KERNEL(mul_k, *)
ZNN_BINARY_KERNEL(div_k, /)

#define ZNN_UNARY_KERNEL(fname, expr)                                         \
    __global__ void fname(const float* x, float* y, long long n) {            \
        const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; \
        if (i < n) { const float v = x[i]; y[i] = (expr); }                   \
    }

ZNN_UNARY_KERNEL(neg_k, -v)
ZNN_UNARY_KERNEL(exp_k, expf(v))
ZNN_UNARY_KERNEL(log_k, logf(v))
ZNN_UNARY_KERNEL(sqrt_k, sqrtf(v))
ZNN_UNARY_KERNEL(relu_k, v > 0 ? v : 0)
ZNN_UNARY_KERNEL(sigmoid_k, 1.f / (1.f + expf(-v)))
ZNN_UNARY_KERNEL(tanh_k, tanhf(v))
ZNN_UNARY_KERNEL(silu_k, v / (1.f + expf(-v)))
__global__ void gelu_k(const float* x, float* y, long long n) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = x[i];
        const float k = 0.7978845608f * (v + 0.044715f * v * v * v);
        y[i] = 0.5f * v * (1.f + tanhf(k));
    }
}

// ---- 朴素 GEMM(v1 正确性优先; 分块版本见后续优化) ----
__global__ void matmul_k(const float* a, const float* b, float* c,
                         int M, int K, int N) {
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        acc += a[m * K + k] * b[k * N + n];
    }
    c[m * N + n] = acc;
}

// ---- host 包装共用 ----
void require_contiguous2(const Tensor& a, const Tensor& b, const Tensor& y) {
    if (!a.is_contiguous() || !b.is_contiguous() || !y.is_contiguous()) {
        throw std::runtime_error("znn cuda: elementwise needs contiguous (v1)");
    }
}

void require_contiguous1(const Tensor& x, const Tensor& y) {
    if (!x.is_contiguous() || !y.is_contiguous()) {
        throw std::runtime_error("znn cuda: elementwise needs contiguous (v1)");
    }
}

} // namespace

#define ZNN_BINARY_HOST(fname, kname)                                         \
    void fname(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,     \
               const void*) {                                                 \
        require_contiguous2(*in[0], *in[1], *out[0]);                         \
        const long long n = out[0]->numel();                                  \
        kname<<<blocks(n), kBlock>>>(in[0]->data<float>(),                    \
                                     in[1]->data<float>(),                    \
                                     out[0]->data<float>(), n);               \
    }

ZNN_BINARY_HOST(add_cuda, add_k)
ZNN_BINARY_HOST(sub_cuda, sub_k)
ZNN_BINARY_HOST(mul_cuda, mul_k)
ZNN_BINARY_HOST(div_cuda, div_k)

#define ZNN_UNARY_HOST(fname, kname)                                          \
    void fname(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,     \
               const void*) {                                                 \
        require_contiguous1(*in[0], *out[0]);                                 \
        const long long n = out[0]->numel();                                  \
        kname<<<blocks(n), kBlock>>>(in[0]->data<float>(),                    \
                                     out[0]->data<float>(), n);               \
    }

ZNN_UNARY_HOST(neg_cuda, neg_k)
ZNN_UNARY_HOST(exp_cuda, exp_k)
ZNN_UNARY_HOST(log_cuda, log_k)
ZNN_UNARY_HOST(sqrt_cuda, sqrt_k)
ZNN_UNARY_HOST(relu_cuda, relu_k)
ZNN_UNARY_HOST(gelu_cuda, gelu_k)
ZNN_UNARY_HOST(sigmoid_cuda, sigmoid_k)
ZNN_UNARY_HOST(tanh_cuda, tanh_k)
ZNN_UNARY_HOST(silu_cuda, silu_k)

void matmul_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                 const void*) {
    const Tensor& a = *in[0];
    const Tensor& b = *in[1];
    Tensor& y = *out[0];
    require_contiguous2(a, b, y);
    const int M = a.size(0), K = a.size(1), N = b.size(1);
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_k<<<grid, block>>>(a.data<float>(), b.data<float>(),
                              y.data<float>(), M, K, N);
}

} // namespace cuda_backend
} // namespace znn
