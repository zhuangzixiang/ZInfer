// ============================================================================
// cuda_kernels.cu — 第一批 CUDA 内核: 逐元素 + GEMM + 优化器
// (第二批见 cuda_kernels2.cu, 第三批见 cuda_kernels3.cu)
// ============================================================================
#include "operator.hpp"
#include "backends/cuda/cuda_backend.hpp"
#include "dynlib.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

namespace znn {
namespace cuda_backend {

namespace {

constexpr int kBlock = 256;

int blocks(long long n) {
    return static_cast<int>((n + kBlock - 1) / kBlock);
}

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

// ---- 朴素 GEMM ----
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

// ---- 优化器设备内核 ----
__global__ void sgd_k(float* p, const float* g, float* vel,
                      long long n, float lr, float momentum, float wd) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (momentum > 0.f) {
        vel[i] = momentum * vel[i] + g[i] + wd * p[i];
        p[i] -= lr * vel[i];
    } else {
        p[i] -= lr * (g[i] + wd * p[i]);
    }
}

__global__ void adam_k(float* p, const float* g, float* m, float* v,
                       long long n, float lr, float b1, float b2,
                       float eps, float wd, float bias1, float bias2) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float grad = g[i] + wd * p[i];
    m[i] = b1 * m[i] + (1.f - b1) * grad;
    v[i] = b2 * v[i] + (1.f - b2) * grad * grad;
    const float mh = m[i] / bias1;
    const float vh = v[i] / bias2;
    p[i] -= lr * mh / (sqrtf(vh) + eps);
}

// ---- host 包装 ----
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

void matmul_cuda(KernelContext& ctx, Span<const Tensor*> in, Span<Tensor*> out,
                 const void*) {
    // 非连续视图(如反向里的转置)先材料化(GPU 原生 d2d 拷贝;
    // 后续可用步长识别直接映射 cuBLAS OP_T 免拷贝)
    Tensor a = in[0]->is_contiguous() ? *in[0] : in[0]->clone();
    Tensor b = in[1]->is_contiguous() ? *in[1] : in[1]->clone();
    Tensor& y = *out[0];
    const int M = a.size(0), K = a.size(1), N = b.size(1);

    // 快速路径: 共享 GEMM 原语(cuBLAS, DLL 缺失时 false → 朴素内核)
    if (gemm_cuda(a.data<float>(), K, b.data<float>(), N,
                  y.data<float>(), N, M, N, K,
                  /*b_transposed=*/false, ctx.precision)) {
        return;
    }

    // 备选路径: 朴素内核(DLL 缺失时)
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_k<<<grid, block>>>(a.data<float>(), b.data<float>(),
                              y.data<float>(), M, K, N);
}

// ---- 优化器 host 入口 ----
void sgd_step_cuda(float* p, const float* g, float* vel, long long n,
                   float lr, float momentum, float wd) {
    sgd_k<<<blocks(n), kBlock>>>(p, g, vel, n, lr, momentum, wd);
}

void adam_step_cuda(float* p, const float* g, float* m, float* v, long long n,
                    float lr, float b1, float b2, float eps, float wd,
                    float bias1, float bias2) {
    adam_k<<<blocks(n), kBlock>>>(p, g, m, v, n, lr, b1, b2, eps, wd,
                                  bias1, bias2);
}

// ---- unary 反向内部算子(9 算子导数表, 与 shared.cpp unary_deriv 同步) ----
__device__ float unary_deriv_d(int fn, float x, float y) {
    switch (fn) {
    case 0: return -1.f;                                  // Neg
    case 1: return y;                                     // Exp
    case 2: return 1.f / x;                               // Log
    case 3: return 0.5f / y;                              // Sqrt
    case 4: return x > 0.f ? 1.f : 0.f;                   // ReLU
    case 5: {                                             // GELU
        const float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
        const float t = tanhf(inner);
        const float dt = (1.f - t * t) * 0.7978845608f * (1.f + 3.f * 0.044715f * x * x);
        return 0.5f * (1.f + t) + 0.5f * x * dt;
    }
    case 6: return y * (1.f - y);                         // Sigmoid
    case 7: return 1.f - y * y;                           // Tanh
    default: {                                            // Silu
        const float sg = 1.f / (1.f + expf(-x));
        return sg + x * sg * (1.f - sg);
    }
    }
}

__global__ void unary_bwd_k(const float* g, const float* x, const float* y,
                            float* gi, long long n, int fn) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    gi[i] = g[i] * unary_deriv_d(fn, x[i], y[i]);
}

void unary_bwd_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                    const void* options) {
    const int fn = static_cast<int>(*static_cast<const unsigned char*>(options));
    const long long n = out[0]->numel();
    unary_bwd_k<<<blocks(n), kBlock>>>(in[0]->data<float>(), in[1]->data<float>(),
                                       in[2]->data<float>(), out[0]->data<float>(),
                                       n, fn);
}

// ---- GradScaler: 梯度 ×(1/s) + inf/nan 检测 ----
__global__ void grad_unscale_k(float* g, float* bad, long long n, float inv) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = g[i] * inv;
    g[i] = v;
    if (!isfinite(v)) *bad = 1.f;   // 各写者写同值, 免原子
}

void grad_unscale_cuda(float* g, float* bad, long long n, float inv) {
    grad_unscale_k<<<blocks(n), kBlock>>>(g, bad, n, inv);
}

void memset0_cuda(void* p, size_t bytes) {
    cudaMemset(p, 0, bytes);
}

} // namespace cuda_backend
} // namespace znn
