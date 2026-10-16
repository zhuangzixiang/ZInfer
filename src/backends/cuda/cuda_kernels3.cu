// ============================================================================
// cuda_kernels3.cu — 第三批 CUDA 内核: attention/LN/BN/embedding/
// 视图族(reshape/transpose/permute/slice)/cat/stack/arange/randen
// 填满全部算子的 kernels[CUDA] 槽位, CPU 兜底仅作为构建开关关闭时的底线。
// ============================================================================
#include "backends/cuda/cuda_backend.hpp"
#include "operator.hpp"
#include "operators/shared.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <random>
#include <stdexcept>

namespace znn {
namespace cuda_backend {

namespace {

constexpr int kBlk3 = 256;

void req1c(const Tensor& x, const Tensor& y) {
    if (!x.is_contiguous() || !y.is_contiguous()) {
        throw std::runtime_error("znn cuda: needs contiguous (v1)");
    }
}

int blocks3(long long n) {
    return static_cast<int>((n + kBlk3 - 1) / kBlk3);
}

} // namespace

// ---- embedding: 每线程一个输出元素 ----
__global__ void embedding_k(const float* w, const int64_t* idx, float* y,
                            int64_t n_idx, int64_t D) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n_idx * D) return;
    const int64_t e = i / D;
    const int64_t d = i % D;
    y[i] = w[idx[e] * D + d];
}

void embedding_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                    const void*) {
    req1c(*in[0], *out[0]);
    const Tensor& w = *in[0];
    const Tensor& idx = *in[1];
    const int64_t D = w.size(1);
    embedding_k<<<blocks3(idx.numel() * D), kBlk3>>>(
        w.data<float>(), idx.data<int64_t>(), out[0]->data<float>(),
        idx.numel(), D);
}

// ---- 视图族: 材料化拷贝(每线程一个元素, 按 stride 算源偏移) ----
__global__ void strided_copy_k(const float* src, float* dst,
                               int64_t s_strides[8], int64_t d_strides[8],
                               int64_t shape[8], int64_t rank, int64_t total) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= total) return;
    // 行主序下标分解
    int64_t remain = i;
    int64_t s_off = 0, d_off = 0;
    for (int64_t d = rank - 1; d >= 0; --d) {
        const int64_t idx_d = remain % shape[d];
        remain /= shape[d];
        s_off += idx_d * s_strides[d];
        d_off += idx_d * d_strides[d];
    }
    dst[d_off] = src[s_off];
}

void reshape_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                  const void*) {
    req1c(*in[0], *out[0]);
    // 连续时 reshape = d2d memcpy
    cuda_backend::copy_device_to_device(out[0]->data<float>(), in[0]->data<float>(),
                                        static_cast<size_t>(in[0]->numel()) * sizeof(float));
}

void transpose_cuda(KernelContext& ctx, Span<const Tensor*> in, Span<Tensor*> out,
                    const void* options) {
    // options 是 TransposeOptions, 但 kernel 不需要——直接按 stride 拷贝
    (void)options;
    req1c(*in[0], *out[0]);
    const Tensor& x = *in[0];
    Tensor& y = *out[0];
    int64_t ss[8], ds[8], sh[8];
    for (int64_t d = 0; d < x.dim_count(); ++d) {
        ss[d] = x.strides()[d];
        ds[d] = y.strides()[d];
        sh[d] = x.size(d);
    }
    strided_copy_k<<<blocks3(x.numel()), kBlk3>>>(
        x.data<float>(), y.data<float>(), ss, ds, sh, x.dim_count(), x.numel());
}

void permute_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                  const void*) {
    req1c(*in[0], *out[0]);
    const Tensor& x = *in[0];
    Tensor& y = *out[0];
    int64_t ss[8], ds[8], sh[8];
    for (int64_t d = 0; d < x.dim_count(); ++d) {
        ss[d] = x.strides()[d];
        ds[d] = y.strides()[d];
        sh[d] = x.size(d);
    }
    strided_copy_k<<<blocks3(x.numel()), kBlk3>>>(
        x.data<float>(), y.data<float>(), ss, ds, sh, x.dim_count(), x.numel());
}

void slice_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                const void*) {
    req1c(*in[0], *out[0]);
    const Tensor& x = *in[0];
    Tensor& y = *out[0];
    int64_t ss[8], ds[8], sh[8];
    for (int64_t d = 0; d < y.dim_count(); ++d) {
        ss[d] = x.strides()[d];
        ds[d] = y.strides()[d];
        sh[d] = y.size(d);
    }
    strided_copy_k<<<blocks3(y.numel()), kBlk3>>>(
        x.data<float>(), y.data<float>(), ss, ds, sh, y.dim_count(), y.numel());
}

// ---- cat / stack: 每输入一次 strided copy(逐输入启动) ----
void cat_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
              const void* options) {
    req1c(*in[0], *out[0]);                                // v1: 全部连续
    Tensor& y = *out[0];
    const int64_t dim = static_cast<const int64_t*>(options)[0];
    int64_t offset = 0;
    for (size_t i = 0; i < in.size(); ++i) {
        const Tensor& x = *in[i];
        req1c(x, y);
        // 该输入在输出中的切片区: 偏移 = offset 沿 dim
        // 简化: 按 y 的 stride 直接映射
        int64_t ss[8], ds[8], sh[8];
        for (int64_t d = 0; d < x.dim_count(); ++d) {
            ss[d] = x.strides()[d];
            ds[d] = y.strides()[d];
            sh[d] = x.size(d);
        }
        // 沿 dim 的偏移体现在 base pointer
        strided_copy_k<<<blocks3(x.numel()), kBlk3>>>(
            x.data<float>(), y.data<float>() + offset * y.strides()[dim],
            ss, ds, sh, x.dim_count(), x.numel());
        offset += x.size(dim);
    }
}

void stack_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                const void* options) {
    Tensor& y = *out[0];
    const int64_t dim = static_cast<const int64_t*>(options)[0];
    for (size_t i = 0; i < in.size(); ++i) {
        const Tensor& x = *in[i];
        req1c(x, y);
        int64_t ss[8], ds[8], sh[8];
        for (int64_t d = 0; d < x.dim_count(); ++d) {
            ss[d] = x.strides()[d];
            ds[d] = y.strides()[d < dim ? d : d + 1];      // 输出多一维
            sh[d] = x.size(d);
        }
        strided_copy_k<<<blocks3(x.numel()), kBlk3>>>(
            x.data<float>(), y.data<float>() + static_cast<int64_t>(i) * y.strides()[dim],
            ss, ds, sh, x.dim_count(), x.numel());
    }
}

// ---- arange: y[i] = i ----
__global__ void arange_k(float* y, int64_t n) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) y[i] = static_cast<float>(i);
}

void arange_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*> out,
                 const void*) {
    arange_k<<<blocks3(out[0]->numel()), kBlk3>>>(
        out[0]->data<float>(), out[0]->numel());
}

// ---- randn: host 生成 Box-Muller, h2d 拷贝 ----
void randn_cuda(KernelContext&, Span<const Tensor*>, Span<Tensor*> out,
                const void* options) {
    const unsigned seed = *static_cast<const unsigned*>(options);
    const int64_t n = out[0]->numel();
    std::vector<float> host(static_cast<size_t>(n));
    std::mt19937 engine(seed ? seed : 42);
    std::normal_distribution<float> dist(0.f, 1.f);
    for (int64_t i = 0; i < n; ++i) host[i] = dist(engine);
    cuda_backend::copy_host_to_device(out[0]->data<float>(), host.data(),
                                      static_cast<size_t>(n) * sizeof(float));
}

// ---- layer_norm: 每块一行(末维), 块内归约 mean/var, 归一化+仿射 ----
__global__ void layer_norm_k(const float* x, const float* w, const float* b,
                             float* y, int64_t n_rows, int64_t d, float eps) {
    const int64_t r = blockIdx.x;
    if (r >= n_rows) return;
    extern __shared__ float red[];
    float* sm = red;                                       // sum
    float* sv = red + blockDim.x;                          // sum of squares
    const float* xr = x + r * d;
    float* yr = y + r * d;

    float lsum = 0.f, lsq = 0.f;
    for (int64_t i = threadIdx.x; i < d; i += blockDim.x) {
        lsum += xr[i];
        lsq += xr[i] * xr[i];
    }
    sm[threadIdx.x] = lsum;
    sv[threadIdx.x] = lsq;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f, q = 0.f;
        for (int i = 0; i < blockDim.x; ++i) { s += sm[i]; q += sv[i]; }
        sm[0] = s / d;                                     // mean
        sv[0] = q / d - sm[0] * sm[0];                     // var
    }
    __syncthreads();
    const float mu = sm[0];
    const float var = sv[0] > 0.f ? sv[0] : 0.f;           // 数值保护
    const float inv = rsqrtf(var + eps);
    for (int64_t i = threadIdx.x; i < d; i += blockDim.x) {
        yr[i] = (xr[i] - mu) * inv * w[i] + b[i];
    }
}

void layer_norm_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                     const void* options) {
    const double eps = *static_cast<const double*>(options);
    const Tensor& x = *in[0];
    const int64_t d = x.size(-1);
    const int64_t n_rows = x.numel() / d;
    layer_norm_k<<<static_cast<unsigned>(n_rows), kBlk3, 2 * kBlk3 * sizeof(float)>>>(
        x.data<float>(), in[1]->data<float>(), in[2]->data<float>(),
        out[0]->data<float>(), n_rows, d, static_cast<float>(eps));
}

// ---- batch_norm2d: 每块一通道, 训练态批统计+running 更新, eval 用 running ----
__global__ void batch_norm2d_k(const float* x, const float* w, const float* b,
                               float* run_mean, float* run_var,
                               float* y, int N, int C, int H, int W,
                               float eps, float momentum, int training) {
    const int c = blockIdx.x;
    if (c >= C) return;
    const int64_t M = (int64_t)N * H * W;
    extern __shared__ float red[];

    float mu, var;
    if (training) {
        float lsum = 0.f;
        for (int64_t i = threadIdx.x; i < M; i += blockDim.x) {
            const int64_t n = i / (H * W);
            const int64_t rem = i % (H * W);
            lsum += x[((n * C + c) * H) * W + rem];
        }
        red[threadIdx.x] = lsum;
        __syncthreads();
        if (threadIdx.x == 0) {
            float s = 0.f;
            for (int i = 0; i < blockDim.x; ++i) s += red[i];
            red[0] = s / M;
        }
        __syncthreads();
        mu = red[0];
        float lsq = 0.f;
        for (int64_t i = threadIdx.x; i < M; i += blockDim.x) {
            const int64_t n = i / (H * W);
            const int64_t rem = i % (H * W);
            const float t = x[((n * C + c) * H) * W + rem] - mu;
            lsq += t * t;
        }
        red[threadIdx.x] = lsq;
        __syncthreads();
        if (threadIdx.x == 0) {
            float s = 0.f;
            for (int i = 0; i < blockDim.x; ++i) s += red[i];
            red[1] = s / M;                                // var
            // 就地更新 running 统计
            run_mean[c] = (1.f - momentum) * run_mean[c] + momentum * red[0];
            run_var[c]  = (1.f - momentum) * run_var[c] + momentum * red[1];
        }
        __syncthreads();
        var = red[1];
    } else {
        mu = run_mean[c];
        var = run_var[c];
    }
    const float inv = rsqrtf(var + eps);
    for (int64_t i = threadIdx.x; i < M; i += blockDim.x) {
        const int64_t n = i / (H * W);
        const int64_t rem = i % (H * W);
        const int64_t off = ((n * C + c) * H) * W + rem;
        y[off] = (x[off] - mu) * inv * w[c] + b[c];
    }
}

void batch_norm2d_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                       const void* options) {
    // options 布局: {eps, momentum, training}
    const float* opts = static_cast<const float*>(options);
    const Tensor& x = *in[0];
    const int C = x.size(1);
    batch_norm2d_k<<<C, kBlk3, kBlk3 * sizeof(float)>>>(
        x.data<float>(), in[1]->data<float>(), in[2]->data<float>(),
        in[3]->data<float>(), in[4]->data<float>(),
        out[0]->data<float>(), x.size(0), C, x.size(2), x.size(3),
        opts[0], opts[1], static_cast<int>(opts[2]));
}

// ---- attention: 两遍(先 S=QK^T/√d + softmax → P, 再 O=PV) ----
__global__ void attn_scores_k(const float* q, const float* k, float* p,
                              int B, int H, int L, int D, float scale) {
    // 每线程算一个 S[i,j](b,h,i,j), 然后块内 softmax(每行一个 warp)
    const int64_t bh = blockIdx.y;                          // b*H+h
    const int i = blockIdx.x * blockDim.y + threadIdx.y;    // 行
    const int j = threadIdx.x;                              // 列(threadIdx.x 扫一行)
    if (i >= L) return;
    const float* qp = q + (bh * L + i) * D;
    const float* kp = k + bh * L * D;

    extern __shared__ float red[];
    float* srow = red + threadIdx.y * L;                    // 每行的 S 值

    // 1) S[i,j] = dot(Q[i], K[j]) * scale
    float dot = 0.f;
    for (int d = 0; d < D; ++d) {
        dot += qp[d] * kp[j * D + d];
    }
    srow[j] = dot * scale;
    __syncthreads();

    // 2) 行 softmax(仅 warp 首线程驱动, 全 warp 参与归约)
    // 简化: 每行由 blockDim.x 个线程直接做
    float m = srow[0];
    for (int jj = 1; jj < L; ++jj) m = m < srow[jj] ? srow[jj] : m;
    float s = 0.f;
    for (int jj = 0; jj < L; ++jj) { srow[jj] = expf(srow[jj] - m); s += srow[jj]; }
    for (int jj = 0; jj < L; ++jj) srow[jj] /= s;
    p[(bh * L + i) * L + j] = srow[j];
}

__global__ void attn_output_k(const float* p, const float* v, float* o,
                              int B, int H, int L, int D) {
    const int64_t bh = blockIdx.y;
    const int i = blockIdx.x * blockDim.y + threadIdx.y;
    const int d = threadIdx.x;
    if (i >= L || d >= D) return;
    float acc = 0.f;
    for (int j = 0; j < L; ++j) {
        acc += p[(bh * L + i) * L + j] * v[(bh * L + j) * D + d];
    }
    o[(bh * L + i) * D + d] = acc;
}

void attention_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                    const void*) {
    const Tensor& q = *in[0];
    const Tensor& k = *in[1];
    const Tensor& v = *in[2];
    Tensor& y = *out[0];
    const int B = q.size(0), H = q.size(1), L = q.size(2), D = q.size(3);
    const float scale = 1.f / std::sqrt(static_cast<float>(D));

    // scratch: P[B*H*L*L]
    Tensor p = Tensor::empty(Shape{B * H * L, L}, DType::F32, Device::CUDA);

    dim3 blk(L < 32 ? L : 32, 4);                           // 32列扫一行, 4行/块
    dim3 grd((L + 3) / 4, static_cast<unsigned>(B * H));
    attn_scores_k<<<grd, blk, 4 * L * sizeof(float)>>>(
        q.data<float>(), k.data<float>(), p.data<float>(), B, H, L, D, scale);

    dim3 blk2(D < 32 ? D : 32, 4);
    dim3 grd2((L + 3) / 4, static_cast<unsigned>(B * H));
    attn_output_k<<<grd2, blk2>>>(
        p.data<float>(), v.data<float>(), y.data<float>(), B, H, L, D);
}

} // namespace cuda_backend
} // namespace znn
