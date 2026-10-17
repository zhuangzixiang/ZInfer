#include "operator.hpp"
#include "backends/cuda/cuda_backend.hpp"
// ============================================================================
// cuda_kernels2.cu — 第二批 CUDA 内核: softmax / 规约 / 损失 / 卷积 / 池化
// (逐元素与 GEMM 见 cuda_kernels.cu)
// ============================================================================
#include <cuda_runtime.h>
#include <stdexcept>

namespace znn {

// 从 shared.hpp 精简(避免 nvcc include 链问题)
struct PoolCallOptions {
    Pool2dOptions pool;
    Shape kernel;
};
namespace cuda_backend {

namespace {

constexpr int kBlock = 256;

void require_contiguous1c(const Tensor& x, const Tensor& y) {
    if (!x.is_contiguous() || !y.is_contiguous()) {
        throw std::runtime_error("znn cuda: needs contiguous (v1)");
    }
}

void require_contiguous2c(const Tensor& a, const Tensor& b, const Tensor& y) {
    if (!a.is_contiguous() || !b.is_contiguous() || !y.is_contiguous()) {
        throw std::runtime_error("znn cuda: needs contiguous (v1)");
    }
}

} // namespace

// ---- softmax(末维; 行内多线程两遍归约, 数值稳定) ----
__global__ void softmax_lastdim_k(const float* x, float* y, int64_t n, int64_t d) {
    const int64_t r = blockIdx.x * (int64_t)blockDim.y + threadIdx.y;
    if (r >= n) return;
    const float* xr = x + r * d;
    float* yr = y + r * d;
    extern __shared__ float red[];                       // [2][blockDim.y][blockDim.x]
    float* smax = red;
    float* ssum = red + blockDim.y * blockDim.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int lane = ty * blockDim.x + tx;

    float local_max = -3.4e38f;                          // 1) 行最大
    for (int64_t i = tx; i < d; i += blockDim.x) {
        local_max = local_max < xr[i] ? xr[i] : local_max;
    }
    smax[lane] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {       // warp 树归约
        if (tx < s) smax[lane] = smax[lane] < smax[lane + s] ? smax[lane + s] : smax[lane];
        __syncthreads();
    }
    const float row_max = smax[ty * blockDim.x];

    float local_sum = 0.f;                               // 2) exp 求和
    for (int64_t i = tx; i < d; i += blockDim.x) {
        yr[i] = expf(xr[i] - row_max);
        local_sum += yr[i];
    }
    ssum[lane] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tx < s) ssum[lane] += ssum[lane + s];
        __syncthreads();
    }
    const float row_sum = ssum[ty * blockDim.x];
    for (int64_t i = tx; i < d; i += blockDim.x) {
        yr[i] /= row_sum;                                // 3) 归一化
    }
}

void softmax_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                  const void*) {
    require_contiguous1c(*in[0], *out[0]);
    const Tensor& x = *in[0];
    const int64_t d = x.size(-1);
    const int64_t n = x.numel() / d;
    dim3 block(32, 8);
    dim3 grid(static_cast<unsigned>((n + 7) / 8));
    softmax_lastdim_k<<<grid, block, 2 * 8 * 32 * sizeof(float)>>>(
        x.data<float>(), out[0]->data<float>(), n, d);
}

// ---- sum / mean(末维; 每块一行) ----
__global__ void sum_lastdim_k(const float* x, float* y, int64_t n, int64_t d, float inv) {
    if (blockIdx.x >= n) return;
    extern __shared__ float acc[];
    const float* xr = x + blockIdx.x * d;
    float l = 0.f;
    for (int64_t i = threadIdx.x; i < d; i += blockDim.x) l += xr[i];
    acc[threadIdx.x] = l;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int i = 0; i < blockDim.x; ++i) s += acc[i];
        y[blockIdx.x] = s * inv;
    }
}

void sum_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    require_contiguous1c(*in[0], *out[0]);
    const Tensor& x = *in[0];
    const int64_t d = x.size(-1);
    const int64_t n = x.numel() / d;
    sum_lastdim_k<<<static_cast<unsigned>(n), kBlock, kBlock * sizeof(float)>>>(
        x.data<float>(), out[0]->data<float>(), n, d, 1.f);
}

void mean_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    require_contiguous1c(*in[0], *out[0]);
    const Tensor& x = *in[0];
    const int64_t d = x.size(-1);
    const int64_t n = x.numel() / d;
    sum_lastdim_k<<<static_cast<unsigned>(n), kBlock, kBlock * sizeof(float)>>>(
        x.data<float>(), out[0]->data<float>(), n, d,
        1.f / static_cast<float>(d));
}

// ---- cross_entropy(logits[N,C] + targets[N] → 标量, 单块两遍) ----
__global__ void cross_entropy_k(const float* x, const int64_t* t, float* loss,
                                int64_t n, int64_t c) {
    extern __shared__ float acc[];
    float l = 0.f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float* row = x + i * c;
        float m = row[0];
        for (int64_t j = 1; j < c; ++j) m = m < row[j] ? row[j] : m;
        float s = 0.f;
        for (int64_t j = 0; j < c; ++j) s += expf(row[j] - m);
        l += m + logf(s) - row[t[i]];
    }
    acc[threadIdx.x] = l;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int i = 0; i < blockDim.x; ++i) s += acc[i];
        loss[0] = s / static_cast<float>(n);
    }
}

void cross_entropy_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                        const void*) {
    require_contiguous2c(*in[0], *in[1], *out[0]);
    const Tensor& x = *in[0];
    cross_entropy_k<<<1, kBlock, kBlock * sizeof(float)>>>(
        x.data<float>(), in[1]->data<int64_t>(), out[0]->data<float>(),
        x.size(0), x.size(1));
}

// ---- mse(同形 → 标量) ----
__global__ void mse_k(const float* p, const float* t, float* loss, int64_t n) {
    extern __shared__ float acc[];
    float l = 0.f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float d = p[i] - t[i];
        l += d * d;
    }
    acc[threadIdx.x] = l;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int i = 0; i < blockDim.x; ++i) s += acc[i];
        loss[0] = s / static_cast<float>(n);
    }
}

void mse_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    require_contiguous2c(*in[0], *in[1], *out[0]);
    mse_k<<<1, kBlock, kBlock * sizeof(float)>>>(
        in[0]->data<float>(), in[1]->data<float>(), out[0]->data<float>(),
        in[0]->numel());
}

// ---- conv2d(每线程一个输出元素, 直接朴素含分组) ----
__global__ void conv2d_k(const float* x, const float* w, float* y,
                         int N, int C, int H, int W, int O, int KH, int KW,
                         int HO, int WO,
                         int sh, int sw, int ph, int pw, int dh, int dw, int G) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)N * O * HO * WO;
    if (i >= total) return;
    const int wo = i % WO;
    const int64_t t0 = i / WO;
    const int ho = t0 % HO;
    const int64_t t1 = t0 / HO;
    const int oc = t1 % O;
    const int n = t1 / O;
    const int Cg = C / G;
    const int Og = O / G;
    const int g = oc / Og;
    float acc = 0.f;
    for (int c = g * Cg; c < (g + 1) * Cg; ++c) {
        for (int ki = 0; ki < KH; ++ki) {
            const int ii = ho * sh + ki * dh - ph;
            if (ii < 0 || ii >= H) continue;
            for (int kj = 0; kj < KW; ++kj) {
                const int jj = wo * sw + kj * dw - pw;
                if (jj < 0 || jj >= W) continue;
                acc += x[((n * C + c) * H + ii) * W + jj]
                     * w[(((int64_t)oc * Cg + (c - g * Cg)) * KH + ki) * KW + kj];
            }
        }
    }
    y[i] = acc;
}

// ---- conv2d 快速路径: im2col + 逐组 GEMM + NCHW 回转 ----
// 每线程填 (输出位置, 输入通道) 的整条 KH×KW 窗口(越界补零)
__global__ void im2col_k(const float* x, float* col,
                         int N, int C, int H, int W, int HO, int WO,
                         int KH, int KW,
                         int sh, int sw, int ph, int pw, int dh, int dw) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)N * HO * WO * C;
    if (i >= total) return;
    const int c = i % C;
    const int64_t t = i / C;
    const int wo = t % WO;
    const int64_t t2 = t / WO;
    const int ho = t2 % HO;
    const int n = t2 / HO;
    const int64_t CK = (int64_t)C * KH * KW;
    float* dst = col + (((int64_t)n * HO + ho) * WO + wo) * CK
               + (int64_t)c * KH * KW;
    const float* xn = x + ((int64_t)n * C + c) * H * W;
    for (int ki = 0; ki < KH; ++ki) {
        const int ii = ho * sh + ki * dh - ph;
        for (int kj = 0; kj < KW; ++kj) {
            const int jj = wo * sw + kj * dw - pw;
            dst[ki * KW + kj] =
                (ii < 0 || ii >= H || jj < 0 || jj >= W) ? 0.f : xn[ii * W + jj];
        }
    }
}

// 行主序 y2[M, O] → NCHW y(一次转置写回)
__global__ void rowmajor_to_nchw_k(const float* y2, float* y,
                                   int O, int HO, int WO, int64_t total) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int wo = i % WO;
    const int64_t t = i / WO;
    const int ho = t % HO;
    const int64_t t2 = t / HO;
    const int o = t2 % O;
    const int n = t2 / O;
    y[i] = y2[(((int64_t)n * HO + ho) * WO + wo) * O + o];
}

void conv2d_cuda(KernelContext& ctx, Span<const Tensor*> in, Span<Tensor*> out,
                 const void* options) {
    require_contiguous2c(*in[0], *in[1], *out[0]);
    const Conv2dOptions* o = static_cast<const Conv2dOptions*>(options);
    const Tensor& x = *in[0];
    const Tensor& w = *in[1];
    Tensor& y = *out[0];
    const int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    const int O = w.size(0), KH = w.size(2), KW = w.size(3);
    const int HO = y.size(2), WO = y.size(3);
    const int G = o->groups, Cg = C / G, Og = O / G;
    const int64_t M = (int64_t)N * HO * WO;
    const int CK = C * KH * KW, CKg = Cg * KH * KW;

    // 快速路径: im2col → 逐组 GEMM(列切片靠 ld 表达, 免拷贝) → NCHW 回转
    Tensor col = Tensor::empty(Shape{M, CK}, DType::F32, Device::CUDA);
    im2col_k<<<(unsigned)((M * C + kBlock - 1) / kBlock), kBlock>>>(
        x.data<float>(), col.data<float>(), N, C, H, W, HO, WO, KH, KW,
        o->stride[0], o->stride[1], o->pad[0], o->pad[1],
        o->dilation[0], o->dilation[1]);
    Tensor y2 = Tensor::empty(Shape{M, O}, DType::F32, Device::CUDA);
    bool ok = true;
    for (int g = 0; g < G && ok; ++g) {
        ok = gemm_cuda(col.data<float>() + (int64_t)g * CKg, CK,   // A 列切片, ld=CK
                       w.data<float>() + (int64_t)g * Og * CK, CK, // B=w^T 布局
                       y2.data<float>() + (int64_t)g * Og, O,      // C 列切片, ld=O
                       (int)M, Og, CKg,
                       /*b_transposed=*/true, ctx.precision);
    }
    if (ok) {
        rowmajor_to_nchw_k<<<(unsigned)((y.numel() + kBlock - 1) / kBlock), kBlock>>>(
            y2.data<float>(), y.data<float>(), O, HO, WO, y.numel());
        return;
    }

    // 兜底: 朴素卷积(cuBLAS 缺失或 GEMM 失败)
    conv2d_k<<<static_cast<unsigned>((y.numel() + kBlock - 1) / kBlock), kBlock>>>(
        x.data<float>(), w.data<float>(), y.data<float>(),
        N, C, H, W, O, KH, KW, HO, WO,
        o->stride[0], o->stride[1], o->pad[0], o->pad[1],
        o->dilation[0], o->dilation[1], G);
}

// ---- max/avg pool(每线程一个输出元素) ----
__global__ void max_pool_k(const float* x, float* y,
                           int N, int C, int H, int W, int HO, int WO,
                           int kh, int kw, int sh, int sw, int ph, int pw) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)N * C * HO * WO;
    if (i >= total) return;
    const int wo = i % WO;
    const int64_t t0 = i / WO;
    const int ho = t0 % HO;
    const int64_t nc = t0 / HO;
    const int64_t base = nc * H * W;
    float m = -3.4e38f;
    for (int ki = 0; ki < kh; ++ki) {
        const int ii = ho * sh + ki - ph;
        if (ii < 0 || ii >= H) continue;
        for (int kj = 0; kj < kw; ++kj) {
            const int jj = wo * sw + kj - pw;
            if (jj < 0 || jj >= W) continue;
            const float v = x[base + ii * W + jj];
            m = m < v ? v : m;
        }
    }
    y[i] = m;
}

__global__ void avg_pool_k(const float* x, float* y,
                           int N, int C, int H, int W, int HO, int WO,
                           int kh, int kw, int sh, int sw, int ph, int pw) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)N * C * HO * WO;
    if (i >= total) return;
    const int wo = i % WO;
    const int64_t t0 = i / WO;
    const int ho = t0 % HO;
    const int64_t nc = t0 / HO;
    const int64_t base = nc * H * W;
    float acc = 0.f;
    int cnt = 0;
    for (int ki = 0; ki < kh; ++ki) {
        const int ii = ho * sh + ki - ph;
        if (ii < 0 || ii >= H) continue;
        for (int kj = 0; kj < kw; ++kj) {
            const int jj = wo * sw + kj - pw;
            if (jj < 0 || jj >= W) continue;
            acc += x[base + ii * W + jj];
            ++cnt;
        }
    }
    y[i] = cnt ? acc / cnt : 0.f;
}

void max_pool2d_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                     const void* options) {
    require_contiguous1c(*in[0], *out[0]);
    const PoolCallOptions* o = static_cast<const PoolCallOptions*>(options);
    const Tensor& x = *in[0];
    Tensor& y = *out[0];
    max_pool_k<<<static_cast<unsigned>((y.numel() + kBlock - 1) / kBlock), kBlock>>>(
        x.data<float>(), y.data<float>(),
        x.size(0), x.size(1), x.size(2), x.size(3), y.size(2), y.size(3),
        o->kernel[0], o->kernel[1], o->pool.stride[0], o->pool.stride[1],
        o->pool.pad[0], o->pool.pad[1]);
}

void avg_pool2d_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                     const void* options) {
    require_contiguous1c(*in[0], *out[0]);
    const PoolCallOptions* o = static_cast<const PoolCallOptions*>(options);
    const Tensor& x = *in[0];
    Tensor& y = *out[0];
    avg_pool_k<<<static_cast<unsigned>((y.numel() + kBlock - 1) / kBlock), kBlock>>>(
        x.data<float>(), y.data<float>(),
        x.size(0), x.size(1), x.size(2), x.size(3), y.size(2), y.size(3),
        o->kernel[0], o->kernel[1], o->pool.stride[0], o->pool.stride[1],
        o->pool.pad[0], o->pool.pad[1]);
}


// ---- conv2d 梯度 CUDA 内核(专用, 铁律: 不允许 CPU 桥接) ----
__global__ void conv2d_dw_k(const float* x, const float* g, float* dw,
                            int N, int C, int H, int W, int O,
                            int KH, int KW, int HO, int WO,
                            int sh, int sw, int ph, int pw, int dh, int dw_, int G) {
    // dw[oc, c, ki, kj] = Σ_{n,ho,wo} g[n,oc,ho,wo] · x[n,c,ho·sh+ki·dh-p, wo·sw+kj·dw_-pw]
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)O * (C / G) * KH * KW;
    if (i >= total) return;
    const int kj = i % KW;
    const int64_t t0 = i / KW;
    const int ki = t0 % KH;
    const int64_t t1 = t0 / KH;
    const int cg = t1 % (C / G);                           // 组内通道
    const int oc = t1 / (C / G);
    const int Cg = C / G;
    const int Og = O / G;
    const int grp = oc / Og;
    const int c = grp * Cg + cg;                           // 全局输入通道
    float acc = 0.f;
    for (int n = 0; n < N; ++n) {
        for (int ho = 0; ho < HO; ++ho) {
            const int ii = ho * sh + ki * dh - ph;
            if (ii < 0 || ii >= H) continue;
            for (int wo = 0; wo < WO; ++wo) {
                const int jj = wo * sw + kj * dw_ - pw;
                if (jj < 0 || jj >= W) continue;
                acc += g[((n * O + oc) * HO + ho) * WO + wo]
                     * x[((n * C + c) * H + ii) * W + jj];
            }
        }
    }
    dw[i] = acc;
}

__global__ void conv2d_dx_k(const float* g, const float* w, float* dx,
                            int N, int C, int H, int W, int O,
                            int KH, int KW, int HO, int WO,
                            int sh, int sw, int ph, int pw, int dh, int dw_, int G) {
    // dx[n,c,i,j] = Σ_{oc,ki,kj} g[n,oc,ho,wo] · w[oc,c,ki,kj]
    //   where ho·sh + ki·dh - ph = i AND wo·sw + kj·dw_ - pw = j
    //   → ho = (i + ph - ki·dh) / sh (须整除且在范围内)
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)N * C * H * W;
    if (i >= total) return;
    const int j = i % W;
    const int64_t t0 = i / W;
    const int ii = t0 % H;
    const int64_t t1 = t0 / H;
    const int c = t1 % C;
    const int n = t1 / C;
    const int Cg = C / G;
    const int Og = O / G;
    const int grp = c / Cg;
    float acc = 0.f;
    for (int oc = grp * Og; oc < (grp + 1) * Og; ++oc) {
        for (int ki = 0; ki < KH; ++ki) {
            const int ho_num = ii + ph - ki * dh;
            if (ho_num < 0 || ho_num % sh != 0) continue;
            const int ho = ho_num / sh;
            if (ho >= HO) continue;
            for (int kj = 0; kj < KW; ++kj) {
                const int wo_num = j + pw - kj * dw_;
                if (wo_num < 0 || wo_num % sw != 0) continue;
                const int wo = wo_num / sw;
                if (wo >= WO) continue;
                const int cg = c - grp * Cg;
                acc += g[((n * O + oc) * HO + ho) * WO + wo]
                     * w[(((int64_t)oc * Cg + cg) * KH + ki) * KW + kj];
            }
        }
    }
    dx[i] = acc;
}

void conv2d_grad_cuda(const float* x, const float* w, const float* g,
                      float* dx, float* dw,
                      int N, int C, int H, int W, int O,
                      int KH, int KW, int HO, int WO,
                      int sh, int sw, int ph, int pw, int dh, int dw_, int G) {
    const int64_t dw_total = (int64_t)O * (C / G) * KH * KW;
    conv2d_dw_k<<<(int)((dw_total + 255) / 256), 256>>>(
        x, g, dw, N, C, H, W, O, KH, KW, HO, WO, sh, sw, ph, pw, dh, dw_, G);
    const int64_t dx_total = (int64_t)N * C * H * W;
    conv2d_dx_k<<<(int)((dx_total + 255) / 256), 256>>>(
        g, w, dx, N, C, H, W, O, KH, KW, HO, WO, sh, sw, ph, pw, dh, dw_, G);
}

} // namespace cuda_backend
} // namespace znn
