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

// One warp per row avoids the shared-memory two-dimensional reduction used by
// the general kernel. This is the common small-classifier/attention case.
__global__ void softmax_small_k(const float* x, float* y, int64_t n, int d) {
    const int64_t r = blockIdx.x;
    if (r >= n) return;
    const int lane = threadIdx.x;
    const float* xr = x + r * d;
    float* yr = y + r * d;
    float m = -3.4e38f;
    for (int i = lane; i < d; i += 32) m = fmaxf(m, xr[i]);
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, off));
    m = __shfl_sync(0xffffffff, m, 0);
    float s = 0.f;
    for (int i = lane; i < d; i += 32) {
        yr[i] = expf(xr[i] - m);
        s += yr[i];
    }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffff, s, off);
    s = __shfl_sync(0xffffffff, s, 0);
    for (int i = lane; i < d; i += 32) yr[i] /= s;
}

// 任意 dim 的 softmax: 每 warp 负责一行(沿 dim 的 stride 步进),
// 外层下标经 odims/xstr 步长表折算基址(与 reduce_any_k 同构)。
struct SoftmaxDimIn {
    int rank;                  // 去掉 softmax 维后的维数
    long long odims[8];
    long long xstr[8];
    long long n_dim;
    long long s_dim;           // x/y 同形(连续化后)步长一致
};

__global__ void softmax_dim_k(const float* x, float* y, long long rows,
                              SoftmaxDimIn m) {
    const int64_t r = blockIdx.x * (int64_t)blockDim.y + threadIdx.y;
    if (r >= rows) return;
    int64_t t = r, base = 0;
    for (int d = m.rank - 1; d >= 0; --d) {
        base += (t % m.odims[d]) * m.xstr[d];
        t /= m.odims[d];
    }
    const int lane = threadIdx.x;
    const float* xr = x + base;
    float* yr = y + base;
    float mx = -3.4e38f;
    for (int64_t i = lane; i < m.n_dim; i += 32)
        mx = fmaxf(mx, xr[i * m.s_dim]);
    for (int off = 16; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, off));
    mx = __shfl_sync(0xffffffff, mx, 0);
    float s = 0.f;
    for (int64_t i = lane; i < m.n_dim; i += 32) {
        yr[i * m.s_dim] = expf(xr[i * m.s_dim] - mx);
        s += yr[i * m.s_dim];
    }
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_down_sync(0xffffffff, s, off);
    s = __shfl_sync(0xffffffff, s, 0);
    for (int64_t i = lane; i < m.n_dim; i += 32) yr[i * m.s_dim] /= s;
}

// upsample2x_nearest: 每线程写 2×2 块(单遍直写, 免 cat+reshape 组合拷贝)
__global__ void upsample2x_k(const float* x, float* y, long long total,
                             long long w_in, long long w_out) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= total) return;
    const long long w = i % w_in;
    const long long r = i / w_in;
    const float v = x[i];
    float* dst = y + r * 2 * w_out + w * 2;
    dst[0] = v;
    dst[1] = v;
    dst[w_out] = v;
    dst[w_out + 1] = v;
}

void upsample2x_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                     const void*) {
    const Tensor& x = *in[0];
    const long long w_in = x.size(3);
    const long long hw_in = x.size(2) * w_in;
    const long long total = x.numel();
    // 输入连续时按平面索引; 非连续走 ref 级材料化
    Tensor xc = x.is_contiguous() ? x : x.clone();
    upsample2x_k<<<(unsigned)((total + kBlock - 1) / kBlock), kBlock>>>(
        dptr(xc), dptr(*out[0]), total, w_in, w_in * 2);
}

void softmax_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                  const void* options) {
    Tensor xc = in[0]->is_contiguous() ? *in[0] : in[0]->clone();
    const Tensor& x = xc;
    // softmax 维来自选项(此前无视 dim 恒取末维 —— 语义错误)
    const int64_t dim_req = static_cast<const int64_t*>(options)[0];
    const int64_t dim = dim_req < 0 ? dim_req + x.dim_count() : dim_req;
    if (dim == x.dim_count() - 1) {              // 末维: 专用快路径
        const int64_t d = x.size(-1);
        const int64_t n = x.numel() / d;
        if (d <= 32) {
            softmax_small_k<<<static_cast<unsigned>(n), 32>>>(dptr(x), dptr(*out[0]),
                                                              n, static_cast<int>(d));
            return;
        }
        dim3 block(32, 8);
        dim3 grid(static_cast<unsigned>((n + 7) / 8));
        softmax_lastdim_k<<<grid, block, 2 * 8 * 32 * sizeof(float)>>>(
            dptr(x), dptr(*out[0]), n, d);
        return;
    }
    // 一般维: 连续输入上按步长表逐行 softmax
    SoftmaxDimIn m;
    m.n_dim = x.size(dim);
    m.s_dim = x.strides()[dim];
    m.rank = 0;
    int64_t rows = 1;
    for (int64_t d = 0; d < x.dim_count(); ++d) {
        if (d == dim) continue;
        m.odims[m.rank] = x.size(d);
        m.xstr[m.rank] = x.strides()[d];
        ++m.rank;
        rows *= x.size(d);
    }
    dim3 block(32, 8);
    dim3 grid(static_cast<unsigned>((rows + 7) / 8));
    softmax_dim_k<<<grid, block>>>(dptr(x), dptr(*out[0]), rows, m);
}

// ---- sum / mean 任意维规约: 每线程沿 dim 求和(与 CPU 语义一致;
// 此前内核只支持末维, 无视 dim 选项 —— 存量缺口, 本块修复) ----
// options 布局与 shared.hpp 的 ReduceAttrs 保持一致(规避 nvcc 包含链)
struct ReduceCallOptions {
    int64_t dim;
    bool keepdim;
};

struct ReduceIn {
    int rank;                  // 去掉被规约维后的维数
    long long odims[8];
    long long xstr[8];
    long long n_dim;
    long long s_dim;
    float scale;               // sum 传 1, mean 传 1/n
};

// 两段式规约: 每输出由 nseg 个块分段求和(块内树归约, 固定分段 → 确定性),
// 再由末核逐输出合并。此前一线程一输出串行扫全维, 输出数少(dim0/标量)
// 时整卡只有一两个块在跑, 万元素级规约达 10ms 量级。
constexpr int kRedBlk = 256;

__global__ void reduce_part_k(const float* x, float* part, long long seg,
                              ReduceIn m) {
    const long long o = blockIdx.x;            // 输出
    const long long p = blockIdx.y;            // 段号
    __shared__ float sh[kRedBlk];
    long long t = o, base = 0;
    for (int d = m.rank - 1; d >= 0; --d) {
        base += (t % m.odims[d]) * m.xstr[d];
        t /= m.odims[d];
    }
    const long long begin = p * seg;
    const long long end = begin + seg < m.n_dim ? begin + seg : m.n_dim;
    float acc = 0.f;
    for (long long j = begin + threadIdx.x; j < end; j += kRedBlk)
        acc += x[base + j * m.s_dim];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = kRedBlk / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) part[o * gridDim.y + p] = sh[0];
}

__global__ void reduce_final_k(const float* part, float* y, long long total, long long nseg,
                               float scale) {
    const long long o = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (o >= total) return;
    float acc = 0.f;
    for (long long p = 0; p < nseg; ++p) acc += part[o * nseg + p];
    y[o] = acc * scale;
}

static void reduce_host(Span<const Tensor*> in, Span<Tensor*> out,
                        const void* options, bool is_mean) {
    Tensor xc = in[0]->is_contiguous() ? *in[0] : in[0]->clone();
    const Tensor& x = xc;
    const ReduceCallOptions* a = static_cast<const ReduceCallOptions*>(options);
    const int64_t dim = a->dim < 0 ? a->dim + x.dim_count() : a->dim;
    ReduceIn m;
    m.n_dim = x.size(dim);
    m.s_dim = x.strides()[dim];
    m.scale = is_mean ? 1.f / (float)m.n_dim : 1.f;
    m.rank = 0;
    for (int64_t d = 0; d < x.dim_count(); ++d) {   // keepdim 的 1 长维贡献 0, 可跳过
        if (d == dim) continue;
        m.odims[m.rank] = x.size(d);
        m.xstr[m.rank] = x.strides()[d];
        ++m.rank;
    }
    const long long total = out[0]->numel();
    // 分段策略: 段长约 4×kRedBlk, 输出少时提高并行(上限防爆显存)
    long long nseg = (m.n_dim + 4 * kRedBlk - 1) / (4 * kRedBlk);
    if (nseg > 4096) nseg = 4096;
    if (nseg < 1) nseg = 1;
    const long long seg = (m.n_dim + nseg - 1) / nseg;
    Tensor part = Tensor::empty(Shape{total * nseg}, DType::F32, Device::CUDA);
    dim3 grid((unsigned)total, (unsigned)nseg);
    reduce_part_k<<<grid, kRedBlk>>>(dptr(x), dptr(part), seg, m);
    reduce_final_k<<<(unsigned)((total + kRedBlk - 1) / kRedBlk), kRedBlk>>>(
        dptr(part), dptr(*out[0]), total, nseg, m.scale);
}

void sum_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
              const void* options) {
    reduce_host(in, out, options, /*is_mean=*/false);
}

void mean_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
               const void* options) {
    reduce_host(in, out, options, /*is_mean=*/true);
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
        dptr(x), dptr_i64(*in[1]), dptr(*out[0]),
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
        dptr(*in[0]), dptr(*in[1]), dptr(*out[0]),
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

// ---- 池化反向(每线程一个输出元素, 重算窗口后原子散射) ----
__global__ void max_pool_grad_k(const float* x, const float* g, float* dx,
                                long long total, int H, int W, int HO, int WO,
                                int kh, int kw, int sh, int sw, int ph, int pw) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int wo = i % WO;
    const long long t = i / WO;
    const int ho = t % HO;
    const long long nc = t / HO;
    const float* xc = x + nc * (long long)H * W;
    long long bmax = -1;
    float m = -3.4e38f;
    for (int ki = 0; ki < kh; ++ki) {
        const int ii = ho * sh + ki - ph;
        if (ii < 0 || ii >= H) continue;
        for (int kj = 0; kj < kw; ++kj) {
            const int jj = wo * sw + kj - pw;
            if (jj < 0 || jj >= W) continue;
            if (xc[(long long)ii * W + jj] > m) {
                m = xc[(long long)ii * W + jj];
                bmax = (long long)ii * W + jj;
            }
        }
    }
    if (bmax >= 0) {
        atomicAdd(dx + nc * (long long)H * W + bmax, g[i]);
    }
}

__global__ void avg_pool_grad_k(const float* g, float* dx, long long total,
                                int H, int W, int HO, int WO,
                                int kh, int kw, int sh, int sw, int ph, int pw) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int wo = i % WO;
    const long long t = i / WO;
    const int ho = t % HO;
    const long long nc = t / HO;
    int cnt = 0;
    for (int ki = 0; ki < kh; ++ki) {
        const int ii = ho * sh + ki - ph;
        if (ii < 0 || ii >= H) continue;
        for (int kj = 0; kj < kw; ++kj) {
            const int jj = wo * sw + kj - pw;
            if (jj < 0 || jj >= W) continue;
            ++cnt;
        }
    }
    const float share = g[i] / (float)cnt;
    for (int ki = 0; ki < kh; ++ki) {
        const int ii = ho * sh + ki - ph;
        if (ii < 0 || ii >= H) continue;
        for (int kj = 0; kj < kw; ++kj) {
            const int jj = wo * sw + kj - pw;
            if (jj < 0 || jj >= W) continue;
            atomicAdd(dx + nc * (long long)H * W + ii * W + jj, share);
        }
    }
}

void max_pool_grad_cuda(const float* x, const float* g, float* dx, long long total,
                        int H, int W, int HO, int WO,
                        int kh, int kw, int sh, int sw, int ph, int pw) {
    max_pool_grad_k<<<(unsigned)((total + kBlock - 1) / kBlock), kBlock>>>(x, g, dx, total, H, W, HO, WO,
                                               kh, kw, sh, sw, ph, pw);
}

void avg_pool_grad_cuda(const float* g, float* dx, long long total,
                        int H, int W, int HO, int WO,
                        int kh, int kw, int sh, int sw, int ph, int pw) {
    avg_pool_grad_k<<<(unsigned)((total + kBlock - 1) / kBlock), kBlock>>>(g, dx, total, H, W, HO, WO,
                                               kh, kw, sh, sw, ph, pw);
}

void conv2d_cuda(KernelContext& ctx, Span<const Tensor*> in, Span<Tensor*> out,
                 const void* options) {
    Tensor xm = in[0]->is_contiguous() ? *in[0] : in[0]->clone();   // 视图材料化
    Tensor wm = in[1]->is_contiguous() ? *in[1] : in[1]->clone();
    const Conv2dOptions* o = static_cast<const Conv2dOptions*>(options);
    const Tensor& x = xm;
    const Tensor& w = wm;
    Tensor& y = *out[0];
    const int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    const int O = w.size(0), KH = w.size(2), KW = w.size(3);
    const int HO = y.size(2), WO = y.size(3);
    const int G = o->groups, Cg = C / G, Og = O / G;
    const int64_t M = (int64_t)N * HO * WO;
    const int CK = C * KH * KW, CKg = Cg * KH * KW;

    if (G > 1) {
        // 分组走直接内核: im2col 列切片路径在 kit 矩阵对拍中暴露数值错
        // (ztestkit conv2d groups), 朴素内核与 CPU ref 逐位一致
        conv2d_k<<<static_cast<unsigned>((y.numel() + kBlock - 1) / kBlock), kBlock>>>(
            dptr(x), dptr(w), dptr(y),
            N, C, H, W, O, KH, KW, HO, WO,
            o->stride[0], o->stride[1], o->pad[0], o->pad[1],
            o->dilation[0], o->dilation[1], G);
        return;
    }

    // 快速路径: im2col → 逐组 GEMM(列切片靠 ld 表达, 免拷贝) → NCHW 回转
    Tensor col = Tensor::empty(Shape{M, CK}, DType::F32, Device::CUDA);
    im2col_k<<<(unsigned)((M * C + kBlock - 1) / kBlock), kBlock>>>(
        dptr(x), dptr(col), N, C, H, W, HO, WO, KH, KW,
        o->stride[0], o->stride[1], o->pad[0], o->pad[1],
        o->dilation[0], o->dilation[1]);
    Tensor y2 = Tensor::empty(Shape{M, O}, DType::F32, Device::CUDA);
    bool ok = true;
    for (int g = 0; g < G && ok; ++g) {
        ok = gemm_cuda(dptr(col) + (int64_t)g * CKg, CK,   // A 列切片, ld=CK
                       dptr(w) + (int64_t)g * Og * CK, CK, // B=w^T 布局
                       dptr(y2) + (int64_t)g * Og, O,      // C 列切片, ld=O
                       (int)M, Og, CKg,
                       /*b_transposed=*/true, ctx.precision);
    }
    if (ok) {
        rowmajor_to_nchw_k<<<(unsigned)((y.numel() + kBlock - 1) / kBlock), kBlock>>>(
            dptr(y2), dptr(y), O, HO, WO, y.numel());
        return;
    }

    // 兜底: 朴素卷积(cuBLAS 缺失或 GEMM 失败)
    conv2d_k<<<static_cast<unsigned>((y.numel() + kBlock - 1) / kBlock), kBlock>>>(
        dptr(x), dptr(w), dptr(y),
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
        dptr(x), dptr(y),
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
        dptr(x), dptr(y),
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
