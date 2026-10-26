#include "operator.hpp"
#include "backends/cuda/cuda_backend.hpp"
#include "operators/batch_norm_options.hpp"
// ============================================================================
// cuda_kernels3.cu — 第三批 CUDA 内核: attention/LN/BN/embedding/
// 视图族(reshape/transpose/permute/slice)/cat/stack/arange/randen
// 填满全部算子的 kernels[CUDA] 槽位, CPU 兜底仅作为构建开关关闭时的底线。
// ============================================================================
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
                            int64_t n_idx, int64_t D, int64_t rows) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n_idx * D) return;
    const int64_t v = idx[i / D];
    // 越界索引写 0(内核不可抛; 与 CPU ref 的 throw 语义在此分歧, 已知项)
    y[i] = (v >= 0 && v < rows) ? w[v * D + i % D] : 0.f;
}

void embedding_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                    const void*) {
    req1c(*in[0], *out[0]);
    const Tensor& w = *in[0];
    const Tensor& idx = *in[1];
    const int64_t D = w.size(1);
    embedding_k<<<blocks3(idx.numel() * D), kBlk3>>>(dptr(w), dptr_i64(idx), dptr(*out[0]),
        idx.numel(), D, w.numel() / D);
}

// ---- 视图族: 材料化拷贝(每线程一个元素, 按 stride 算偏移) ----
// 步长经结构体按值传入: 数组形参会衰退为指针(宿主栈地址), 设备端
// 解引用即越界 —— 存量内核缺陷, 本块修复
struct CopyMap {
    int rank;
    long long dims[8];
    long long se[8];
    long long de[8];
};

__global__ void strided_copy_k(const float* src, float* dst,
                               long long total, CopyMap m) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= total) return;
    // 行主序下标分解
    int64_t remain = i;
    int64_t s_off = 0, d_off = 0;
    for (int d = m.rank - 1; d >= 0; --d) {
        const int64_t idx_d = remain % m.dims[d];
        remain /= m.dims[d];
        s_off += idx_d * m.se[d];
        d_off += idx_d * m.de[d];
    }
    dst[d_off] = src[s_off];
}

namespace {
// 通用映射填充: dims/se 取自 x, de 取 y 同位维(stack 的错位由调用方修正)
CopyMap make_copy_map(const Tensor& x, const Tensor& y) {
    CopyMap m;
    m.rank = (int)x.dim_count();
    for (int64_t d = 0; d < m.rank; ++d) {
        m.dims[d] = x.size(d);
        m.se[d] = x.strides()[d];
        m.de[d] = y.strides()[d];
    }
    return m;
}
} // namespace

void reshape_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                  const void*) {
    req1c(*in[0], *out[0]);
    // 连续时 reshape = d2d memcpy
    cuda_backend::copy_device_to_device(dptr(*out[0]), dptr(*in[0]),
                                        static_cast<size_t>(in[0]->numel()) * sizeof(float));
}

void transpose_cuda(KernelContext& ctx, Span<const Tensor*> in, Span<Tensor*> out,
                    const void* options) {
    // options: TransposeOptions{d0,d1} —— 轴交换映射: x 轴 d 落在 y 的对侧轴
    struct TOptions { int64_t d0, d1; };
    const TOptions* o = static_cast<const TOptions*>(options);
    req1c(*in[0], *out[0]);
    const Tensor& x = *in[0];
    Tensor& y = *out[0];
    const int64_t d0 = o->d0 < 0 ? o->d0 + x.dim_count() : o->d0;
    const int64_t d1 = o->d1 < 0 ? o->d1 + x.dim_count() : o->d1;
    CopyMap m;
    m.rank = (int)x.dim_count();
    for (int64_t d = 0; d < m.rank; ++d) {
        m.dims[d] = x.size(d);
        m.se[d] = x.strides()[d];
        m.de[d] = (d == d0) ? y.strides()[d1]
                            : (d == d1 ? y.strides()[d0] : y.strides()[d]);
    }
    strided_copy_k<<<blocks3(x.numel()), kBlk3>>>(
        dptr(x), dptr(y), x.numel(), m);
}

// 通用双步长拷贝 host 入口(copy_view_to 的 CUDA 路径共用)
void strided_copy_cuda(const float* src, float* dst, int rank,
                       const long long* ss, const long long* ds,
                       const long long* sh, long long total) {
    CopyMap m;
    m.rank = rank;
    for (int d = 0; d < rank; ++d) {
        m.dims[d] = sh[d];
        m.se[d] = ss[d];
        m.de[d] = ds[d];
    }
    strided_copy_k<<<blocks3(total), kBlk3>>>(src, dst, total, m);
}

void permute_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                  const void* options) {
    // options: PermuteOptions{order[8], rank} —— y 轴 k = x 轴 order[k],
    // 故 x 轴 d 的目标步长 = y.strides()[order^{-1}(d)]
    struct POptions { int64_t order[8]; int64_t rank; };
    const POptions* o = static_cast<const POptions*>(options);
    req1c(*in[0], *out[0]);
    const Tensor& x = *in[0];
    Tensor& y = *out[0];
    int64_t inv[8];
    for (int64_t k = 0; k < o->rank; ++k) {
        inv[o->order[k]] = k;
    }
    CopyMap m;
    m.rank = (int)x.dim_count();
    for (int64_t d = 0; d < m.rank; ++d) {
        m.dims[d] = x.size(d);
        m.se[d] = x.strides()[d];
        m.de[d] = y.strides()[inv[d]];
    }
    strided_copy_k<<<blocks3(x.numel()), kBlk3>>>(
        dptr(x), dptr(y), x.numel(), m);
}

void slice_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                const void* options) {
    // options: SliceOptions{dim, begin, end} —— 源指针按 begin 偏移,
    // 遍历 y 网格(被切维长度已缩短)
    struct SOptions { int64_t dim; int64_t begin; int64_t end; };
    const SOptions* o = static_cast<const SOptions*>(options);
    req1c(*in[0], *out[0]);
    const Tensor& x = *in[0];
    Tensor& y = *out[0];
    const int64_t dim = o->dim < 0 ? o->dim + x.dim_count() : o->dim;
    CopyMap m;
    m.rank = (int)y.dim_count();
    for (int64_t d = 0; d < m.rank; ++d) {
        m.dims[d] = y.size(d);
        m.se[d] = x.strides()[d];
        m.de[d] = y.strides()[d];
    }
    strided_copy_k<<<blocks3(y.numel()), kBlk3>>>(
        dptr(x) + o->begin * x.strides()[dim],
        dptr(y), y.numel(), m);
}

// ---- cat / stack: 每输入一次 strided copy(逐输入启动) ----
void cat_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
              const void* options) {
    Tensor& y = *out[0];
    const int64_t dim = static_cast<const int64_t*>(options)[0];
    int64_t offset = 0;
    for (size_t i = 0; i < in.size(); ++i) {
        const Tensor x = in[i]->is_contiguous() ? *in[i] : in[i]->clone();
        // 沿 dim 的偏移体现在 base pointer
        strided_copy_k<<<blocks3(x.numel()), kBlk3>>>(
            dptr(x), dptr(y) + offset * y.strides()[dim],
            x.numel(), make_copy_map(x, y));
        offset += x.size(dim);
    }
}

void stack_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                const void* options) {
    Tensor& y = *out[0];
    const int64_t dim = static_cast<const int64_t*>(options)[0];
    for (size_t i = 0; i < in.size(); ++i) {
        const Tensor x = in[i]->is_contiguous() ? *in[i] : in[i]->clone();
        CopyMap m;
        m.rank = (int)x.dim_count();
        for (int64_t d = 0; d < m.rank; ++d) {
            m.dims[d] = x.size(d);
            m.se[d] = x.strides()[d];
            m.de[d] = y.strides()[d < dim ? d : d + 1];      // 输出多一维
        }
        strided_copy_k<<<blocks3(x.numel()), kBlk3>>>(
            dptr(x),
            dptr(y) + static_cast<int64_t>(i) * y.strides()[dim],
            x.numel(), m);
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
        dptr(*out[0]), out[0]->numel());
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
    cuda_backend::copy_host_to_device(dptr(*out[0]), host.data(),
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

// 任意 dim 的 LN: 每 warp 一组(组 = 固定其余下标扫 dim, 跨维步进),
// 与 softmax_dim_k 同构; E[x^2]-mu^2 单遍方差(数值保护同末维内核)。
struct LnDimIn {
    long long D;          // 归一化轴长度
    long long inner;      // dim 之后元素积(组内步进)
};

__global__ void layer_norm_dim_k(const float* x, const float* w, const float* b,
                                 float* y, long long groups, LnDimIn m, float eps) {
    const int64_t gi = blockIdx.x * (int64_t)blockDim.y + threadIdx.y;
    if (gi >= groups) return;
    const int lane = threadIdx.x;
    const int64_t base = gi / m.inner * m.D * m.inner + gi % m.inner;
    const float* xr = x + base;
    float* yr = y + base;
    float ls = 0.f, lq = 0.f;
    for (int64_t d = lane; d < m.D; d += 32) {
        const float v = xr[d * m.inner];
        ls += v; lq += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        ls += __shfl_down_sync(0xffffffff, ls, off);
        lq += __shfl_down_sync(0xffffffff, lq, off);
    }
    const float mu = __shfl_sync(0xffffffff, ls, 0) / m.D;
    float var = __shfl_sync(0xffffffff, lq, 0) / m.D - mu * mu;
    var = var > 0.f ? var : 0.f;
    const float inv = rsqrtf(var + eps);
    for (int64_t d = lane; d < m.D; d += 32)
        yr[d * m.inner] = (xr[d * m.inner] - mu) * inv * w[d] + b[d];
}

__global__ void layer_norm_spatial_k(const float* x,const float* w,const float* b,float* y,long long groups,LnDimIn m,float eps) {
    const int lane=threadIdx.x, part=threadIdx.y;
    const long long gi=(long long)blockIdx.x*32+lane;
    const long long base=gi/m.inner*m.D*m.inner+gi%m.inner;
    __shared__ float sums[8][32], squares[8][32];
    float s=0,q=0;
    if(gi<groups)for(long long d=part;d<m.D;d+=8){float v=x[base+d*m.inner];s+=v;q+=v*v;}
    sums[part][lane]=s;squares[part][lane]=q;__syncthreads();
    s=0;q=0;for(int j=0;j<8;++j){s+=sums[j][lane];q+=squares[j][lane];}
    const float mean=s/m.D,inv=rsqrtf(fmaxf(q/m.D-mean*mean,0.f)+eps);
    if(gi<groups)for(long long d=part;d<m.D;d+=8)y[base+d*m.inner]=(x[base+d*m.inner]-mean)*inv*w[d]+b[d];
}
void layer_norm_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                     const void* options) {
    // options = LayerNormOptions{double eps; int64_t dim}(此前仅 double eps,
    // 末维假设 —— 现 dim 经结构体传入)
    struct LnOpts { double eps; int64_t dim; };
    const LnOpts* o = static_cast<const LnOpts*>(options);
    const Tensor& x = *in[0];
    const int64_t rank = x.dim_count();
    const int64_t dim = o->dim < 0 ? rank - 1 : o->dim;
    if (dim == rank - 1 && x.is_contiguous()) {          // 末维: 原快路径
        const int64_t d = x.size(-1);
        const int64_t n_rows = x.numel() / d;
        layer_norm_k<<<static_cast<unsigned>(n_rows), kBlk3,
                       2 * kBlk3 * sizeof(float)>>>(
            dptr(x), dptr(*in[1]), dptr(*in[2]),
            dptr(*out[0]), n_rows, d, static_cast<float>(o->eps));
        return;
    }
    LnDimIn m;
    m.D = x.size(dim);
    m.inner = 1;
    int64_t groups = 1;
    for (int64_t d = 0; d < rank; ++d) {
        if (d < dim) groups *= x.size(d);
        else if (d > dim) m.inner *= x.size(d);
    }
    groups *= m.inner;              // 组数 = outer × inner(gi = o*inner + i)
    Tensor xc = x.is_contiguous() ? x : x.clone();       // 异步池流序释放, 局部安全
    if(m.inner>=1024) {
        layer_norm_spatial_k<<<(unsigned)((groups+31)/32),dim3(32,8)>>>(dptr(xc),dptr(*in[1]),dptr(*in[2]),dptr(*out[0]),groups,m,float(o->eps));
        return;
    }
    layer_norm_dim_k<<<(unsigned)((groups + 7) / 8), dim3(32, 8)>>>(
        dptr(xc), dptr(*in[1]), dptr(*in[2]), dptr(*out[0]), groups, m,
        (float)o->eps);
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

// options 实体布局(与 operators/batch_norm2d.cpp 的 BatchNormOptions 一致;
// 此前按 float[3] 强转 {double,double,bool} —— bool 位型读成 0.0f,
// 训练标志恒为假 → CUDA BN 一直跑 eval 统计)
void batch_norm2d_cuda(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                       const void* options) {
    const BatchNormOptions* o = static_cast<const BatchNormOptions*>(options);
    const Tensor& x = *in[0];
    const int C = x.size(1);
    batch_norm2d_k<<<C, kBlk3, kBlk3 * sizeof(float)>>>(
        dptr(x), dptr(*in[1]), dptr(*in[2]),
        dptr(*in[3]), dptr(*in[4]),
        dptr(*out[0]), x.size(0), C, x.size(2), x.size(3),
        (float)o->eps, (float)o->momentum, o->training ? 1 : 0);
}

// ---- attention: 快速路径 = 批量 GEMM(QK^T → 行 softmax → PV, 每批一个头);
// 兜底内核任意 L/D 均正确(每线程一个输出元素 + 单线程行 softmax) ----
__global__ void attn_scores_k(const float* q, const float* k, float* p,
                              int64_t total, int L, int D, float scale) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= total) return;                        // total = bh*L*L
    const int64_t bh = i / ((int64_t)L * L);
    const int64_t r = i % ((int64_t)L * L);
    const float* qp = q + (bh * L + r / L) * D;
    const float* kp = k + (bh * L + r % L) * D;
    float dot = 0.f;
    for (int d = 0; d < D; ++d) dot += qp[d] * kp[d];
    p[i] = dot * scale;
}

__global__ void attn_softmax_k(float* p, int64_t rows, int L, float scale) {
    const int64_t r = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (r >= rows) return;                         // 每线程一行, 原地 softmax
    float* row = p + r * L;                        // scale 折入指数(快速路径的
    float m = row[0];                              // 分数未缩放, naive 路径传 1)
    for (int j = 1; j < L; ++j) m = m < row[j] ? row[j] : m;
    float s = 0.f;
    for (int j = 0; j < L; ++j) { row[j] = expf(scale * (row[j] - m)); s += row[j]; }
    for (int j = 0; j < L; ++j) row[j] /= s;
}

__global__ void attn_output_k(const float* p, const float* v, float* o,
                              int64_t total, int L, int D) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= total) return;                        // total = bh*L*D
    const int64_t bh = i / ((int64_t)L * D);
    const int64_t r = i % ((int64_t)L * D);
    const float* pp = p + (bh * L + r / D) * L;
    const float* vp = v + bh * L * D + r % D;
    float acc = 0.f;
    for (int j = 0; j < L; ++j) acc += pp[j] * vp[(int64_t)j * D];
    o[i] = acc;
}

// Small attention fusion: one thread computes one output element and keeps
// the score row in registers. This removes the temporary [B*H,L,L] tensor and
// the three launch boundaries that dominate short sequences.
__global__ void attn_fused_small_k(const float* q, const float* k,
                                   const float* v, float* o,
                                   int64_t total, int L, int D,
                                   float scale, int causal) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t bh = i / ((int64_t)L * D);
    const int64_t r = i % ((int64_t)L * D);
    const int row = (int)(r / D), col = (int)(r % D);
    const float* qp = q + (bh * L + row) * D;
    const float* kp = k + bh * L * D;
    const float* vp = v + bh * L * D;
    const int limit = causal ? row + 1 : L;
    float m = -FLT_MAX;
    for (int j = 0; j < limit; ++j) {
        float s = 0.f;
        for (int d = 0; d < D; ++d) s += qp[d] * kp[j * D + d];
        m = fmaxf(m, s * scale);
    }
    float denom = 0.f, acc = 0.f;
    for (int j = 0; j < limit; ++j) {
        float s = 0.f;
        for (int d = 0; d < D; ++d) s += qp[d] * kp[j * D + d];
        const float w = expf(s * scale - m);
        denom += w;
        acc += w * vp[j * D + col];
    }
    o[i] = acc / denom;
}

// ---- embedding 反向: 梯度行散射累加(重复索引原子加) ----
__global__ void embedding_grad_k(const float* g, const long long* idx, float* dw,
                                 long long total, int D) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= total) return;
    atomicAdd(dw + idx[i / D] * D + i % D, g[i]);
}

void embedding_grad_cuda(const float* g, const long long* idx, float* dw,
                         long long total, int D) {
    embedding_grad_k<<<blocks3(total), kBlk3>>>(g, idx, dw, total, D);
}

void attention_cuda(KernelContext& ctx, Span<const Tensor*> in, Span<Tensor*> out,
                    const void* options) {
    const Tensor& q = *in[0];
    const Tensor& k = *in[1];
    const Tensor& v = *in[2];
    Tensor& y = *out[0];
    const int B = q.size(0), H = q.size(1), L = q.size(2), D = q.size(3);
    const float scale = 1.f / std::sqrt(static_cast<float>(D));
    const int64_t bh = (int64_t)B * H;

    struct Options { bool causal; };
    const bool causal = options && static_cast<const Options*>(options)->causal;
    if (L <= 64 && D <= 128 && bh * L * D <= (1 << 20)) {
        const int64_t total = bh * L * D;
        attn_fused_small_k<<<(unsigned)((total + kBlk3 - 1) / kBlk3), kBlk3>>>(
            dptr(q), dptr(k), dptr(v), dptr(y), total, L, D, scale,
            causal ? 1 : 0);
        return;
    }

    // scratch: P[bh, L, L](先存原始分数, softmax 原地覆盖)
    Tensor p = Tensor::empty(Shape{bh * L, L}, DType::F32, Device::CUDA);

    // 快速路径: S=Q×K^T(批量, K 原布局 b_transposed 即 × K^T) → softmax → O=P×V
    if (gemm_batched_cuda(dptr(q), D, L * D,
                          dptr(k), D, L * D,
                          dptr(p), L, (int64_t)L * L,
                          L, L, D, (int)bh,
                          /*b_transposed=*/true, ctx.precision)) {
        attn_softmax_k<<<(unsigned)((bh * L + kBlk3 - 1) / kBlk3), kBlk3>>>(
            dptr(p), bh * L, L, scale);
        if (gemm_batched_cuda(dptr(p), L, (int64_t)L * L,
                              dptr(v), D, (int64_t)L * D,
                              dptr(y), D, (int64_t)L * D,
                              L, D, L, (int)bh,
                              /*b_transposed=*/false, ctx.precision)) {
            return;
        }
    }

    // 兜底: 朴素内核(cuBLAS 缺失或 GEMM 失败)
    const int64_t ns = bh * L * L, no = bh * L * D;
    attn_scores_k<<<(unsigned)((ns + kBlk3 - 1) / kBlk3), kBlk3>>>(
        dptr(q), dptr(k), dptr(p), ns, L, D, scale);
    attn_softmax_k<<<(unsigned)((bh * L + kBlk3 - 1) / kBlk3), kBlk3>>>(
        dptr(p), bh * L, L, 1.f);   // naive 分数已缩放
    attn_output_k<<<(unsigned)((no + kBlk3 - 1) / kBlk3), kBlk3>>>(
        dptr(p), dptr(v), dptr(y), no, L, D);
}

} // namespace cuda_backend
} // namespace znn
