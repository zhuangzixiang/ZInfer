// ============================================================================
// metal_kernels.mm — Metal 算子内核(MSL 运行时编译)
//
// MSL 着色器内嵌为字符串常量(与 Vulkan 的构建期 SPIR-V 相对 —— Metal 的
// 官方推荐路径是运行时编译, 无离线工具依赖)。数学与已验证的
// Vulkan/CUDA 版逐行对应; 本机无 Apple 硬件, 首次 macOS 构建即验证。
// ============================================================================
#include "operator.hpp"
#include "backends/metal/metal_backend.hpp"
#include "backends/metal/metal_device.hpp"

#if defined(__APPLE__) && defined(ZNN_METAL)

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdexcept>
#include <string>

namespace znn {
namespace metal_backend {

namespace {

id<MTLBuffer> mbuf(const Tensor& t) {
    return device::buf_of(t.data<float>());   // Storage 裸句柄 = MtlBuf*
}

uint32_t rup(uint32_t n, uint32_t a) { return (n + a - 1) / a * a; }  // 向上取整到组宽

// MSL: 通用逐元素(13 算子共用; 二元全维广播, 右对齐步长表)
const char* kMslElementwise = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void elementwise(device const float* a [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        device float* y [[buffer(2)]],
                        constant const int* pc [[buffer(3)]]) {
    // pc[0]=n pc[1]=op pc[2]=rank
    // pc[10+d]=dims pc[20+d]=sa pc[30+d]=sb(维长 1 → 步长 0)
    const int i = (int)((thread_position_in_grid).x);
    if (i >= pc[0]) return;
    int t = i, oa = 0, ob = 0;
    for (int d = pc[2] - 1; d >= 0; --d) {
        const int idx = t % pc[10 + d];
        t /= pc[10 + d];
        oa += idx * pc[20 + d];
        ob += idx * pc[30 + d];
    }
    const float va = a[oa];
    const int op = pc[1];
    float r;
    if (op == 0) r = va + b[ob];
    else if (op == 1) r = va - b[ob];
    else if (op == 2) r = va * b[ob];
    else if (op == 3) r = va / b[ob];
    else if (op == 4) r = -va;
    else if (op == 5) r = exp(va);
    else if (op == 6) r = log(va);
    else if (op == 7) r = sqrt(va);
    else if (op == 8) r = va > 0.0f ? va : 0.0f;
    else if (op == 9) { const float k = 0.7978845608f * (va + 0.044715f * va * va * va);
                        r = 0.5f * va * (1.0f + tanh(k)); }
    else if (op == 10) r = 1.0f / (1.0f + exp(-va));
    else if (op == 11) r = tanh(va);
    else r = va / (1.0f + exp(-va));   // 12 silu
    y[i] = r;
}
)MSL";

// MSL: GEMM 行主序 C[M,N]=A×op(B); TB=1 时 B 为 [N,K](× w^T)
// 8x8 threadgroup 分块, threadgroup memory 缓冲
const char* kMslGemm = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void gemm(device const float* a [[buffer(0)]],
                 device const float* b [[buffer(1)]],
                 device float* y [[buffer(2)]],
                 constant const int* pc [[buffer(3)]]) {
    // pc = {M, N, K, lda, ldb, ldc, TB}
    const int row = (int)((thread_position_in_grid).y);
    const int col = (int)((thread_position_in_grid).x);
    const int lr = (int)((thread_position_in_threadgroup).y);
    const int lc = (int)((thread_position_in_threadgroup).x);
    const bool inr = (row < pc[0]) && (col < pc[1]);

    threadgroup float ta[8][9];
    threadgroup float tb[8][9];

    float acc = 0.0f;
    const int tiles = (pc[2] + 7) / 8;
    for (int t = 0; t < tiles; ++t) {
        const int ka = t * 8 + lc;
        // ta 按 row 守、tb 按 col 守(独立): 边界 tile 时读者所需单元必被
        // 同行/同列线程写过; 若联合守(inr)会漏写 → 读到未初始化 threadgroup
        ta[lr][lc] = (row < pc[0] && ka < pc[2]) ? a[row * pc[3] + ka] : 0.0f;
        const int kb = t * 8 + lr;
        if (pc[6] == 1) {
            tb[lc][lr] = (col < pc[1] && kb < pc[2]) ? b[col * pc[4] + kb] : 0.0f;
        } else {
            tb[lc][lr] = (col < pc[1] && kb < pc[2]) ? b[kb * pc[4] + col] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int k = 0; k < 8; ++k) {
            acc += ta[lr][k] * tb[lc][k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (inr) {
        y[row * pc[5] + col] = acc;
    }
}
)MSL";

// MSL: 任意维规约(每线程沿 dim 累加 K 个, 步长表同 CUDA/Vulkan 版)
const char* kMslReduce = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void reduce(device const float* x [[buffer(0)]],
                   device float* y [[buffer(2)]],
                   constant const int* pc [[buffer(3)]]) {
    // pc[0]=total pc[1]=n_dim pc[2]=s_dim pc[3]=RANK pc[4]=MEAN
    // pc[10+d]=odims[d], pc[20+d]=xstr[d]
    const int o = (int)((thread_position_in_grid).x);
    if (o >= pc[0]) return;
    int t = o;
    int base = 0;
    for (int d = pc[3] - 1; d >= 0; --d) {
        base += (t % pc[10 + d]) * pc[20 + d];
        t /= pc[10 + d];
    }
    float acc = 0.0f;
    for (int j = 0; j < pc[1]; ++j) {
        acc += x[base + j * pc[2]];
    }
    y[o] = (pc[4] == 1) ? acc / (float)pc[1] : acc;
}
)MSL";

// MSL: softmax 沿末维(每线程一行, 数值稳定)
const char* kMslSoftmax = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void softmax(device const float* x [[buffer(0)]],
                    device float* y [[buffer(2)]],
                    constant const int* pc [[buffer(3)]]) {
    const int r = (int)((thread_position_in_grid).x);
    if (r >= pc[0]) return;
    const int base = r * pc[1];
    float m = -3.4e38f;
    for (int j = 0; j < pc[1]; ++j) {
        const float v = x[base + j];
        if (v > m) m = v;
    }
    float s = 0.0f;
    for (int j = 0; j < pc[1]; ++j) {
        s += exp(x[base + j] - m);
    }
    for (int j = 0; j < pc[1]; ++j) {
        y[base + j] = exp(x[base + j] - m) / s;
    }
}
)MSL";

void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(std::string("znn metal: ") + what);
}

// 通用逐元素 host 入口(op 编码与 CUDA/Vulkan 一致; 二元全维广播)
void elementwise(Span<const Tensor*> in, Span<Tensor*> out, int op) {
    Tensor a = in[0]->is_contiguous() ? *in[0] : in[0]->clone();   // 视图材料化
    Tensor& y = *out[0];
    static id<MTLComputePipelineState> p = nil;   // 管线复用(同源码)
    if (!p) p = device::pipeline(kMslElementwise, "elementwise");
    int pc[40] = {0};
    pc[0] = (int)y.numel();
    pc[1] = op;
    if (op <= 3) {                                // 二元: 广播步长图
        Tensor b = in[1]->is_contiguous() ? *in[1] : in[1]->clone();
        pc[2] = (int)y.dim_count();
        int64_t ca = 1, cb = 1;
        for (int64_t d = y.dim_count() - 1; d >= 0; --d) {
            const int64_t ad = d - (int64_t)(y.dim_count() - a.dim_count());
            const int64_t bd = d - (int64_t)(y.dim_count() - b.dim_count());
            const int64_t av = ad < 0 ? 1 : a.size(ad);
            const int64_t bv = bd < 0 ? 1 : b.size(bd);
            if ((av != y.size(d) && av != 1) || (bv != y.size(d) && bv != 1)) {
                throw std::runtime_error("znn: shapes not broadcastable");
            }
            pc[10 + d] = (int)y.size(d);
            pc[20 + d] = av == 1 ? 0 : (int)ca;
            pc[30 + d] = bv == 1 ? 0 : (int)cb;
            ca *= av;
            cb *= bv;
        }
        const uint32_t n = (uint32_t)y.numel();
        device::dispatch(p, mbuf(a), mbuf(b), mbuf(y), pc, sizeof(pc),
                         MTLSizeMake(rup(n, 256), 1, 1), MTLSizeMake(256, 1, 1), nil);
        return;
    }
    require(a.is_contiguous() && y.is_contiguous(), "elementwise needs contiguous");
    pc[2] = 1;                                    // 一元: 平凡图
    pc[10] = pc[0];
    pc[20] = 1;
    pc[30] = 1;
    const uint32_t n = (uint32_t)y.numel();
    device::dispatch(p, mbuf(a), mbuf(a), mbuf(y), pc, sizeof(pc),
                     MTLSizeMake(rup(n, 256), 1, 1), MTLSizeMake(256, 1, 1), nil);
}

} // namespace

// ---- 逐元素 13 算子 ----
void add_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 0);
}
void sub_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 1);
}
void mul_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 2);
}
void div_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 3);
}
void neg_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 4);
}
void exp_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 5);
}
void log_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 6);
}
void sqrt_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 7);
}
void relu_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 8);
}
void gelu_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 9);
}
void sigmoid_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 10);
}
void tanh_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 11);
}
void silu_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 12);
}

// ---- GEMM / linear ----
namespace {
void gemm_run(const Tensor& a, const Tensor& b, Tensor& y,
              int M, int N, int K, int lda, int ldb, int tb) {
    static id<MTLComputePipelineState> p = nil;
    if (!p) p = device::pipeline(kMslGemm, "gemm");
    const int pc[7] = {M, N, K, lda, ldb, (int)y.size(1), tb};
    device::dispatch(p, mbuf(a), mbuf(b), mbuf(y), pc, sizeof(pc),
                     MTLSizeMake(rup((uint32_t)N, 8), rup((uint32_t)M, 8), 1),
                     MTLSizeMake(8, 8, 1), nil);
}
} // namespace

void matmul_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    const Tensor& a = *in[0];
    const Tensor& b = *in[1];
    Tensor& y = *out[0];
    require(a.is_contiguous() && b.is_contiguous() && y.is_contiguous(),
            "matmul needs contiguous");
    gemm_run(a, b, y, (int)a.size(0), (int)b.size(1), (int)a.size(1),
             (int)a.size(1), (int)b.size(1), 0);
}

void linear_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                  const void*) {
    const Tensor& x = *in[0];
    const Tensor& w = *in[1];
    Tensor& y = *out[0];
    require(x.is_contiguous() && w.is_contiguous() && y.is_contiguous(),
            "linear needs contiguous");
    require(in.size() <= 2 || !in[2],
            "linear bias use add op composition");
    gemm_run(x, w, y, (int)x.size(0), (int)w.size(0), (int)x.size(1),
             (int)x.size(1), (int)w.size(1), 1);
}

// ---- 规约 / softmax ----
namespace {
void reduce_run(Span<const Tensor*> in, Span<Tensor*> out, int64_t dim, bool mean) {
    const Tensor& x = *in[0];
    Tensor& y = *out[0];
    require(x.is_contiguous() && y.is_contiguous(), "reduce needs contiguous");
    if (dim < 0) dim += x.dim_count();
    static id<MTLComputePipelineState> p = nil;
    if (!p) p = device::pipeline(kMslReduce, "reduce");
    int pc[40] = {0};                       // total,n,s,RANK,MEAN + 表
    pc[0] = (int)y.numel();
    pc[1] = (int)x.size(dim);
    pc[2] = (int)x.strides()[dim];
    pc[3] = (int)(x.dim_count() - 1);
    pc[4] = mean ? 1 : 0;
    int r = 0;
    for (int64_t d = 0; d < x.dim_count(); ++d) {
        if (d == dim) continue;
        pc[10 + r] = (int)x.size(d);
        pc[20 + r] = (int)x.strides()[d];
        ++r;
    }
    const uint32_t total = (uint32_t)pc[0];
    device::dispatch(p, mbuf(x), mbuf(x), mbuf(y), pc, sizeof(pc),
                     MTLSizeMake(rup(total, 256), 1, 1), MTLSizeMake(256, 1, 1), nil);
}
} // namespace

void sum_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
               const void* options) {
    struct RA { int64_t dim; bool keepdim; };
    reduce_run(in, out, static_cast<const RA*>(options)->dim, false);
}

void mean_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                const void* options) {
    struct RA { int64_t dim; bool keepdim; };
    reduce_run(in, out, static_cast<const RA*>(options)->dim, true);
}

void softmax_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                   const void* options) {
    const Tensor& x = *in[0];
    Tensor& y = *out[0];
    require(x.is_contiguous() && y.is_contiguous(), "softmax needs contiguous");
    struct SO { int64_t dim; };
    const int64_t dim = static_cast<const SO*>(options)->dim;
    const int64_t d = dim < 0 ? dim + x.dim_count() : dim;
    if (d != x.dim_count() - 1) {
        throw std::runtime_error("znn metal: softmax last-dim only (fallback)");
    }
    static id<MTLComputePipelineState> p = nil;
    if (!p) p = device::pipeline(kMslSoftmax, "softmax");
    const int pc[2] = {(int)(x.numel() / x.size(d)), (int)x.size(d)};
    const uint32_t rows = (uint32_t)pc[0];
    device::dispatch(p, mbuf(x), mbuf(x), mbuf(y), pc, sizeof(pc),
                     MTLSizeMake(rup(rows, 256), 1, 1), MTLSizeMake(256, 1, 1), nil);
}

// ---- 直连内核(unary_bwd 经注册表槽位; 其余经 backend_slots 宏直调) ----
namespace {

// MSL: 一元反向 gi = g × f'(x, y); FN 0Neg..8Silu(与 CUDA unary_deriv_d 同表)
const char* kMslUnaryBwd = R"MSL(
#include <metal_stdlib>
using namespace metal;

float deriv(int fn, float xv, float yv) {
    if (fn == 0) return -1.0f;
    if (fn == 1) return yv;
    if (fn == 2) return 1.0f / xv;
    if (fn == 3) return 0.5f / yv;
    if (fn == 4) return xv > 0.0f ? 1.0f : 0.0f;
    if (fn == 5) {
        const float inner = 0.7978845608f * (xv + 0.044715f * xv * xv * xv);
        const float t = tanh(inner);
        const float dt = (1.0f - t * t) * 0.7978845608f
                       * (1.0f + 3.0f * 0.044715f * xv * xv);
        return 0.5f * (1.0f + t) + 0.5f * xv * dt;
    }
    if (fn == 6) return yv * (1.0f - yv);
    if (fn == 7) return 1.0f - yv * yv;
    const float sg = 1.0f / (1.0f + exp(-xv));   // Silu
    return sg + xv * sg * (1.0f - sg);
}

kernel void unarybwd(device const float* g [[buffer(0)]],
                     device const float* x [[buffer(1)]],
                     device const float* yo [[buffer(2)]],
                     device float* gi [[buffer(3)]],
                     constant const int* pc [[buffer(8)]]) {
    const int i = (int)((thread_position_in_grid).x);
    if (i >= pc[0]) return;
    gi[i] = g[i] * deriv(pc[1], x[i], yo[i]);
}
)MSL";

// MSL: 双步长拷贝(reshape/slice/transpose/permute/cat/stack 底层原语)
const char* kMslCopy = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void stridedcopy(device const float* s [[buffer(0)]],
                        device float* d [[buffer(1)]],
                        constant const int* pc [[buffer(8)]]) {
    // pc[0]=total pc[1]=rank pc[10+d]=ss pc[20+d]=ds pc[30+d]=sh
    const int i = (int)((thread_position_in_grid).x);
    if (i >= pc[0]) return;
    int t = i, so = 0, dof = 0;
    for (int d2 = pc[1] - 1; d2 >= 0; --d2) {
        const int idx = t % pc[30 + d2];
        t /= pc[30 + d2];
        so += idx * pc[10 + d2];
        dof += idx * pc[20 + d2];
    }
    d[dof] = s[so];
}
)MSL";

// MSL: 梯度广播/置换 out[i] = scale × g[Σ idx·eff]
const char* kMslBcastGrad = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void bcastgrad(device const float* g [[buffer(0)]],
                      device float* outv [[buffer(1)]],
                      constant const int* pc [[buffer(8)]]) {
    // pc[0]=total pc[1]=rank pc[10+d]=dims pc[20+d]=eff; scale 存 pc[30] 位型
    const int i = (int)((thread_position_in_grid).x);
    if (i >= pc[0]) return;
    int t = i, go = 0;
    for (int d2 = pc[1] - 1; d2 >= 0; --d2) {
        go += (t % pc[10 + d2]) * pc[20 + d2];
        t /= pc[10 + d2];
    }
    const float scale = *(constant float*)(&pc[30]);
    outv[i] = scale * g[go];
}
)MSL";

// MSL: 优化器步进(0=SGD momentum / 1=Adam, 与 CUDA sgd_k/adam_k 同构)
const char* kMslOpt = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void optstep(device float* p [[buffer(0)]],
                    device const float* g [[buffer(1)]],
                    device float* u1 [[buffer(2)]],
                    device float* u2 [[buffer(3)]],
                    constant const int* pc [[buffer(8)]]) {
    // pc[0]=n pc[1]=opt; pc[2..] 存 float 位型(lr,p1..p6)
    const int i = (int)((thread_position_in_grid).x);
    if (i >= pc[0]) return;
    const constant float* pf = (constant float*)&pc[2];
    const float lr = pf[0];
    if (pc[1] == 0) {
        if (pf[1] > 0.0f) {
            u1[i] = pf[1] * u1[i] + g[i] + pf[2] * p[i];
            p[i] -= lr * u1[i];
        } else {
            p[i] -= lr * (g[i] + pf[2] * p[i]);
        }
    } else {
        const float grad = g[i] + pf[4] * p[i];
        u1[i] = pf[1] * u1[i] + (1.0f - pf[1]) * grad;
        u2[i] = pf[2] * u2[i] + (1.0f - pf[2]) * grad * grad;
        const float mh = u1[i] / pf[5];
        const float vh = u2[i] / pf[6];
        p[i] -= lr * mh / (sqrt(vh) + pf[3]);
    }
}
)MSL";

} // namespace

void unary_bwd_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                     const void* options) {
    static id<MTLComputePipelineState> p = nil;
    if (!p) p = device::pipeline(kMslUnaryBwd, "unarybwd");
    const int fn = (int)*static_cast<const unsigned char*>(options);
    const int pc[2] = {(int)out[0]->numel(), fn};
    id<MTLBuffer> bufs[4] = {mbuf(*in[0]), mbuf(*in[1]), mbuf(*in[2]),
                             mbuf(*out[0])};
    const uint32_t n = (uint32_t)pc[0];
    device::dispatch_bufs(p, bufs, 4, pc, sizeof(pc),
                          MTLSizeMake(rup(n, 256), 1, 1), MTLSizeMake(256, 1, 1));
}

namespace {
void copy_pc_fill(int* pc, long long total, int rank, const long long* ss,
                  const long long* ds, const long long* sh) {
    pc[0] = (int)total;
    pc[1] = rank;
    for (int d = 0; d < rank; ++d) {
        pc[10 + d] = (int)ss[d];
        pc[20 + d] = (int)ds[d];
        pc[30 + d] = (int)sh[d];
    }
}
} // namespace

void grad_bcast_metal(const float* g, float* out, long long total, int rank,
                      const long long* dims, const long long* eff, float scale) {
    static id<MTLComputePipelineState> p = nil;
    if (!p) p = device::pipeline(kMslBcastGrad, "bcastgrad");
    int pc[34] = {0};
    pc[0] = (int)total;
    pc[1] = rank;
    for (int d = 0; d < rank; ++d) {
        pc[10 + d] = (int)dims[d];
        pc[20 + d] = (int)eff[d];
    }
    std::memcpy(&pc[30], &scale, sizeof(scale));   // 尾部 float 位型
    id<MTLBuffer> bufs[2] = {device::buf_of(const_cast<float*>(g)),
                             device::buf_of(out)};
    const uint32_t n = (uint32_t)total;
    device::dispatch_bufs(p, bufs, 2, pc, sizeof(pc),
                          MTLSizeMake(rup(n, 256), 1, 1), MTLSizeMake(256, 1, 1));
}

void strided_copy_metal(const float* src, float* dst, int rank,
                        const long long* ss, const long long* ds,
                        const long long* sh, long long total) {
    static id<MTLComputePipelineState> p = nil;
    if (!p) p = device::pipeline(kMslCopy, "stridedcopy");
    int pc[40] = {0};
    copy_pc_fill(pc, total, rank, ss, ds, sh);
    id<MTLBuffer> bufs[2] = {device::buf_of(const_cast<float*>(src)),
                             device::buf_of(dst)};
    const uint32_t n = (uint32_t)total;
    device::dispatch_bufs(p, bufs, 2, pc, sizeof(pc),
                          MTLSizeMake(rup(n, 256), 1, 1), MTLSizeMake(256, 1, 1));
}

void sgd_step_metal(float* p, const float* g, float* vel, long long n,
                    float lr, float momentum, float wd) {
    static id<MTLComputePipelineState> pl = nil;
    if (!pl) pl = device::pipeline(kMslOpt, "optstep");
    int pc[9] = {0};
    pc[0] = (int)n;
    pc[1] = 0;
    const float pf[3] = {lr, momentum, wd};
    std::memcpy(&pc[2], pf, sizeof(pf));
    id<MTLBuffer> bufs[4] = {device::buf_of(p),
                             device::buf_of(const_cast<float*>(g)),
                             device::buf_of(vel), device::buf_of(vel)};
    device::dispatch_bufs(pl, bufs, 4, pc, sizeof(pc),
                          MTLSizeMake(rup((uint32_t)n, 256), 1, 1),
                          MTLSizeMake(256, 1, 1));
}

void adam_step_metal(float* p, const float* g, float* m, float* v,
                     long long n, float lr, float b1, float b2, float eps,
                     float wd, float bias1, float bias2) {
    static id<MTLComputePipelineState> pl = nil;
    if (!pl) pl = device::pipeline(kMslOpt, "optstep");
    int pc[9] = {0};
    pc[0] = (int)n;
    pc[1] = 1;
    const float pf[7] = {lr, b1, b2, eps, wd, bias1, bias2};
    std::memcpy(&pc[2], pf, sizeof(pf));
    id<MTLBuffer> bufs[4] = {device::buf_of(p),
                             device::buf_of(const_cast<float*>(g)),
                             device::buf_of(m), device::buf_of(v)};
    device::dispatch_bufs(pl, bufs, 4, pc, sizeof(pc),
                          MTLSizeMake(rup((uint32_t)n, 256), 1, 1),
                          MTLSizeMake(256, 1, 1));
}

} // namespace metal_backend
} // namespace znn

#endif // __APPLE__ && ZNN_METAL
