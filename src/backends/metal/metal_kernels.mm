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
#include "operators/shared.hpp"
#include "operators/batch_norm_options.hpp"
#include <cmath>
#include <random>
#include <vector>

namespace znn {
namespace metal_backend {

namespace {

device::MBuf mbuf(const Tensor& t) {
    return device::MBuf{device::buf_of(t.data<Tensor::MetalDataPtr>()),
                        (uint64_t)(t.storage_offset() * 4)};
}

uint32_t rup(uint32_t n, uint32_t a) { return (n + a - 1) / a * a; }  // 向上取整到组宽

// MSL: 通用逐元素(13 算子共用; 二元全维广播, 右对齐步长表)
const char* kMslElementwise = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void elementwise(device const float* a [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        device float* y [[buffer(2)]],
                        constant const int* pc [[buffer(3)]], uint3 tpig [[thread_position_in_grid]]) {
    // pc[0]=n pc[1]=op pc[2]=rank
    // pc[10+d]=dims pc[20+d]=sa pc[30+d]=sb(维长 1 → 步长 0)
    const int i = (int)(tpig.x);
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
                 constant const int* pc [[buffer(3)]], uint3 tpig [[thread_position_in_grid]], uint3 tpit [[thread_position_in_threadgroup]]) {
    // pc = {M, N, K, lda, ldb, ldc, TB}
    const int row = (int)(tpig.y);
    const int col = (int)(tpig.x);
    const int lr = (int)(tpit.y);
    const int lc = (int)(tpit.x);
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

// MSL: 任意维规约(每 threadgroup 一输出, 线程沿 dim 协作 + 组内存树归约;
// 与 CUDA/Vulkan 版同构 —— 此前一线程一输出, 少输出场景整卡欠载)
const char* kMslReduce = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void reduce(device const float* x [[buffer(0)]],
                   device float* y [[buffer(2)]],
                   constant const int* pc [[buffer(3)]],
                   uint3 tpig [[thread_position_in_grid]],
                   uint3 tptg [[thread_position_in_threadgroup]]) {
    // pc[0]=total pc[1]=n_dim pc[2]=s_dim pc[3]=RANK pc[4]=MEAN
    // pc[10+d]=odims[d], pc[20+d]=xstr[d]
    const int out = (int)(tpig.x) / 256;      // 每组一输出 → 组号 = 线程序/256
    if (out >= pc[0]) return;
    int t = out;
    int base = 0;
    for (int d = pc[3] - 1; d >= 0; --d) {
        base += (t % pc[10 + d]) * pc[20 + d];
        t /= pc[10 + d];
    }
    threadgroup float sh[256];
    const int lane = (int)(tptg.x);
    float acc = 0.0f;
    for (int j = lane; j < pc[1]; j += 256) {
        acc += x[base + j * pc[2]];
    }
    sh[lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 128; s > 0; s >>= 1) {
        if (lane < s) sh[lane] += sh[lane + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        y[out] = (pc[4] == 1) ? sh[0] / (float)pc[1] : sh[0];
    }
}
)MSL";

// MSL: 任意 dim 的 softmax(每 threadgroup 一行, 组内协作 max/sum; 数值稳定)
const char* kMslSoftmaxDim = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void softmax_dim(device const float* x [[buffer(0)]],
                        device float* y [[buffer(2)]],
                        constant const int* pc [[buffer(3)]],
                        uint3 tpig [[thread_position_in_grid]],
                        uint3 tptg [[thread_position_in_threadgroup]]) {
    // pc[0]=rows pc[1]=n_dim pc[2]=s_dim pc[3]=RANK
    // pc[10+d]=odims[d], pc[20+d]=xstr[d]
    const int r = (int)(tpig.x) / 256;        // 每组一行 → 组号 = 线程序/256
    if (r >= pc[0]) return;
    int t = r;
    int base = 0;
    for (int d = pc[3] - 1; d >= 0; --d) {
        base += (t % pc[10 + d]) * pc[20 + d];
        t /= pc[10 + d];
    }
    threadgroup float sh[256];
    const int lane = (int)(tptg.x);
    float m = -3.4e38f;
    for (int j = lane; j < pc[1]; j += 256) {
        m = fmax(m, x[base + j * pc[2]]);
    }
    sh[lane] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 128; s > 0; s >>= 1) {
        if (lane < s) sh[lane] = fmax(sh[lane], sh[lane + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mx = sh[0];
    float acc = 0.0f;
    for (int j = lane; j < pc[1]; j += 256) {
        const float e = exp(x[base + j * pc[2]] - mx);
        y[base + j * pc[2]] = e;
        acc += e;
    }
    sh[lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int s = 128; s > 0; s >>= 1) {
        if (lane < s) sh[lane] += sh[lane + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float sum = sh[0];
    for (int j = lane; j < pc[1]; j += 256) {
        y[base + j * pc[2]] /= sum;
    }
}
)MSL";

// MSL: softmax 沿末维(每线程一行, 数值稳定)
const char* kMslSoftmax = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void softmax(device const float* x [[buffer(0)]],
                    device float* y [[buffer(2)]],
                    constant const int* pc [[buffer(3)]], uint3 tpig [[thread_position_in_grid]]) {
    const int r = (int)(tpig.x);
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

const char* kMslAttention = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct AttentionPc { int bh; int L; int D; float scale; int causal; };
kernel void attention_fused(device const float* q [[buffer(0)]],
                            device const float* k [[buffer(1)]],
                            device const float* v [[buffer(2)]],
                            device float* o [[buffer(3)]],
                            constant AttentionPc& pc [[buffer(8)]],
                            uint id [[thread_position_in_grid]]) {
    const int total = pc.bh * pc.L * pc.D;
    if ((int)id >= total) return;
    const int head = (int)id / (pc.L * pc.D);
    const int rem = (int)id % (pc.L * pc.D);
    const int row = rem / pc.D, col = rem % pc.D;
    const int limit = pc.causal ? row + 1 : pc.L;
    const int qb = (head * pc.L + row) * pc.D;
    float m = -3.4e38f;
    for (int j = 0; j < limit; ++j) {
        float s = 0.0f;
        const int kb = (head * pc.L + j) * pc.D;
        for (int d = 0; d < pc.D; ++d) s += q[qb + d] * k[kb + d];
        m = max(m, s * pc.scale);
    }
    float den = 0.0f, acc = 0.0f;
    for (int j = 0; j < limit; ++j) {
        float s = 0.0f;
        const int kb = (head * pc.L + j) * pc.D;
        for (int d = 0; d < pc.D; ++d) s += q[qb + d] * k[kb + d];
        const float w = exp(s * pc.scale - m);
        den += w;
        acc += w * v[(head * pc.L + j) * pc.D + col];
    }
    o[id] = acc / den;
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
    Tensor xm = in[0]->is_contiguous() ? *in[0] : in[0]->clone();   // 视图材料化
    const Tensor& x = xm;
    Tensor& y = *out[0];
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
                     MTLSizeMake(total, 1, 1), MTLSizeMake(256, 1, 1), nil);
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

const char* kMslUpsample = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void upsample2x(device const float* x [[buffer(0)]],
                       device float* y [[buffer(2)]],
                       constant const int* pc [[buffer(3)]],
                       uint3 tpig [[thread_position_in_grid]]) {
    // pc[0]=total pc[1]=w_in pc[2]=w_out; 每线程写 2x2 块
    const int i = (int)(tpig.x);
    if (i >= pc[0]) return;
    const int w = i % pc[1];
    const int r = i / pc[1];
    const float v = x[i];
    const int base = r * 2 * pc[2] + w * 2;
    y[base] = v;
    y[base + 1] = v;
    y[base + pc[2]] = v;
    y[base + pc[2] + 1] = v;
}
)MSL";

void upsample2x_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                      const void*) {
    Tensor xm = in[0]->is_contiguous() ? *in[0] : in[0]->clone();
    const Tensor& x = xm;
    static id<MTLComputePipelineState> p = nil;
    if (!p) p = device::pipeline(kMslUpsample, "upsample2x");
    const int pc[3] = {(int)x.numel(), (int)x.size(3), (int)(x.size(3) * 2)};
    const device::MBuf bufs[2] = {mbuf(x), mbuf(*out[0])};
    device::dispatch_bufs(p, bufs, 2, pc, sizeof(pc),
                          MTLSizeMake(rup((uint32_t)pc[0], 256), 1, 1),
                          MTLSizeMake(256, 1, 1));
}

void softmax_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                   const void* options) {
    Tensor xm = in[0]->is_contiguous() ? *in[0] : in[0]->clone();   // 视图材料化
    const Tensor& x = xm;
    Tensor& y = *out[0];
    struct SO { int64_t dim; };
    const int64_t dim = static_cast<const SO*>(options)->dim;
    const int64_t d = dim < 0 ? dim + x.dim_count() : dim;
    if (d == x.dim_count() - 1) {             // 末维: 每线程一行
        static id<MTLComputePipelineState> p = nil;
        if (!p) p = device::pipeline(kMslSoftmax, "softmax");
        const int pc[2] = {(int)(x.numel() / x.size(d)), (int)x.size(d)};
        const uint32_t rows = (uint32_t)pc[0];
        device::dispatch(p, mbuf(x), mbuf(x), mbuf(y), pc, sizeof(pc),
                         MTLSizeMake(rup(rows, 256), 1, 1), MTLSizeMake(256, 1, 1), nil);
        return;
    }
    // 一般维: 每 threadgroup 一行(此前直接抛错降级 —— 现原生)
    static id<MTLComputePipelineState> pd = nil;
    if (!pd) pd = device::pipeline(kMslSoftmaxDim, "softmax_dim");
    int pc[40] = {0};
    pc[1] = (int)x.size(d);
    pc[2] = (int)x.strides()[d];
    pc[3] = (int)(x.dim_count() - 1);
    pc[0] = 1;
    int r = 0;
    for (int64_t k = 0; k < x.dim_count(); ++k) {
        if (k == d) continue;
        pc[10 + r] = (int)x.size(k);
        pc[20 + r] = (int)x.strides()[k];
        ++r;
        pc[0] *= (int)x.size(k);
    }
    device::dispatch(pd, mbuf(x), mbuf(x), mbuf(y), pc, sizeof(pc),
                     MTLSizeMake((uint32_t)pc[0], 1, 1), MTLSizeMake(256, 1, 1), nil);
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
                     constant const int* pc [[buffer(8)]], uint3 tpig [[thread_position_in_grid]]) {
    const int i = (int)(tpig.x);
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
                        constant const int* pc [[buffer(8)]], uint3 tpig [[thread_position_in_grid]]) {
    // pc[0]=total pc[1]=rank pc[10+d]=ss pc[20+d]=ds pc[30+d]=sh
    const int i = (int)(tpig.x);
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
                      constant const int* pc [[buffer(8)]], uint3 tpig [[thread_position_in_grid]]) {
    // pc[0]=total pc[1]=rank pc[10+d]=dims pc[20+d]=eff; scale 存 pc[30] 位型
    const int i = (int)(tpig.x);
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
                    constant const int* pc [[buffer(8)]], uint3 tpig [[thread_position_in_grid]]) {
    // pc[0]=n pc[1]=opt; pc[2..] 存 float 位型(lr,p1..p7)
    const int i = (int)(tpig.x);
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
        p[i] = p[i] * (1.0f - lr * pf[7]) - lr * mh / (sqrt(vh) + pf[3]);
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
    const device::MBuf bufs[4] = {mbuf(*in[0]), mbuf(*in[1]), mbuf(*in[2]),
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

void grad_bcast_metal(const Tensor& g, Tensor& out, long long total, int rank,
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
    const device::MBuf bufs[2] = {mbuf(g), mbuf(out)};
    const uint32_t n = (uint32_t)total;
    device::dispatch_bufs(p, bufs, 2, pc, sizeof(pc),
                          MTLSizeMake(rup(n, 256), 1, 1), MTLSizeMake(256, 1, 1));
}

void strided_copy_metal(const Tensor& src, Tensor& dst, int rank,
                        const long long* ss, const long long* ds,
                        const long long* sh, long long total) {
    static id<MTLComputePipelineState> p = nil;
    if (!p) p = device::pipeline(kMslCopy, "stridedcopy");
    int pc[40] = {0};
    copy_pc_fill(pc, total, rank, ss, ds, sh);
    const device::MBuf bufs[2] = {mbuf(src), mbuf(dst)};
    const uint32_t n = (uint32_t)total;
    device::dispatch_bufs(p, bufs, 2, pc, sizeof(pc),
                          MTLSizeMake(rup(n, 256), 1, 1), MTLSizeMake(256, 1, 1));
}

void strided_copy_raw_mt(const void* src, uint64_t soff, void* dst, uint64_t doff,
                         int rank, const long long* ss, const long long* ds,
                         const long long* sh, long long total) {
    static id<MTLComputePipelineState> p = nil;
    if (!p) p = device::pipeline(kMslCopy, "stridedcopy");
    int pc[40] = {0};
    copy_pc_fill(pc, total, rank, ss, ds, sh);
    const device::MBuf bufs[2] = {{device::buf_of(const_cast<void*>(src)), soff},
                                  {device::buf_of(dst), doff}};
    const uint32_t n = (uint32_t)total;
    device::dispatch_bufs(p, bufs, 2, pc, sizeof(pc),
                          MTLSizeMake(rup(n, 256), 1, 1), MTLSizeMake(256, 1, 1));
}

void sgd_step_metal(const Tensor& p, const Tensor& g, const Tensor& vel,
                    long long n, float lr, float momentum, float wd) {
    static id<MTLComputePipelineState> pl = nil;
    if (!pl) pl = device::pipeline(kMslOpt, "optstep");
    int pc[9] = {0};
    pc[0] = (int)n;
    pc[1] = 0;
    const float pf[3] = {lr, momentum, wd};
    std::memcpy(&pc[2], pf, sizeof(pf));
    const device::MBuf bufs[4] = {mbuf(p), mbuf(g), mbuf(vel), mbuf(vel)};
    device::dispatch_bufs(pl, bufs, 4, pc, sizeof(pc),
                          MTLSizeMake(rup((uint32_t)n, 256), 1, 1),
                          MTLSizeMake(256, 1, 1));
}

void adam_step_metal(const Tensor& p, const Tensor& g, const Tensor& m,
                     const Tensor& v, long long n, float lr, float b1,
                     float b2, float eps, float wd, float decay, float bias1, float bias2) {
    static id<MTLComputePipelineState> pl = nil;
    if (!pl) pl = device::pipeline(kMslOpt, "optstep");
    int pc[10] = {0};
    pc[0] = (int)n;
    pc[1] = 1;
    const float pf[8] = {lr, b1, b2, eps, wd, bias1, bias2, decay};
    std::memcpy(&pc[2], pf, sizeof(pf));
    const device::MBuf bufs[4] = {mbuf(p), mbuf(g), mbuf(m), mbuf(v)};
    device::dispatch_bufs(pl, bufs, 4, pc, sizeof(pc),
                          MTLSizeMake(rup((uint32_t)n, 256), 1, 1),
                          MTLSizeMake(256, 1, 1));
}

// ============================================================ SB2-SB4: 17 算子

namespace {
// Metal 共享内存: 句柄→mapped 指针可直接 memcpy(d2d 免命令缓冲)
void* mh(const Tensor& t) {
    return device::mapped_of(t.data<Tensor::MetalDataPtr>());
}
} // namespace

// ---- MSL: 填充(arange) + 嵌入(前向收集 / CAS 梯度散射) ----
const char* kMslFill = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void fillarange(device float* y [[buffer(0)]],
                       constant const int* pc [[buffer(8)]], uint3 tpig [[thread_position_in_grid]]) {
    const int i = (int)(tpig.x);
    if (i >= pc[0]) return;
    y[i] = (float)i;
}

kernel void embedfwd(device const float* w [[buffer(0)]],
                     device const ulong* idx [[buffer(1)]],
                     device float* y [[buffer(2)]],
                     constant const int* pc [[buffer(8)]], uint3 tpig [[thread_position_in_grid]]) {
    const int i = (int)(tpig.x);
    if (i >= pc[0]) return;
    const int v = (int)idx[i / pc[1]];
    y[i] = (v >= 0 && v < pc[2]) ? w[v * pc[1] + i % pc[1]] : 0.0f;
}

kernel void embedgrad(device const float* g [[buffer(0)]],
                      device const ulong* idx [[buffer(1)]],
                      device float* dw [[buffer(2)]],
                      device atomic_uint* dwu [[buffer(3)]],
                      constant const int* pc [[buffer(8)]], uint3 tpig [[thread_position_in_grid]]) {
    const int i = (int)(tpig.x);
    if (i >= pc[0]) return;
    const int v = (int)idx[i / pc[1]];
    const uint slot = (uint)(v * pc[1] + i % pc[1]);
    const float add = g[i];
    uint old = atomic_load_explicit(dwu + slot, memory_order_relaxed);
    for (;;) {
        const uint neu = as_type<uint>(as_type<float>(old) + add);
        uint expected = old;
        if (atomic_compare_exchange_weak_explicit(dwu + slot, &expected, neu,
                memory_order_relaxed, memory_order_relaxed)) break;
        old = expected;
    }
    (void)dw;
}
)MSL";

// ---- MSL: LN(每 threadgroup 一行) + BN(每 threadgroup 一通道) ----
const char* kMslNorm = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void lnrow(device const float* x [[buffer(0)]],
                  device const float* w [[buffer(1)]],
                  device const float* b [[buffer(2)]],
                  device float* y [[buffer(5)]],
                  constant const int* pc [[buffer(8)]],
                  threadgroup float* sm [[threadgroup(0)]],
                  threadgroup float* sv [[threadgroup(1)]], uint3 tpig [[thread_position_in_grid]], uint3 tpit [[thread_position_in_threadgroup]]) {
    const int r = (int)(tpig.y);
    if (r >= pc[0]) return;
    const int lid = (int)(tpit.x);
    const int base = r * pc[1];
    float ls = 0.0f, lq = 0.0f;
    for (int i = lid; i < pc[1]; i += 256) {
        ls += x[base + i];
        lq += x[base + i] * x[base + i];
    }
    sm[lid] = ls; sv[lid] = lq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        float s = 0.0f, q = 0.0f;
        for (int i = 0; i < 256; ++i) { s += sm[i]; q += sv[i]; }
        sm[0] = s / (float)pc[1];
        sv[0] = max(q / (float)pc[1] - sm[0] * sm[0], 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = rsqrt(sv[0] + as_type<float>(pc[4]));
    for (int i = lid; i < pc[1]; i += 256) {
        y[base + i] = (x[base + i] - sm[0]) * inv * w[i] + b[i];
    }
}

kernel void lngroup(device const float* x [[buffer(0)]],
 device const float* w [[buffer(1)]], device const float* b [[buffer(2)]],
 device float* y [[buffer(5)]], constant const int* pc [[buffer(8)]],
 threadgroup float* sm [[threadgroup(0)]], threadgroup float* sv [[threadgroup(1)]],
 uint3 group [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]]) {
 int lid=int(tid.x), lane=lid%32, part=lid/32, g=int(group.x)*32+lane;
 int base=g/pc[3]*pc[1]*pc[3]+g%pc[3];
 float s=0,q=0;
 if(g<pc[0])for(int d=part;d<pc[1];d+=8){float v=x[base+d*pc[3]];s+=v;q+=v*v;}
 sm[lid]=s;sv[lid]=q;threadgroup_barrier(mem_flags::mem_threadgroup);
 s=0;q=0;for(int j=0;j<8;++j){s+=sm[j*32+lane];q+=sv[j*32+lane];}
 float mu=s/float(pc[1]),inv=rsqrt(max(q/float(pc[1])-mu*mu,0.0f)+as_type<float>(pc[4]));
 if(g<pc[0])for(int d=part;d<pc[1];d+=8)y[base+d*pc[3]]=(x[base+d*pc[3]]-mu)*inv*w[d]+b[d];
}

kernel void bnchannel(device const float* x [[buffer(0)]],
                      device const float* w [[buffer(1)]],
                      device const float* b [[buffer(2)]],
                      device float* rm [[buffer(3)]],
                      device float* rv [[buffer(4)]],
                      device float* y [[buffer(5)]],
                      constant const int* pc [[buffer(8)]],
                      threadgroup float* sm [[threadgroup(0)]], uint3 tpig [[thread_position_in_grid]], uint3 tpit [[thread_position_in_threadgroup]]) {
    const int c = (int)(tpig.y);
    if (c >= pc[1]) return;
    const int lid = (int)(tpit.x);
    const int m = pc[0], hw = pc[2] * pc[3];
    float mu, var;
    if (pc[6] != 0) {
        float ls = 0.0f;
        for (int i = lid; i < m; i += 256) {
            ls += x[((i / hw) * pc[1] + c) * hw + i % hw];
        }
        sm[lid] = ls;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float s = 0.0f;
            for (int i = 0; i < 256; ++i) s += sm[i];
            sm[0] = s / (float)m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mu = sm[0];
        float lq = 0.0f;
        for (int i = lid; i < m; i += 256) {
            const float t = x[((i / hw) * pc[1] + c) * hw + i % hw] - mu;
            lq += t * t;
        }
        sm[lid] = lq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0) {
            float s = 0.0f;
            for (int i = 0; i < 256; ++i) s += sm[i];
            const float v = s / (float)m;
            const float momentum = as_type<float>(pc[5]);
            rm[c] = (1.0f - momentum) * rm[c] + momentum * mu;
            rv[c] = (1.0f - momentum) * rv[c] + momentum * v;
            sm[0] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        var = sm[0];
    } else {
        mu = rm[c]; var = rv[c];
    }
    const float inv = rsqrt(var + as_type<float>(pc[4]));
    for (int i = lid; i < m; i += 256) {
        const int off = ((i / hw) * pc[1] + c) * hw + i % hw;
        y[off] = (x[off] - mu) * inv * w[c] + b[c];
    }
}
)MSL";

// ---- MSL: CE + MSE(单 threadgroup 两遍) ----
const char* kMslLoss = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void cekernel(device const float* x [[buffer(0)]],
                     device const ulong* t [[buffer(2)]],
                     device float* y [[buffer(3)]],
                     constant const int* pc [[buffer(8)]],
                     threadgroup float* acc [[threadgroup(0)]], uint3 tpit [[thread_position_in_threadgroup]]) {
    const int lid = (int)(tpit.x);
    float l = 0.0f;
    for (int i = lid; i < pc[0]; i += 256) {
        const int base = i * pc[1];
        float m = x[base];
        for (int j = 1; j < pc[1]; ++j) m = max(m, x[base + j]);
        float s = 0.0f;
        for (int j = 0; j < pc[1]; ++j) s += exp(x[base + j] - m);
        l += m + log(s) - x[base + (int)t[i]];
    }
    acc[lid] = l;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        float s = 0.0f;
        for (int i = 0; i < 256; ++i) s += acc[i];
        y[0] = s / (float)pc[0];
    }
}

kernel void msekernel(device const float* p [[buffer(0)]],
                      device const float* t [[buffer(1)]],
                      device float* y [[buffer(3)]],
                      constant const int* pc [[buffer(8)]],
                      threadgroup float* acc [[threadgroup(0)]], uint3 tpit [[thread_position_in_threadgroup]]) {
    const int lid = (int)(tpit.x);
    float l = 0.0f;
    for (int i = lid; i < pc[0]; i += 256) {
        const float d = p[i] - t[i];
        l += d * d;
    }
    acc[lid] = l;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        float s = 0.0f;
        for (int i = 0; i < 256; ++i) s += acc[i];
        y[0] = s / (float)pc[0];
    }
}
)MSL";

// ---- MSL: conv2d fwd/dw/dx + 池化 4 路(max/avg × fwd/grad CAS) ----
const char* kMslConv = R"MSL(
#include <metal_stdlib>
using namespace metal;

#define CV pc[0],pc[1],pc[2],pc[3],pc[4],pc[5],pc[6],pc[7],pc[8], \
          pc[9],pc[10],pc[11],pc[12],pc[13],pc[14],pc[15],pc[16]

struct ConvPc { int n,c,h,w,o,kh,kw,ho,wo,sh,sw,ph,pw,dh,dw,g; };

kernel void convfwd(device const float* x [[buffer(0)]],
                    device const float* wt [[buffer(1)]],
                    device float* y [[buffer(3)]],
                    constant const int* pcp [[buffer(8)]], uint3 tpig [[thread_position_in_grid]]) {
    const ConvPc pc = *(constant const ConvPc*)pcp;
    const int i = (int)(tpig.x);
    const int total = pc.n * pc.o * pc.ho * pc.wo;
    if (i >= total) return;
    const int wo = i % pc.wo;
    const int t0 = i / pc.wo;
    const int ho = t0 % pc.ho;
    const int t1 = t0 / pc.ho;
    const int oc = t1 % pc.o;
    const int n = t1 / pc.o;
    const int cg = pc.c / pc.g, og = pc.o / pc.g, grp = oc / og;
    float acc = 0.0f;
    for (int c = grp * cg; c < (grp + 1) * cg; ++c) {
        for (int ki = 0; ki < pc.kh; ++ki) {
            const int ii = ho * pc.sh + ki * pc.dh - pc.ph;
            if (ii < 0 || ii >= pc.h) continue;
            for (int kj = 0; kj < pc.kw; ++kj) {
                const int jj = wo * pc.sw + kj * pc.dw - pc.pw;
                if (jj < 0 || jj >= pc.w) continue;
                acc += x[((n * pc.c + c) * pc.h + ii) * pc.w + jj]
                     * wt[(((oc * cg) + (c - grp * cg)) * pc.kh + ki) * pc.kw + kj];
            }
        }
    }
    y[i] = acc;
}

kernel void convdw(device const float* x [[buffer(0)]],
                   device const float* g [[buffer(2)]],
                   device float* dw [[buffer(3)]],
                   constant const int* pcp [[buffer(8)]], uint3 tpig [[thread_position_in_grid]]) {
    const ConvPc pc = *(constant const ConvPc*)pcp;
    const int i = (int)(tpig.x);
    const int total = pc.o * (pc.c / pc.g) * pc.kh * pc.kw;
    if (i >= total) return;
    const int kj = i % pc.kw;
    const int t0 = i / pc.kw;
    const int ki = t0 % pc.kh;
    const int t1 = t0 / pc.kh;
    const int cg = pc.c / pc.g, og = pc.o / pc.g;
    const int cg_in = t1 % cg, oc = t1 / cg, grp = oc / og;
    const int c = grp * cg + cg_in;
    float acc = 0.0f;
    for (int n = 0; n < pc.n; ++n) {
        for (int ho = 0; ho < pc.ho; ++ho) {
            const int ii = ho * pc.sh + ki * pc.dh - pc.ph;
            if (ii < 0 || ii >= pc.h) continue;
            for (int wo = 0; wo < pc.wo; ++wo) {
                const int jj = wo * pc.sw + kj * pc.dw - pc.pw;
                if (jj < 0 || jj >= pc.w) continue;
                acc += g[((n * pc.o + oc) * pc.ho + ho) * pc.wo + wo]
                     * x[((n * pc.c + c) * pc.h + ii) * pc.w + jj];
            }
        }
    }
    dw[i] = acc;
}

kernel void convdx(device const float* g [[buffer(2)]],
                   device const float* wt [[buffer(1)]],
                   device float* dx [[buffer(3)]],
                   constant const int* pcp [[buffer(8)]], uint3 tpig [[thread_position_in_grid]]) {
    const ConvPc pc = *(constant const ConvPc*)pcp;
    const int i = (int)(tpig.x);
    const int total = pc.n * pc.c * pc.h * pc.w;
    if (i >= total) return;
    const int j = i % pc.w;
    const int t0 = i / pc.w;
    const int ii = t0 % pc.h;
    const int t1 = t0 / pc.h;
    const int c = t1 % pc.c;
    const int n = t1 / pc.c;
    const int cg = pc.c / pc.g, og = pc.o / pc.g, grp = c / cg;
    float acc = 0.0f;
    for (int oc = grp * og; oc < (grp + 1) * og; ++oc) {
        for (int ki = 0; ki < pc.kh; ++ki) {
            const int ho_num = ii + pc.ph - ki * pc.dh;
            if (ho_num < 0 || ho_num % pc.sh != 0) continue;
            const int ho = ho_num / pc.sh;
            if (ho >= pc.ho) continue;
            for (int kj = 0; kj < pc.kw; ++kj) {
                const int wo_num = j + pc.pw - kj * pc.dw;
                if (wo_num < 0 || wo_num % pc.sw != 0) continue;
                const int wo = wo_num / pc.sw;
                if (wo >= pc.wo) continue;
                acc += g[((n * pc.o + oc) * pc.ho + ho) * pc.wo + wo]
                     * wt[(((oc * cg) + (c - grp * cg)) * pc.kh + ki) * pc.kw + kj];
            }
        }
    }
    dx[i] = acc;
}
)MSL";

const char* kMslPool = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void poolmax(device const float* x [[buffer(0)]],
                    device float* y [[buffer(2)]],
                    constant const int* pc [[buffer(8)]], uint3 tpig [[thread_position_in_grid]]) {
    const int i = (int)(tpig.x);
    if (i >= pc[0]) return;
    const int wo = i % pc[4];
    const int t0 = i / pc[4];
    const int ho = t0 % pc[3];
    const int nchw = t0 / pc[3];
    const int base = nchw * pc[1] * pc[2];
    float m = -3.4e38f;
    for (int ki = 0; ki < pc[5]; ++ki) {
        const int ii = ho * pc[7] + ki - pc[9];
        if (ii < 0 || ii >= pc[1]) continue;
        for (int kj = 0; kj < pc[6]; ++kj) {
            const int jj = wo * pc[8] + kj - pc[10];
            if (jj < 0 || jj >= pc[2]) continue;
            m = max(m, x[base + ii * pc[2] + jj]);
        }
    }
    y[i] = m;
}

kernel void poolglobalavg(device const float* x [[buffer(0)]],
 device float* y [[buffer(2)]],constant const int* pc [[buffer(8)]],
 threadgroup float* sums [[threadgroup(0)]],uint3 group [[threadgroup_position_in_grid]],uint3 tid [[thread_position_in_threadgroup]]) {
 uint lane=tid.x,channel=group.x;int spatial=pc[1]*pc[2];float sum=0;
 for(int i=int(lane);i<spatial;i+=256)sum+=x[channel*spatial+i];
 sums[lane]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint s=128;s>0;s>>=1){if(lane<s)sums[lane]+=sums[lane+s];threadgroup_barrier(mem_flags::mem_threadgroup);}
 if(lane==0)y[channel]=sums[0]/float(spatial);
}

kernel void poolavg(device const float* x [[buffer(0)]],
                    device float* y [[buffer(2)]],
                    constant const int* pc [[buffer(8)]], uint3 tpig [[thread_position_in_grid]]) {
    const int i = (int)(tpig.x);
    if (i >= pc[0]) return;
    const int wo = i % pc[4];
    const int t0 = i / pc[4];
    const int ho = t0 % pc[3];
    const int nchw = t0 / pc[3];
    const int base = nchw * pc[1] * pc[2];
    float acc = 0.0f;
    int cnt = 0;
    for (int ki = 0; ki < pc[5]; ++ki) {
        const int ii = ho * pc[7] + ki - pc[9];
        if (ii < 0 || ii >= pc[1]) continue;
        for (int kj = 0; kj < pc[6]; ++kj) {
            const int jj = wo * pc[8] + kj - pc[10];
            if (jj < 0 || jj >= pc[2]) continue;
            acc += x[base + ii * pc[2] + jj];
            ++cnt;
        }
    }
    y[i] = cnt > 0 ? acc / (float)cnt : 0.0f;
}

kernel void poolmaxgrad(device const float* g [[buffer(0)]],
                        device const float* x [[buffer(1)]],
                        device float* dx [[buffer(2)]],
                        device atomic_uint* dxu [[buffer(3)]],
                        constant const int* pc [[buffer(8)]], uint3 tpig [[thread_position_in_grid]]) {
    const int i = (int)(tpig.x);
    if (i >= pc[0]) return;
    const int wo = i % pc[4];
    const int t0 = i / pc[4];
    const int ho = t0 % pc[3];
    const int nchw = t0 / pc[3];
    const float gv = g[i];
    int bmax = -1;
    float m = -3.4e38f;
    for (int ki = 0; ki < pc[5]; ++ki) {
        const int ii = ho * pc[7] + ki - pc[9];
        if (ii < 0 || ii >= pc[1]) continue;
        for (int kj = 0; kj < pc[6]; ++kj) {
            const int jj = wo * pc[8] + kj - pc[10];
            if (jj < 0 || jj >= pc[2]) continue;
            const float v = x[nchw * pc[1] * pc[2] + ii * pc[2] + jj];
            if (v > m) { m = v; bmax = ii * pc[2] + jj; }
        }
    }
    if (bmax < 0) return;
    const uint slot = (uint)(nchw * pc[1] * pc[2] + bmax);
    uint old = atomic_load_explicit(dxu + slot, memory_order_relaxed);
    for (;;) {
        const uint neu = as_type<uint>(as_type<float>(old) + gv);
        uint expected = old;
        if (atomic_compare_exchange_weak_explicit(dxu + slot, &expected, neu,
                memory_order_relaxed, memory_order_relaxed)) break;
        old = expected;
    }
    (void)dx;
}

kernel void poolavggrad(device const float* g [[buffer(0)]],
                        device float* dx [[buffer(2)]],
                        device atomic_uint* dxu [[buffer(3)]],
                        constant const int* pc [[buffer(8)]], uint3 tpig [[thread_position_in_grid]]) {
    const int i = (int)(tpig.x);
    if (i >= pc[0]) return;
    const int wo = i % pc[4];
    const int t0 = i / pc[4];
    const int ho = t0 % pc[3];
    const int nchw = t0 / pc[3];
    const float gv = g[i];
    int cnt = 0;
    for (int ki = 0; ki < pc[5]; ++ki) {
        const int ii = ho * pc[7] + ki - pc[9];
        if (ii < 0 || ii >= pc[1]) continue;
        for (int kj = 0; kj < pc[6]; ++kj) {
            const int jj = wo * pc[8] + kj - pc[10];
            if (jj < 0 || jj >= pc[2]) continue;
            ++cnt;
        }
    }
    if (cnt == 0) return;
    const float add = gv / (float)cnt;
    for (int ki = 0; ki < pc[5]; ++ki) {
        const int ii = ho * pc[7] + ki - pc[9];
        if (ii < 0 || ii >= pc[1]) continue;
        for (int kj = 0; kj < pc[6]; ++kj) {
            const int jj = wo * pc[8] + kj - pc[10];
            if (jj < 0 || jj >= pc[2]) continue;
            const uint slot = (uint)(nchw * pc[1] * pc[2] + ii * pc[2] + jj);
            uint old = atomic_load_explicit(dxu + slot, memory_order_relaxed);
            for (;;) {
                const uint neu = as_type<uint>(as_type<float>(old) + add);
                uint expected = old;
                if (atomic_compare_exchange_weak_explicit(dxu + slot, &expected, neu,
                        memory_order_relaxed, memory_order_relaxed)) break;
                old = expected;
            }
        }
    }
    (void)dx;
}
)MSL";

// ---- Metal 共享内存 d2d ----
void reshape_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                   const void*) {
    const Tensor& x = *in[0];
    Tensor& y = *out[0];
    if (x.is_contiguous() && y.is_contiguous()) {
        std::memcpy(mh(y), mh(x), (size_t)x.numel() * 4);
        return;
    }
    const int rank = (int)y.dim_count();
    long long ss[8], ds[8], sh[8];
    for (int d = 0; d < rank; ++d) {
        const int64_t sd = d + x.dim_count() - rank;
        ss[d] = sd >= 0 ? x.strides()[sd] : 0;
        ds[d] = y.strides()[d];
        sh[d] = y.size(d);
    }
    strided_copy_dispatch(x, y, rank, ss, ds, sh, y.numel());
}

void transpose_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                     const void* options) {
    struct TOptions { int64_t d0, d1; };
    const TOptions* o = static_cast<const TOptions*>(options);
    const Tensor& x = *in[0];
    Tensor& y = *out[0];
    const int64_t r = x.dim_count();
    const int64_t d0 = o->d0 < 0 ? o->d0 + r : o->d0;
    const int64_t d1 = o->d1 < 0 ? o->d1 + r : o->d1;
    long long ss[8], ds[8], sh[8];
    for (int64_t d = 0; d < r; ++d) {
        ss[d] = x.strides()[d];
        ds[d] = (d == d0) ? y.strides()[d1]
                          : (d == d1 ? y.strides()[d0] : y.strides()[d]);
        sh[d] = x.size(d);
    }
    strided_copy_dispatch(x, y, (int)r, ss, ds, sh, x.numel());
}

void permute_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                   const void* options) {
    struct POptions { int64_t order[8]; int64_t rank; };
    const POptions* o = static_cast<const POptions*>(options);
    const Tensor& x = *in[0];
    Tensor& y = *out[0];
    int64_t inv[8];
    for (int64_t k = 0; k < o->rank; ++k) inv[o->order[k]] = k;
    long long ss[8], ds[8], sh[8];
    for (int64_t d = 0; d < o->rank; ++d) {
        ss[d] = x.strides()[d];
        ds[d] = y.strides()[inv[d]];
        sh[d] = x.size(d);
    }
    strided_copy_dispatch(x, y, (int)o->rank, ss, ds, sh, x.numel());
}

void slice_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                 const void* options) {
    struct SOptions { int64_t dim; int64_t begin; int64_t end; };
    const SOptions* o = static_cast<const SOptions*>(options);
    const Tensor& x = *in[0];
    Tensor& y = *out[0];
    const int64_t dim = o->dim < 0 ? o->dim + x.dim_count() : o->dim;
    const int rank = (int)y.dim_count();
    long long ss[8], ds[8], sh[8];
    for (int d = 0; d < rank; ++d) {
        ss[d] = x.strides()[d];
        ds[d] = y.strides()[d];
        sh[d] = y.size(d);
    }
    // Metal 共享内存: mapped 指针可算术, 直接偏移源
    Tensor xs = x;   // 共享句柄的"伪偏移视图": 用步长表吸收偏移
    // begin*stride 并入 ss 不行(逐维) —— 用 copy_view_to 语义: 直接调
    // impl 层等价: 构造 dst 视图写 —— Metal 下最简: memcpy per 行组.
    // 退而求其次: 用 CPU 式遍历(mapped 可直接读写)
    float* xp = (float*)mh(x) + x.storage_offset()
                + o->begin * x.strides()[dim];
    float* yp = (float*)mh(y) + y.storage_offset();
    int64_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t e = 0; e < y.numel(); ++e) {
        int64_t so = 0, dof = 0;
        for (int64_t d = 0; d < rank; ++d) {
            so += idx[d] * ss[d];
            dof += idx[d] * ds[d];
        }
        yp[dof] = xp[so];
        for (int64_t d = rank - 1; d >= 0; --d) {
            if (++idx[d] < sh[d]) break;
            idx[d] = 0;
        }
    }
    (void)xs;
}

void cat_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
               const void* options) {
    Tensor& y = *out[0];
    const int64_t dim = static_cast<const int64_t*>(options)[0];
    int64_t offset = 0;
    float* yp = (float*)mh(y) + y.storage_offset();
    for (size_t i = 0; i < in.size(); ++i) {
        const Tensor& x = *in[i];
        const float* xp = (float*)mh(x) + x.storage_offset();
        const int rank = (int)x.dim_count();
        int64_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        const int64_t* xs = x.strides();
        const int64_t* ys = y.strides();
        for (int64_t e = 0; e < x.numel(); ++e) {
            int64_t so = 0, dof = offset * ys[dim];
            for (int64_t d = 0; d < rank; ++d) {
                so += idx[d] * xs[d];
                dof += idx[d] * ys[d];
            }
            yp[dof] = xp[so];
            for (int64_t d = rank - 1; d >= 0; --d) {
                if (++idx[d] < x.size(d)) break;
                idx[d] = 0;
            }
        }
        offset += x.size(dim);
    }
}

void stack_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                 const void* options) {
    Tensor& y = *out[0];
    const int64_t dim = static_cast<const int64_t*>(options)[0];
    float* yp = (float*)mh(y) + y.storage_offset();
    for (size_t i = 0; i < in.size(); ++i) {
        const Tensor& x = *in[i];
        const float* xp = (float*)mh(x) + x.storage_offset();
        const int rank = (int)x.dim_count();
        int64_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        const int64_t* xs = x.strides();
        const int64_t* ys = y.strides();
        for (int64_t e = 0; e < x.numel(); ++e) {
            int64_t so = 0, dof = (int64_t)i * ys[dim];
            for (int64_t d = 0; d < rank; ++d) {
                so += idx[d] * xs[d];
                dof += idx[d] * ys[d < dim ? d : d + 1];
            }
            yp[dof] = xp[so];
            for (int64_t d = rank - 1; d >= 0; --d) {
                if (++idx[d] < x.size(d)) break;
                idx[d] = 0;
            }
        }
    }
}

void arange_metal(KernelContext&, Span<const Tensor*>, Span<Tensor*> out,
                  const void*) {
    float* p = (float*)mh(*out[0]);
    const int64_t n = out[0]->numel();
    for (int64_t i = 0; i < n; ++i) p[i] = (float)i;   // 共享内存直写
}

void randn_metal(KernelContext&, Span<const Tensor*>, Span<Tensor*> out,
                 const void* options) {
    const unsigned seed = *static_cast<const unsigned*>(options);
    const int64_t n = out[0]->numel();
    std::vector<float> host((size_t)n);
    std::mt19937 engine(seed ? seed : 42);
    std::normal_distribution<float> dist(0.f, 1.f);
    for (int64_t i = 0; i < n; ++i) host[(size_t)i] = dist(engine);
    std::memcpy(mh(*out[0]), host.data(), (size_t)n * 4);
}

void embedding_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                     const void*) {
    const Tensor& w = *in[0];
    const Tensor& idx = *in[1];
    const int64_t D = w.size(1);
    static id<MTLComputePipelineState> p = nil;
    if (!p) p = device::pipeline(kMslFill, "embedfwd");
    int pc[3] = {(int)(idx.numel() * D), (int)D, (int)(w.numel() / D)};
    const device::MBuf bufs[3] = {mbuf(w), mbuf(idx), mbuf(*out[0])};
    device::dispatch_bufs(p, bufs, 3, pc, sizeof(pc),
                          MTLSizeMake(rup((uint32_t)pc[0], 256), 1, 1),
                          MTLSizeMake(256, 1, 1));
}

void embedding_grad_metal(const Tensor& g, const Tensor& idx, Tensor& dw) {
    const int64_t D = dw.size(1);
    static id<MTLComputePipelineState> p = nil;
    if (!p) p = device::pipeline(kMslFill, "embedgrad");
    int pc[3] = {(int)(idx.numel() * D), (int)D, 0};
    const device::MBuf bufs[4] = {mbuf(g), mbuf(idx), mbuf(dw), mbuf(dw)};
    device::dispatch_bufs(p, bufs, 4, pc, sizeof(pc),
                          MTLSizeMake(rup((uint32_t)pc[0], 256), 1, 1),
                          MTLSizeMake(256, 1, 1));
}

void layer_norm_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                      const void* options) {
    // options = LayerNormOptions{double eps; int64_t dim}
    struct LnOpts { double eps; int64_t dim; };
    const LnOpts* o = static_cast<const LnOpts*>(options);
    Tensor xc = in[0]->is_contiguous() ? *in[0] : in[0]->clone();
    const Tensor& x = xc;
    const int64_t rank = x.dim_count();
    const int64_t dim = o->dim < 0 ? rank - 1 : o->dim;
    if (dim == rank - 1) {
        const int64_t d = x.size(-1);
        const int64_t rows = x.numel() / d;
        static id<MTLComputePipelineState> p = nil;
        if (!p) p = device::pipeline(kMslNorm, "lnrow");
        struct { int32_t r, dm, h, w; float eps, mom; int32_t tr; } pc{
            (int32_t)rows, (int32_t)d, 0, 0, (float)o->eps, 0.f, 0};
        const device::MBuf bufs[6] = {mbuf(x), mbuf(*in[1]), mbuf(*in[2]),
                                      mbuf(x), mbuf(x), mbuf(*out[0])};
        device::dispatch_bufs_tg(p, bufs, 6, &pc, sizeof(pc),
                                 MTLSizeMake(256, (uint32_t)rows, 1),
                                 MTLSizeMake(256, 1, 1), 256 * 4, 256 * 4);
        return;
    }
    int64_t D = x.size(dim), inner = 1, groups = 1;
    for (int64_t d = 0; d < rank; ++d) {
        if (d < dim) groups *= x.size(d);
        else if (d > dim) inner *= x.size(d);
    }
    groups *= inner;               // 组数 = outer × inner(gi = o*inner + i)
    static id<MTLComputePipelineState> pg = nil;
    if (!pg) pg = device::pipeline(kMslNorm, "lngroup");
    struct { int32_t r, dm, h, w; float eps, mom; int32_t tr; } pc{
        (int32_t)groups, (int32_t)D, 0, (int32_t)inner, (float)o->eps, 0.f, 0};
    const device::MBuf bufs[6] = {mbuf(x), mbuf(*in[1]), mbuf(*in[2]),
                                  mbuf(x), mbuf(x), mbuf(*out[0])};
    device::dispatch_bufs_tg(pg, bufs, 6, &pc, sizeof(pc),
                             MTLSizeMake((uint32_t)((groups+31)/32)*256, 1, 1),
                             MTLSizeMake(256, 1, 1), 256 * 4, 256 * 4);
}

void batch_norm2d_metal(KernelContext&, Span<const Tensor*> in,
                        Span<Tensor*> out, const void* options) {
    // options = BatchNormOptions{double eps, double momentum, bool training}
    // (此前 float[3] 强转读错训练标志 —— 见 CUDA 同款修复)
    const BatchNormOptions* o = static_cast<const BatchNormOptions*>(options);
    const Tensor& x = *in[0];
    static id<MTLComputePipelineState> p = nil;
    if (!p) p = device::pipeline(kMslNorm, "bnchannel");
    struct { int32_t m, c, h, w; float eps, mom; int32_t tr; } pc{
        (int32_t)(x.size(0) * x.size(2) * x.size(3)), (int32_t)x.size(1),
        (int32_t)x.size(2), (int32_t)x.size(3),
        (float)o->eps, (float)o->momentum, o->training ? 1 : 0};
    const device::MBuf bufs[6] = {mbuf(x), mbuf(*in[1]), mbuf(*in[2]),
                                  mbuf(*in[3]), mbuf(*in[4]), mbuf(*out[0])};
    device::dispatch_bufs_tg(p, bufs, 6, &pc, sizeof(pc),
                             MTLSizeMake(256, (uint32_t)x.size(1), 1),
                             MTLSizeMake(256, 1, 1), 256 * 4, 0);
}

void cross_entropy_metal(KernelContext&, Span<const Tensor*> in,
                         Span<Tensor*> out, const void*) {
    const Tensor& x = *in[0];
    const Tensor& t = *in[1];
    static id<MTLComputePipelineState> p = nil;
    if (!p) p = device::pipeline(kMslLoss, "cekernel");
    int pc[2] = {(int)x.size(0), (int)x.size(1)};
    const device::MBuf bufs[4] = {mbuf(x), mbuf(x), mbuf(t), mbuf(*out[0])};
    device::dispatch_bufs_tg(p, bufs, 4, pc, sizeof(pc),
                             MTLSizeMake(256, 1, 1), MTLSizeMake(256, 1, 1),
                             256 * 4, 0);
}

void mse_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
               const void*) {
    static id<MTLComputePipelineState> p = nil;
    if (!p) p = device::pipeline(kMslLoss, "msekernel");
    int pc[2] = {(int)in[0]->numel(), 0};
    const device::MBuf bufs[4] = {mbuf(*in[0]), mbuf(*in[1]),
                                  mbuf(*in[1]), mbuf(*out[0])};
    device::dispatch_bufs_tg(p, bufs, 4, pc, sizeof(pc),
                             MTLSizeMake(256, 1, 1), MTLSizeMake(256, 1, 1),
                             256 * 4, 0);
}

namespace {
struct ConvPcM {
    int32_t n, c, h, w, o, kh, kw, ho, wo;
    int32_t sh, sw, ph, pw, dh, dwd, g;
};
ConvPcM conv_pcm(const Tensor& x, const Tensor& wt, const Tensor& y,
                 const Conv2dOptions& o) {
    return ConvPcM{(int32_t)x.size(0), (int32_t)x.size(1), (int32_t)x.size(2),
                   (int32_t)x.size(3), (int32_t)wt.size(0), (int32_t)wt.size(2),
                   (int32_t)wt.size(3), (int32_t)y.size(2), (int32_t)y.size(3),
                   (int32_t)o.stride[0], (int32_t)o.stride[1],
                   (int32_t)o.pad[0], (int32_t)o.pad[1],
                   (int32_t)o.dilation[0], (int32_t)o.dilation[1],
                   (int32_t)o.groups};
}
} // namespace

void conv2d_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                  const void* options) {
    const Conv2dOptions* o = static_cast<const Conv2dOptions*>(options);
    const Tensor& x=*in[0]; const Tensor& w=*in[1];
    if(x.is_contiguous() && w.is_contiguous() && x.size(0)==1 && o->groups==1
       && w.size(2)==1 && w.size(3)==1 && o->stride[0]==1 && o->stride[1]==1
       && o->pad[0]==0 && o->pad[1]==0) {
        const int spatial=int(x.size(2)*x.size(3));
        gemm_run(w,x,*out[0],int(w.size(0)),spatial,int(x.size(1)),int(x.size(1)),spatial,0);
        return;
    }
    static id<MTLComputePipelineState> p = nil;
    if (!p) p = device::pipeline(kMslConv, "convfwd");
    ConvPcM pc = conv_pcm(*in[0], *in[1], *out[0], *o);
    const device::MBuf bufs[4] = {mbuf(*in[0]), mbuf(*in[1]),
                                  mbuf(*in[0]), mbuf(*out[0])};
    device::dispatch_bufs(p, bufs, 4, &pc, sizeof(pc),
                          MTLSizeMake(rup((uint32_t)out[0]->numel(), 256), 1, 1),
                          MTLSizeMake(256, 1, 1));
}

void conv2d_grad_metal(const Tensor& x, const Tensor& w, const Tensor& g,
                       Tensor& dx, Tensor& dw, const void* options) {
    const Conv2dOptions* o = static_cast<const Conv2dOptions*>(options);
    ConvPcM pc = conv_pcm(x, w, g, *o);
    static id<MTLComputePipelineState> pdw = nil;
    if (!pdw) pdw = device::pipeline(kMslConv, "convdw");
    const device::MBuf bufs_dw[4] = {mbuf(x), mbuf(w), mbuf(g), mbuf(dw)};
    device::dispatch_bufs(pdw, bufs_dw, 4, &pc, sizeof(pc),
                          MTLSizeMake(rup((uint32_t)dw.numel(), 256), 1, 1),
                          MTLSizeMake(256, 1, 1));
    static id<MTLComputePipelineState> pdx = nil;
    if (!pdx) pdx = device::pipeline(kMslConv, "convdx");
    const device::MBuf bufs_dx[4] = {mbuf(x), mbuf(w), mbuf(g), mbuf(dx)};
    device::dispatch_bufs(pdx, bufs_dx, 4, &pc, sizeof(pc),
                          MTLSizeMake(rup((uint32_t)dx.numel(), 256), 1, 1),
                          MTLSizeMake(256, 1, 1));
}

namespace {
struct PoolPcM {
    int32_t total, h, w, ho, wo, kh, kw, sh, sw, ph, pw;
};
PoolPcM pool_pcm(const Tensor& x, const Tensor& y, const PoolCallOptions& o) {
    return PoolPcM{(int32_t)y.numel(), (int32_t)x.size(2), (int32_t)x.size(3),
                   (int32_t)y.size(2), (int32_t)y.size(3),
                   (int32_t)o.kernel[0], (int32_t)o.kernel[1],
                   (int32_t)o.pool.stride[0], (int32_t)o.pool.stride[1],
                   (int32_t)o.pool.pad[0], (int32_t)o.pool.pad[1]};
}
} // namespace

void max_pool2d_metal(KernelContext&, Span<const Tensor*> in,
                      Span<Tensor*> out, const void* options) {
    const PoolCallOptions* o = static_cast<const PoolCallOptions*>(options);
    static id<MTLComputePipelineState> p = nil;
    if (!p) p = device::pipeline(kMslPool, "poolmax");
    PoolPcM pc = pool_pcm(*in[0], *out[0], *o);
    const device::MBuf bufs[4] = {mbuf(*in[0]), mbuf(*in[0]),
                                  mbuf(*out[0]), mbuf(*out[0])};
    device::dispatch_bufs(p, bufs, 4, &pc, sizeof(pc),
                          MTLSizeMake(rup((uint32_t)out[0]->numel(), 256), 1, 1),
                          MTLSizeMake(256, 1, 1));
}

void avg_pool2d_metal(KernelContext&, Span<const Tensor*> in,
                      Span<Tensor*> out, const void* options) {
    const PoolCallOptions* o = static_cast<const PoolCallOptions*>(options);
    static id<MTLComputePipelineState> p = nil;
    if (!p) p = device::pipeline(kMslPool, "poolavg");
    PoolPcM pc = pool_pcm(*in[0], *out[0], *o);
    const device::MBuf bufs[4] = {mbuf(*in[0]), mbuf(*in[0]),
                                  mbuf(*out[0]), mbuf(*out[0])};
    if(pc.ho==1 && pc.wo==1 && pc.ph==0 && pc.pw==0 && pc.kh==pc.h && pc.kw==pc.w) {
        static id<MTLComputePipelineState> pg=nil;
        if(!pg)pg=device::pipeline(kMslPool,"poolglobalavg");
        device::dispatch_bufs_tg(pg,bufs,4,&pc,sizeof(pc),MTLSizeMake(uint32_t(pc.total)*256,1,1),MTLSizeMake(256,1,1),256*4,0);return;
    }
    device::dispatch_bufs(p, bufs, 4, &pc, sizeof(pc),
                          MTLSizeMake(rup((uint32_t)out[0]->numel(), 256), 1, 1),
                          MTLSizeMake(256, 1, 1));
}

void max_pool_grad_metal(const Tensor& x, const Tensor& g, Tensor& dx,
                         const void* options) {
    const PoolCallOptions* o = static_cast<const PoolCallOptions*>(options);
    static id<MTLComputePipelineState> p = nil;
    if (!p) p = device::pipeline(kMslPool, "poolmaxgrad");
    PoolPcM pc = pool_pcm(x, g, *o);
    pc.total = (int32_t)g.numel();
    const device::MBuf bufs[4] = {mbuf(g), mbuf(x), mbuf(dx), mbuf(dx)};
    device::dispatch_bufs(p, bufs, 4, &pc, sizeof(pc),
                          MTLSizeMake(rup((uint32_t)g.numel(), 256), 1, 1),
                          MTLSizeMake(256, 1, 1));
}

void avg_pool_grad_metal(const Tensor& x, const Tensor& g, Tensor& dx,
                         const void* options) {
    const PoolCallOptions* o = static_cast<const PoolCallOptions*>(options);
    static id<MTLComputePipelineState> p = nil;
    if (!p) p = device::pipeline(kMslPool, "poolavggrad");
    PoolPcM pc = pool_pcm(x, g, *o);
    pc.total = (int32_t)g.numel();
    const device::MBuf bufs[4] = {mbuf(g), mbuf(g), mbuf(dx), mbuf(dx)};
    device::dispatch_bufs(p, bufs, 4, &pc, sizeof(pc),
                          MTLSizeMake(rup((uint32_t)g.numel(), 256), 1, 1),
                          MTLSizeMake(256, 1, 1));
}

void attention_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                     const void* options) {
    const Tensor& q = *in[0];
    const Tensor& k = *in[1];
    const Tensor& v = *in[2];
    Tensor& y = *out[0];
    const int64_t B = q.size(0), H = q.size(1), L = q.size(2), D = q.size(3);
    const float scale = 1.f / std::sqrt((float)D);
    struct Options { bool causal; };
    const bool causal = options && static_cast<const Options*>(options)->causal;
    if (L <= 64 && q.is_contiguous() && k.is_contiguous()
        && v.is_contiguous() && y.is_contiguous()) {
        static id<MTLComputePipelineState> p = nil;
        if (!p) p = device::pipeline(kMslAttention, "attention_fused");
        struct Pc { int bh, L, D; float scale; int causal; } pc{
            (int)(B * H), (int)L, (int)D, scale, causal ? 1 : 0};
        const device::MBuf bufs[4] = {mbuf(q), mbuf(k), mbuf(v), mbuf(y)};
        device::dispatch_bufs(p, bufs, 4, &pc, sizeof(pc),
                              MTLSizeMake(rup((uint32_t)(B * H * L * D), 64), 1, 1),
                              MTLSizeMake(64, 1, 1));
        return;
    }
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            Tensor qh = q.slice(0, b, b + 1).slice(1, h, h + 1).reshape({L, D});
            Tensor kh = k.slice(0, b, b + 1).slice(1, h, h + 1).reshape({L, D});
            Tensor vh = v.slice(0, b, b + 1).slice(1, h, h + 1).reshape({L, D});
            Tensor p = Tensor::linear(qh, kh);
            Tensor sm = Activator::softmax(Tensor::mul(
                p, Tensor::full(Shape{}, scale, DType::F32, q.device())));
            Tensor o1 = Tensor::matmul(sm, vh).reshape({1, L, D});
            // 共享内存: 直接按步长写 y 的 (b,h) 头
            const float* op = (const float*)mh(o1);
            float* yp = (float*)mh(y) + (b * H + h) * L * D;
            std::memcpy(yp, op, (size_t)(L * D) * 4);
        }
    }
}

} // namespace metal_backend
} // namespace znn

#endif // __APPLE__ && ZNN_METAL
