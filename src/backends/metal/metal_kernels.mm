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

// MSL: 通用逐元素(13 算子共用; op/bcast/n 经 constant 缓冲)
const char* kMslElementwise = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void elementwise(device const float* a [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        device float* y [[buffer(2)]],
                        constant const int* pc [[buffer(3)]]) {
    // pc[0]=n pc[1]=op pc[2]=bcast
    const int i = (int)((thread_position_in_grid).x);
    if (i >= pc[0]) return;
    const float va = a[i];
    const float vb = (pc[2] == 1) ? b[0] : b[i];
    const int op = pc[1];
    float r;
    if (op == 0) r = va + vb;
    else if (op == 1) r = va - vb;
    else if (op == 2) r = va * vb;
    else if (op == 3) r = va / vb;
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

// 通用逐元素 host 入口(op 编码与 CUDA/Vulkan 一致)
void elementwise(Span<const Tensor*> in, Span<Tensor*> out, int op, int bcast) {
    require(in[0]->is_contiguous() && out[0]->is_contiguous(),
            "elementwise needs contiguous");
    const Tensor& a = *in[0];
    Tensor& y = *out[0];
    static id<MTLComputePipelineState> p = nil;   // 管线复用(同源码)
    if (!p) p = device::pipeline(kMslElementwise, "elementwise");
    const int pc[3] = {(int)y.numel(), op, bcast};
    id<MTLBuffer> vb = bcast ? mbuf(*in[1]) : mbuf(a);
    const uint32_t n = (uint32_t)y.numel();
    device::dispatch(p, mbuf(a), vb, mbuf(y), pc, sizeof(pc),
                     MTLSizeMake(rup(n, 256), 1, 1), MTLSizeMake(256, 1, 1), nil);
}

} // namespace

// ---- 逐元素 13 算子 ----
void add_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 0, in[0]->numel() != in[1]->numel() ? 1 : 0);
}
void sub_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 1, in[0]->numel() != in[1]->numel() ? 1 : 0);
}
void mul_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 2, in[0]->numel() != in[1]->numel() ? 1 : 0);
}
void div_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 3, in[0]->numel() != in[1]->numel() ? 1 : 0);
}
void neg_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 4, 0);
}
void exp_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 5, 0);
}
void log_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 6, 0);
}
void sqrt_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 7, 0);
}
void relu_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 8, 0);
}
void gelu_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 9, 0);
}
void sigmoid_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 10, 0);
}
void tanh_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 11, 0);
}
void silu_metal(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 12, 0);
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

} // namespace metal_backend
} // namespace znn

#endif // __APPLE__ && ZNN_METAL
