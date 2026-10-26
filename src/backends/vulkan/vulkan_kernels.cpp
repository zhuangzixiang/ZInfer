// ============================================================================
// vulkan_kernels.cpp — Vulkan 算子内核(SPIR-V 管线派发)
//
// 逐元素 13 算子共用 common.comp(OP 特化); matmul/linear 走 gemm.comp
// (TB=1 即 × w^T); sum/mean 走 reduce.comp; softmax 走 softmax.comp(末维)。
// 失败(设备不可用/形状不支持)抛错, 派发器透明回落 CPU。
// ============================================================================
#include "operator.hpp"
#include "backends/vulkan/vulkan_backend.hpp"

#if defined(ZNN_VULKAN)

#define VK_USE_PLATFORM_WIN32_KHR 0
#include "vulkan/vulkan.h"
#include "backends/vulkan/gen/common.h"
#include "backends/vulkan/gen/gemm.h"
#include "backends/vulkan/gen/gemm_big.h"
#include "backends/vulkan/gen/reduce.h"
#include "backends/vulkan/gen/softmax.h"
#include "backends/vulkan/gen/softmaxdim.h"
#include "backends/vulkan/gen/upsample.h"
#include "backends/vulkan/gen/unarybwd.h"
#include "backends/vulkan/gen/copy.h"
#include "backends/vulkan/gen/bcastgrad.h"
#include "backends/vulkan/gen/opt.h"
#include "backends/vulkan/gen/fill.h"
#include "backends/vulkan/gen/embed.h"
#include "backends/vulkan/gen/norm.h"
#include "backends/vulkan/gen/loss.h"
#include "backends/vulkan/gen/conv.h"
#include "backends/vulkan/gen/pool.h"
#include "backends/vulkan/gen/attention.h"
#include "operators/shared.hpp"
#include "operators/batch_norm_options.hpp"
#include <cmath>
#include <random>
#include <vector>
#include <stdexcept>
#include <cstring>

namespace znn {
namespace vulkan_backend {

namespace {
namespace bridge = device_bridge;

void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(std::string("znn vulkan: ") + what);
}

// Storage 裸句柄(基址, 无绑定偏移 —— VkDescriptorBufferInfo.offset 受
// minStorageBufferOffsetAlignment 限制; 视图偏移一律走推送常量/拷贝区偏移)
VBuf vhandle(const Tensor& t) {
    return VBuf{t.data<Tensor::VulkanDataPtr>(), 0};
}
void* vbase(const Tensor& t) {
    return t.data<Tensor::VulkanDataPtr>();
}

// 右对齐广播步长(维长 1 → 步长 0); 与 CUDA bcast_map 同构
struct BcMap {
    int32_t n, rank;
    int32_t dims[8], sa[8], sb[8];
};
bool bcast_map(const Tensor& a, const Tensor& b, const Shape& out, BcMap& m) {
    m.rank = (int32_t)out.rank;
    int64_t ca = 1, cb = 1;
    for (int64_t d = out.rank - 1; d >= 0; --d) {
        const int64_t ad = d - (int64_t)(out.rank - a.dim_count());
        const int64_t bd = d - (int64_t)(out.rank - b.dim_count());
        const int64_t av = ad < 0 ? 1 : a.size(ad);
        const int64_t bv = bd < 0 ? 1 : b.size(bd);
        if ((av != out.dims[d] && av != 1) || (bv != out.dims[d] && bv != 1)) {
            return false;
        }
        m.dims[d] = (int32_t)out.dims[d];
        m.sa[d] = av == 1 ? 0 : (int32_t)ca;
        m.sb[d] = bv == 1 ? 0 : (int32_t)cb;
        ca *= av;
        cb *= bv;
    }
    return true;
}

// 通用逐元素 host 入口: op 编码见 common.comp; 二元全维广播
void elementwise(Span<const Tensor*> in, Span<Tensor*> out, int op) {
    Tensor a = in[0]->is_contiguous() ? *in[0] : in[0]->clone();   // 视图材料化
    Tensor& y = *out[0];
    BcMap pc{};
    pc.n = (int32_t)y.numel();
    if (op <= 3) {                               // 二元: 广播步长图
        Tensor b = in[1]->is_contiguous() ? *in[1] : in[1]->clone();
        if (!bcast_map(a, b, y.shape(), pc)) {
            throw std::runtime_error("znn: shapes not broadcastable");
        }
        void* p = bridge::get_pipeline(spirv::common, spirv::common_bytes,
                                       (uint32_t)op, 0);
        require(p, "pipeline create failed");
        const uint32_t groups = (uint32_t)((pc.n + 255) / 256);
        bridge::dispatch(p, vhandle(a), vhandle(b), vhandle(y),
                         &pc, sizeof(pc), groups, 1, 1);
        return;
    }
    pc.rank = 1;                                 // 一元: 平凡图
    pc.dims[0] = pc.n;
    pc.sa[0] = 1;
    pc.sb[0] = 1;
    require(a.is_contiguous() && y.is_contiguous(), "elementwise needs contiguous");
    void* p = bridge::get_pipeline(spirv::common, spirv::common_bytes,
                                   (uint32_t)op, 0);
    require(p, "pipeline create failed");
    const uint32_t groups = (uint32_t)((pc.n + 255) / 256);
    bridge::dispatch(p, vhandle(a), vhandle(a), vhandle(y),
                     &pc, sizeof(pc), groups, 1, 1);
}
} // namespace

// ---- 逐元素 13 算子(OP 编码: 0+ 1- 2* 3/ 4neg 5exp 6log 7sqrt
//      8relu 9gelu 10sigmoid 11tanh 12silu; 二元全维广播) ----
void add_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 0);
}
void sub_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 1);
}
void mul_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 2);
}
void div_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 3);
}
void neg_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 4);
}
void exp_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 5);
}
void log_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 6);
}
void sqrt_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 7);
}
void relu_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 8);
}
void gelu_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 9);
}
void sigmoid_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 10);
}
void tanh_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 11);
}
void silu_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 12);
}

// ---- GEMM: C[M,N] = A × op(B)(TB=1 时 B 为 [N,K] 行主序) ----
namespace {
bool gemm(const float* A, int lda, const float* Bp, int ldb,
          float* C, int ldc, int M, int N, int K, int tb) {
    if (!available()) return false;
    // 借用张量存储接口: 裸指针场景由调用方包一层(见 matmul_vulkan)
    (void)A; (void)Bp; (void)C;
    (void)lda; (void)ldb; (void)ldc; (void)M; (void)N; (void)K; (void)tb;
    return false;
}
} // namespace

// 内核选择(cuBLAS 同做法): 小中规模 32×32(占用高), 大规模 64×64
// (B 装载摊销翻倍, 大工作集不崩塌); 交叉点 ~50 GFLOP(实测)
namespace {
void* gemm_pipeline(int64_t flops, int big_c0) {
    if (flops >= (int64_t)50 << 30) {   // ≥50 GFLOP → 64×64 大内核
        return bridge::get_pipeline(spirv::gemm_big, spirv::gemm_big_bytes, big_c0, 0);
    }
    return bridge::get_pipeline(spirv::gemm, spirv::gemm_bytes, big_c0, 0);
}
} // namespace

void matmul_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                   const void*) {
    const Tensor& a = *in[0];
    const Tensor& b = *in[1];
    Tensor& y = *out[0];
    require(a.is_contiguous() && b.is_contiguous() && y.is_contiguous(),
            "matmul needs contiguous");
    const int M = (int)a.size(0), K = (int)a.size(1), N = (int)b.size(1);
    const int64_t flops = (int64_t)M * N * K * 2;
    const bool big = flops >= (int64_t)50 << 30;
    const int tile = big ? 64 : 32;
    const int mstep = big ? 64 : 128;    // 32 核 M 持久化: 每组连做 4 行块
    void* p = gemm_pipeline(flops, 0);
    require(p, "gemm pipeline");
    struct { int32_t M, N, K, lda, ldb, ldc; } pc{M, N, K, K, N, N};
    const uint32_t gx = (uint32_t)((N + tile - 1) / tile);
    const uint32_t gy = (uint32_t)((M + mstep - 1) / mstep);
    bridge::dispatch(p, vhandle(a), vhandle(b), vhandle(y), &pc, sizeof(pc), gx, gy, 1);
}

void linear_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                   const void*) {
    // y = x @ w^T (+b 经后续 add 广播, 由算子组合层处理); TB=1
    const Tensor& x = *in[0];
    const Tensor& w = *in[1];
    Tensor& y = *out[0];
    require(x.is_contiguous() && w.is_contiguous() && y.is_contiguous(),
            "linear needs contiguous");
    const int M = (int)x.size(0), K = (int)x.size(1), N = (int)w.size(0);
    const int64_t flops = (int64_t)M * N * K * 2;
    const bool big = flops >= (int64_t)50 << 30;
    const int tile = big ? 64 : 32;
    void* p = gemm_pipeline(flops, 1);
    require(p, "gemm-t pipeline");
    struct { int32_t M, N, K, lda, ldb, ldc; } pc{M, N, K, K, K, N};
    const int mstep = big ? 64 : 128;   // 32 核 M 持久化(与 matmul 同)
    const uint32_t gx = (uint32_t)((N + tile - 1) / tile);
    const uint32_t gy = (uint32_t)((M + mstep - 1) / mstep);
    bridge::dispatch(p, vhandle(x), vhandle(w), vhandle(y), &pc, sizeof(pc), gx, gy, 1);
    // bias: 复用 elementwise 全维广播加(b[N] 对 y[M,N] 右对齐广播;
    // 此前直接抛错 —— common.comp 的 stride-map 广播本就支持行向量)
    if (in.size() > 2 && in[2]) {
        Tensor ya = *out[0], bb = *in[2];
        const Tensor* ins[2] = {&ya, &bb};
        Tensor* outs[1] = {out[0]};
        elementwise(Span<const Tensor*>(ins, 2), Span<Tensor*>(outs, 1), 0);
    }
}

// ---- 规约: sum/mean 沿 dim(任意维; 每 workgroup 一输出协作归约) ----
namespace {
void reduce_run(Span<const Tensor*> in, Span<Tensor*> out, int64_t dim, bool mean) {
    Tensor xm = in[0]->is_contiguous() ? *in[0] : in[0]->clone();   // 视图材料化
    const Tensor& x = xm;
    Tensor& y = *out[0];
    if (dim < 0) dim += x.dim_count();
    void* p = bridge::get_pipeline(spirv::reduce, spirv::reduce_bytes,
                                  (uint32_t)(x.dim_count() - 1),
                                  mean ? 1u : 0u);
    require(p, "reduce pipeline");
    struct Pc {
        int32_t total; int32_t n_dim; int32_t s_dim;
        int32_t odims[8]; int32_t xstr[8];
    } pc{};
    pc.n_dim = (int32_t)x.size(dim);
    pc.s_dim = (int32_t)x.strides()[dim];
    pc.total = (int32_t)y.numel();
    int r = 0;
    for (int64_t d = 0; d < x.dim_count(); ++d) {
        if (d == dim) continue;
        pc.odims[r] = (int32_t)x.size(d);
        pc.xstr[r] = (int32_t)x.strides()[d];
        ++r;
    }
    const uint32_t groups = (uint32_t)pc.total;   // 每 workgroup 一输出
    bridge::dispatch(p, vhandle(x), vhandle(x), vhandle(y), &pc, sizeof(pc), groups, 1, 1);
}
} // namespace

void sum_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                const void* options) {
    // options 布局与 shared.hpp ReduceAttrs 一致
    struct RA { int64_t dim; bool keepdim; };
    const RA* a = static_cast<const RA*>(options);
    reduce_run(in, out, a->dim, false);
}

void mean_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                 const void* options) {
    struct RA { int64_t dim; bool keepdim; };
    const RA* a = static_cast<const RA*>(options);
    reduce_run(in, out, a->dim, true);
}

// ---- softmax(末维快路径 + 任意 dim 通用路径) ----
void upsample2x_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                       const void*) {
    Tensor xm = in[0]->is_contiguous() ? *in[0] : in[0]->clone();
    const Tensor& x = xm;
    void* p = bridge::get_pipeline(spirv::upsample, spirv::upsample_bytes, 0, 0, 2);
    require(p, "upsample pipeline");
    struct Pc { int32_t total, w_in, w_out; } pc{
        (int32_t)x.numel(), (int32_t)x.size(3), (int32_t)(x.size(3) * 2)};
    const VBuf bufs[2] = {vhandle(x), vhandle(*out[0])};
    bridge::dispatch_bufs(p, bufs, 2, &pc, sizeof(pc),
                          (unsigned)((pc.total + 255) / 256), 1, 1);
}

void softmax_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                    const void* options) {
    Tensor xm = in[0]->is_contiguous() ? *in[0] : in[0]->clone();   // 视图材料化
    const Tensor& x = xm;
    Tensor& y = *out[0];
    struct SO { int64_t dim; };
    const int64_t dim = static_cast<const SO*>(options)->dim;
    const int64_t d = dim < 0 ? dim + x.dim_count() : dim;
    if (d == x.dim_count() - 1) {                 // 末维: 每线程一行
        void* p = bridge::get_pipeline(spirv::softmax, spirv::softmax_bytes, 0, 0);
        require(p, "softmax pipeline");
        struct Pc { int32_t rows; int32_t L; } pc{(int32_t)(x.numel() / x.size(d)),
                                                  (int32_t)x.size(d)};
        const uint32_t groups = (uint32_t)((pc.rows + 255) / 256);
        bridge::dispatch(p, vhandle(x), vhandle(x), vhandle(y), &pc, sizeof(pc),
                         groups, 1, 1);
        return;
    }
    // 一般维: 每 workgroup 一行(此前直接抛错降级 —— 现原生)
    void* p = bridge::get_pipeline(spirv::softmaxdim, spirv::softmaxdim_bytes,
                                   (uint32_t)(x.dim_count() - 1), 0);
    require(p, "softmaxdim pipeline");
    struct Pc {
        int32_t rows; int32_t n_dim; int32_t s_dim;
        int32_t odims[8]; int32_t xstr[8];
    } pc{};
    pc.n_dim = (int32_t)x.size(d);
    pc.s_dim = (int32_t)x.strides()[d];
    pc.rows = 1;
    int r = 0;
    for (int64_t k = 0; k < x.dim_count(); ++k) {
        if (k == d) continue;
        pc.odims[r] = (int32_t)x.size(k);
        pc.xstr[r] = (int32_t)x.strides()[k];
        ++r;
        pc.rows *= (int32_t)x.size(k);
    }
    bridge::dispatch(p, vhandle(x), vhandle(x), vhandle(y), &pc, sizeof(pc),
                     (uint32_t)pc.rows, 1, 1);
}

// ---- 直连内核(unary_bwd 经注册表槽位; 其余经 backend_slots 宏直调) ----
void unary_bwd_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                      const void* options) {
    const int fn = (int)*static_cast<const unsigned char*>(options);
    void* p = bridge::get_pipeline(spirv::unarybwd, spirv::unarybwd_bytes,
                                   (uint32_t)fn, 0, 4);
    require(p, "unarybwd pipeline");
    const int32_t n = (int32_t)out[0]->numel();
    const VBuf bufs[4] = {vhandle(*in[0]), vhandle(*in[1]), vhandle(*in[2]),
                         vhandle(*out[0])};
    bridge::dispatch_bufs(p, bufs, 4, &n, sizeof(n),
                          (unsigned)((n + 255) / 256), 1, 1);
}

void grad_bcast_vulkan(const Tensor& g, Tensor& out, long long total, int rank,
                       const long long* dims, const long long* eff, float scale) {
    void* p = bridge::get_pipeline(spirv::bcastgrad, spirv::bcastgrad_bytes,
                                   0, 0, 2);
    require(p, "bcastgrad pipeline");
    struct Pc {
        int32_t total, rank;
        int32_t dims[8], eff[8];
        float scale;
    } pc{};
    pc.total = (int32_t)total;
    pc.rank = rank;
    pc.scale = scale;
    for (int d = 0; d < rank; ++d) {
        pc.dims[d] = (int32_t)dims[d];
        pc.eff[d] = (int32_t)eff[d];
    }
    const VBuf bufs[2] = {vhandle(g), vhandle(out)};
    bridge::dispatch_bufs(p, bufs, 2, &pc, sizeof(pc),
                          (unsigned)((total + 255) / 256), 1, 1);
}

void strided_copy_vulkan(const Tensor& src, Tensor& dst, int rank,
                         const long long* ss, const long long* ds,
                         const long long* sh, long long total) {
    void* p = bridge::get_pipeline(spirv::copy, spirv::copy_bytes, 0, 0, 2);
    require(p, "copy pipeline");
    struct Pc {
        int32_t total, rank;
        int32_t ss[8], ds[8], sh[8];
        int32_t soff, doff;
    } pc{};
    pc.total = (int32_t)total;
    pc.rank = rank;
    pc.soff = (int32_t)src.storage_offset();
    pc.doff = (int32_t)dst.storage_offset();
    for (int d = 0; d < rank; ++d) {
        pc.ss[d] = (int32_t)ss[d];
        pc.ds[d] = (int32_t)ds[d];
        pc.sh[d] = (int32_t)sh[d];
    }
    const VBuf bufs[2] = {{vbase(src), 0}, {vbase(dst), 0}};
    bridge::dispatch_bufs(p, bufs, 2, &pc, sizeof(pc),
                          (unsigned)((total + 255) / 256), 1, 1);
}

void clone_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                  const void*) {
    const Tensor& src = *in[0];
    Tensor& dst = *out[0];
    const int rank = static_cast<int>(dst.dim_count());
    long long ss[8] = {}, ds[8] = {}, sh[8] = {};
    long long total = 1;
    for (int d = 0; d < rank; ++d) {
        ss[d] = src.strides()[d];
        ds[d] = dst.strides()[d];
        sh[d] = dst.size(d);
        total *= sh[d];
    }
    strided_copy_vulkan(src, dst, rank, ss, ds, sh, total);
}

void strided_copy_raw_vk(const void* src, uint64_t soff, void* dst, uint64_t doff,
                         int rank, const long long* ss, const long long* ds,
                         const long long* sh, long long total) {
    void* p = bridge::get_pipeline(spirv::copy, spirv::copy_bytes, 0, 0, 2);
    require(p, "copy pipeline");
    struct Pc {
        int32_t total, rank;
        int32_t ss[8], ds[8], sh[8];
        int32_t soff, doff;
    } pc{};
    pc.total = (int32_t)total;
    pc.rank = rank;
    pc.soff = (int32_t)(soff / 4);   // 字节 → 元素(推送常量, 绕开绑定对齐)
    pc.doff = (int32_t)(doff / 4);
    for (int d = 0; d < rank; ++d) {
        pc.ss[d] = (int32_t)ss[d];
        pc.ds[d] = (int32_t)ds[d];
        pc.sh[d] = (int32_t)sh[d];
    }
    const VBuf bufs[2] = {{const_cast<void*>(src), 0}, {dst, 0}};
    bridge::dispatch_bufs(p, bufs, 2, &pc, sizeof(pc),
                          (unsigned)((total + 255) / 256), 1, 1);
}

void sgd_step_vulkan(const Tensor& p, const Tensor& g, const Tensor& vel,
                     long long n, float lr, float momentum, float wd) {
    void* pl = bridge::get_pipeline(spirv::opt, spirv::opt_bytes, 0, 0, 4);
    require(pl, "sgd pipeline");
    struct Pc { int32_t n; float lr, p1, p2; } pc{(int32_t)n, lr, momentum, wd};
    const VBuf bufs[4] = {vhandle(p), vhandle(g), vhandle(vel), vhandle(vel)};
    bridge::dispatch_bufs(pl, bufs, 4, &pc, sizeof(pc),
                          (unsigned)((n + 255) / 256), 1, 1);
}

void adam_step_vulkan(const Tensor& p, const Tensor& g, const Tensor& m,
                      const Tensor& v, long long n, float lr, float b1,
                      float b2, float eps, float wd, float decay, float bias1, float bias2) {
    void* pl = bridge::get_pipeline(spirv::opt, spirv::opt_bytes, 1, 0, 4);
    require(pl, "adam pipeline");
    struct Pc { int32_t n; float lr, p1, p2, p3, p4, p5, p6, p7; } pc{
        (int32_t)n, lr, b1, b2, eps, wd, bias1, bias2, decay};
    const VBuf bufs[4] = {vhandle(p), vhandle(g), vhandle(m), vhandle(v)};
    bridge::dispatch_bufs(pl, bufs, 4, &pc, sizeof(pc),
                          (unsigned)((n + 255) / 256), 1, 1);
}

// ============================================================ SB2-SB4

namespace {
void* hdl(const Tensor& t) { return t.data<Tensor::VulkanDataPtr>(); }
uint64_t grp(long long n) { return (uint64_t)((n + 255) / 256); }
} // namespace

// ---- 布局族: 步长表 + strided_copy_raw(数据全程留 GPU) ----
void reshape_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                    const void*) {
    const Tensor& x = *in[0];
    Tensor& y = *out[0];
    if (x.is_contiguous() && y.is_contiguous()) {
        // 视图偏移经 VkBufferCopy 区偏移(绑定层无对齐顾虑)
        copy_device_to_device_off(hdl(y), (uint64_t)y.storage_offset() * 4,
                                  hdl(x), (uint64_t)x.storage_offset() * 4,
                                  (size_t)x.numel() * 4);
        return;
    }
    const int rank = (int)y.dim_count();
    long long ss[8], ds[8], sh[8];
    for (int d = 0; d < rank; ++d) {
        const int64_t sd = d + x.dim_count() - rank;   // 右对齐(维度数可不同)
        ss[d] = sd >= 0 ? x.strides()[sd] : 0;
        ds[d] = y.strides()[d];
        sh[d] = y.size(d);
    }
    strided_copy_raw(Device::Vulkan, hdl(x), x.storage_offset(), hdl(y),
                     y.storage_offset(), rank, ss, ds, sh, y.numel());
}

void transpose_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
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
    strided_copy_raw(Device::Vulkan, hdl(x), x.storage_offset(), hdl(y),
                     y.storage_offset(), (int)r, ss, ds, sh, x.numel());
}

void permute_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                    const void* options) {
    struct POptions { int64_t order[8]; int64_t rank; };
    const POptions* o = static_cast<const POptions*>(options);
    const Tensor& x = *in[0];
    Tensor& y = *out[0];
    int64_t inv[8];
    for (int64_t k = 0; k < o->rank; ++k) {
        inv[o->order[k]] = k;
    }
    long long ss[8], ds[8], sh[8];
    for (int64_t d = 0; d < o->rank; ++d) {
        ss[d] = x.strides()[d];
        ds[d] = y.strides()[inv[d]];
        sh[d] = x.size(d);
    }
    strided_copy_raw(Device::Vulkan, hdl(x), x.storage_offset(), hdl(y),
                     y.storage_offset(), (int)o->rank, ss, ds, sh, x.numel());
}

void slice_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
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
    strided_copy_raw(Device::Vulkan, hdl(x),
                     x.storage_offset() + o->begin * x.strides()[dim],
                     hdl(y), y.storage_offset(), rank, ss, ds, sh, y.numel());
}

void cat_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                const void* options) {
    Tensor& y = *out[0];
    const int64_t dim = static_cast<const int64_t*>(options)[0];
    int64_t offset = 0;
    for (size_t i = 0; i < in.size(); ++i) {
        const Tensor& x = *in[i];
        const int rank = (int)x.dim_count();
        long long ss[8], ds[8], sh[8];
        for (int d = 0; d < rank; ++d) {
            ss[d] = x.strides()[d];
            ds[d] = y.strides()[d];
            sh[d] = x.size(d);
        }
        strided_copy_raw(Device::Vulkan, hdl(x), x.storage_offset(), hdl(y),
                         y.storage_offset() + offset * y.strides()[dim],
                         rank, ss, ds, sh, x.numel());
        offset += x.size(dim);
    }
}

void stack_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                  const void* options) {
    Tensor& y = *out[0];
    const int64_t dim = static_cast<const int64_t*>(options)[0];
    for (size_t i = 0; i < in.size(); ++i) {
        const Tensor& x = *in[i];
        const int rank = (int)x.dim_count();
        long long ss[8], ds[8], sh[8];
        for (int d = 0; d < rank; ++d) {
            ss[d] = x.strides()[d];
            ds[d] = y.strides()[d < dim ? d : d + 1];   // 输出多一维
            sh[d] = x.size(d);
        }
        strided_copy_raw(Device::Vulkan, hdl(x), x.storage_offset(), hdl(y),
                         y.storage_offset() + (int64_t)i * y.strides()[dim],
                         rank, ss, ds, sh, x.numel());
    }
}

// ---- 生成族 ----
void arange_vulkan(KernelContext&, Span<const Tensor*>, Span<Tensor*> out,
                   const void*) {
    void* p = bridge::get_pipeline(spirv::fill, spirv::fill_bytes, 0, 0, 1);
    require(p, "fill pipeline");
    const int32_t n = (int32_t)out[0]->numel();
    const VBuf bufs[1] = {vhandle(*out[0])};
    bridge::dispatch_bufs(p, bufs, 1, &n, sizeof(n), 1, 1, 1);
}

void randn_vulkan(KernelContext&, Span<const Tensor*>, Span<Tensor*> out,
                  const void* options) {
    // 与 CUDA 同源: 宿主 Box-Muller 后上传(数值逐位一致)
    const unsigned seed = *static_cast<const unsigned*>(options);
    const int64_t n = out[0]->numel();
    std::vector<float> host((size_t)n);
    std::mt19937 engine(seed ? seed : 42);
    std::normal_distribution<float> dist(0.f, 1.f);
    for (int64_t i = 0; i < n; ++i) host[(size_t)i] = dist(engine);
    copy_host_to_device(hdl(*out[0]), host.data(), (size_t)n * sizeof(float));
}

// ---- embedding(前向行收集; 梯度 CAS 散射) ----
void embedding_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                      const void*) {
    const Tensor& w = *in[0];
    const Tensor& idx = *in[1];
    const int64_t D = w.size(1);
    void* p = bridge::get_pipeline(spirv::embed, spirv::embed_bytes, 0, 0, 3);
    require(p, "embed pipeline");
    struct Pc { int32_t total, d, rows; } pc{(int32_t)(idx.numel() * D),
                                             (int32_t)D,
                                             (int32_t)(w.numel() / D)};
    const VBuf bufs[3] = {vhandle(w), vhandle(idx), vhandle(*out[0])};
    bridge::dispatch_bufs(p, bufs, 3, &pc, sizeof(pc),
                          (unsigned)grp(idx.numel() * D), 1, 1);
}

void embedding_grad_vulkan(const Tensor& g, const Tensor& idx, Tensor& dw) {
    const int64_t D = dw.size(1);
    void* p = bridge::get_pipeline(spirv::embed, spirv::embed_bytes, 1, 0, 4);
    require(p, "embed-grad pipeline");
    struct Pc { int32_t total, d, rows; } pc{(int32_t)(idx.numel() * D),
                                             (int32_t)D, 0};
    const VBuf bufs[4] = {vhandle(g), vhandle(idx), vhandle(dw), vhandle(dw)};
    bridge::dispatch_bufs(p, bufs, 4, &pc, sizeof(pc),
                          (unsigned)grp(idx.numel() * D), 1, 1);
}

// ---- 归一化族 ----
void layer_norm_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                       const void* options) {
    // options = LayerNormOptions{double eps; int64_t dim}
    struct LnOpts { double eps; int64_t dim; };
    const LnOpts* o = static_cast<const LnOpts*>(options);
    Tensor xc = in[0]->is_contiguous() ? *in[0] : in[0]->clone();
    const Tensor& x = xc;
    const int64_t rank = x.dim_count();
    const int64_t dim = o->dim < 0 ? rank - 1 : o->dim;
    if (dim == rank - 1) {                               // 末维快路径
        const int64_t d = x.size(-1);
        const int64_t rows = x.numel() / d;
        void* p = bridge::get_pipeline(spirv::norm, spirv::norm_bytes, 0, 0, 6);
        require(p, "ln pipeline");
        struct Pc { int32_t rows, d, h, w; float eps, momentum; int32_t training; } pc{
            (int32_t)rows, (int32_t)d, 0, 0, (float)o->eps, 0.f, 0};
        const VBuf bufs[6] = {vhandle(x), vhandle(*in[1]), vhandle(*in[2]),
                              vhandle(x), vhandle(x), vhandle(*out[0])};
        bridge::dispatch_bufs(p, bufs, 6, &pc, sizeof(pc), 1, (unsigned)rows, 1);
        return;
    }
    int64_t D = x.size(dim), inner = 1, groups = 1;
    for (int64_t d = 0; d < rank; ++d) {
        if (d < dim) groups *= x.size(d);
        else if (d > dim) inner *= x.size(d);
    }
    groups *= inner;               // 组数 = outer × inner(gi = o*inner + i)
    const bool spatial=inner>=1024;
    void* p = bridge::get_pipeline(spirv::norm, spirv::norm_bytes, spatial?3:2, 0, 6);
    require(p, "ln-dim pipeline");
    struct Pc { int32_t rows, d, h, w; float eps, momentum; int32_t training; } pc{
        (int32_t)groups, (int32_t)D, 0, (int32_t)inner, (float)o->eps, 0.f, 0};
    const VBuf bufs[6] = {vhandle(x), vhandle(*in[1]), vhandle(*in[2]),
                          vhandle(x), vhandle(x), vhandle(*out[0])};
    bridge::dispatch_bufs(p, bufs, 6, &pc, sizeof(pc),
                          (unsigned)(spatial?(groups+31)/32:groups), 1, 1);
}

void batch_norm2d_vulkan(KernelContext&, Span<const Tensor*> in,
                         Span<Tensor*> out, const void* options) {
    // options = BatchNormOptions{double eps, double momentum, bool training}
    // (此前 float[3] 强转读错训练标志 —— 见 CUDA 同款修复)
    const BatchNormOptions* o = static_cast<const BatchNormOptions*>(options);
    const Tensor& x = *in[0];
    const int C = (int)x.size(1);
    void* p = bridge::get_pipeline(spirv::norm, spirv::norm_bytes, 1, 0, 6);
    require(p, "bn pipeline");
    struct Pc { int32_t m, c, h, w; float eps, momentum; int32_t training; } pc{
        (int32_t)(x.size(0) * x.size(2) * x.size(3)), C,
        (int32_t)x.size(2), (int32_t)x.size(3),
        (float)o->eps, (float)o->momentum, o->training ? 1 : 0};
    const VBuf bufs[6] = {vhandle(x), vhandle(*in[1]), vhandle(*in[2]),
                          vhandle(*in[3]), vhandle(*in[4]), vhandle(*out[0])};
    bridge::dispatch_bufs(p, bufs, 6, &pc, sizeof(pc), 1, (unsigned)C, 1);
}

// ---- 损失族 ----
void cross_entropy_vulkan(KernelContext&, Span<const Tensor*> in,
                          Span<Tensor*> out, const void*) {
    const Tensor& x = *in[0];
    const Tensor& t = *in[1];
    void* p = bridge::get_pipeline(spirv::loss, spirv::loss_bytes, 0, 0, 4);
    require(p, "ce pipeline");
    struct Pc { int32_t n, c; } pc{(int32_t)x.size(0), (int32_t)x.size(1)};
    const VBuf bufs[4] = {vhandle(x), vhandle(x), vhandle(t), vhandle(*out[0])};
    bridge::dispatch_bufs(p, bufs, 4, &pc, sizeof(pc), 1, 1, 1);
}

void mse_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                const void*) {
    void* p = bridge::get_pipeline(spirv::loss, spirv::loss_bytes, 1, 0, 4);
    require(p, "mse pipeline");
    struct Pc { int32_t n, c; } pc{(int32_t)in[0]->numel(), 0};
    const VBuf bufs[4] = {vhandle(*in[0]), vhandle(*in[1]),
                          vhandle(*in[1]), vhandle(*out[0])};
    bridge::dispatch_bufs(p, bufs, 4, &pc, sizeof(pc), 1, 1, 1);
}

// ---- conv2d(前向/dw/dx; 与 CUDA conv2d_k/dw_k/dx_k 逐行同构) ----
namespace {
struct ConvPc {
    int32_t n, c, h, w, o, kh, kw, ho, wo;
    int32_t sh, sw, ph, pw, dh, dwd, g;
};
ConvPc conv_pc(const Tensor& x, const Tensor& wt, const Tensor& y,
               const Conv2dOptions& o) {
    return ConvPc{(int32_t)x.size(0), (int32_t)x.size(1), (int32_t)x.size(2),
                  (int32_t)x.size(3), (int32_t)wt.size(0), (int32_t)wt.size(2),
                  (int32_t)wt.size(3), (int32_t)y.size(2), (int32_t)y.size(3),
                  (int32_t)o.stride[0], (int32_t)o.stride[1],
                  (int32_t)o.pad[0], (int32_t)o.pad[1],
                  (int32_t)o.dilation[0], (int32_t)o.dilation[1],
                  (int32_t)o.groups};
}
} // namespace

void conv2d_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                   const void* options) {
    const Conv2dOptions* o = static_cast<const Conv2dOptions*>(options);
    const Tensor& x = *in[0];
    const Tensor& w = *in[1];
    Tensor& y = *out[0];
    if(x.is_contiguous() && w.is_contiguous() && x.size(0)==1 && o->groups==1
       && w.size(2)==1 && w.size(3)==1 && o->stride[0]==1 && o->stride[1]==1
       && o->pad[0]==0 && o->pad[1]==0) {
        const int M=int(w.size(0)), N=int(x.size(2)*x.size(3)), K=int(x.size(1));
        const int64_t flops=int64_t(M)*N*K*2;
        const bool big=flops >= (int64_t(50)<<30);
        auto p=gemm_pipeline(flops,0); require(p,"pointwise gemm pipeline");
        struct {int32_t M,N,K,lda,ldb,ldc;} pc{M,N,K,K,N,N};
        bridge::dispatch(p,vhandle(w),vhandle(x),vhandle(y),&pc,sizeof(pc),
            (N+(big?63:31))/(big?64:32),(M+(big?63:127))/(big?64:128),1);
        return;
    }
    void* p = bridge::get_pipeline(spirv::conv, spirv::conv_bytes, 0, 0, 4);
    require(p, "conv pipeline");
    ConvPc pc = conv_pc(x, w, y, *o);
    const VBuf bufs[4] = {vhandle(x), vhandle(w), vhandle(x), vhandle(y)};
    bridge::dispatch_bufs(p, bufs, 4, &pc, sizeof(pc),
                          (unsigned)grp(y.numel()), 1, 1);
}

void conv2d_grad_vulkan(const Tensor& x, const Tensor& w, const Tensor& g,
                        Tensor& dx, Tensor& dw, const void* options) {
    const Conv2dOptions* o = static_cast<const Conv2dOptions*>(options);
    ConvPc pc = conv_pc(x, w, g, *o);   // 输出几何取自 g(=前向输出形状)
    void* pdw = bridge::get_pipeline(spirv::conv, spirv::conv_bytes, 1, 0, 4);
    require(pdw, "conv-dw pipeline");
    const VBuf bufs_dw[4] = {vhandle(x), vhandle(w), vhandle(g), vhandle(dw)};
    bridge::dispatch_bufs(pdw, bufs_dw, 4, &pc, sizeof(pc),
                          (unsigned)grp(dw.numel()), 1, 1);
    void* pdx = bridge::get_pipeline(spirv::conv, spirv::conv_bytes, 2, 0, 4);
    require(pdx, "conv-dx pipeline");
    const VBuf bufs_dx[4] = {vhandle(x), vhandle(w), vhandle(g), vhandle(dx)};
    bridge::dispatch_bufs(pdx, bufs_dx, 4, &pc, sizeof(pc),
                          (unsigned)grp(dx.numel()), 1, 1);
}

// ---- 池化 ----
namespace {
struct PoolPc {
    int32_t total, h, w, ho, wo, kh, kw, sh, sw, ph, pw;
};
PoolPc pool_pc(const Tensor& x, const Tensor& y, const PoolCallOptions& o) {
    return PoolPc{(int32_t)y.numel(), (int32_t)x.size(2), (int32_t)x.size(3),
                  (int32_t)y.size(2), (int32_t)y.size(3),
                  (int32_t)o.kernel[0], (int32_t)o.kernel[1],
                  (int32_t)o.pool.stride[0], (int32_t)o.pool.stride[1],
                  (int32_t)o.pool.pad[0], (int32_t)o.pool.pad[1]};
}
} // namespace

void max_pool2d_vulkan(KernelContext&, Span<const Tensor*> in,
                       Span<Tensor*> out, const void* options) {
    const PoolCallOptions* o = static_cast<const PoolCallOptions*>(options);
    void* p = bridge::get_pipeline(spirv::pool, spirv::pool_bytes, 0, 0, 4);
    require(p, "pool pipeline");
    PoolPc pc = pool_pc(*in[0], *out[0], *o);
    const VBuf bufs[4] = {vhandle(*in[0]), vhandle(*in[0]),
                          vhandle(*out[0]), vhandle(*out[0])};
    bridge::dispatch_bufs(p, bufs, 4, &pc, sizeof(pc),
                          (unsigned)grp(out[0]->numel()), 1, 1);
}

void avg_pool2d_vulkan(KernelContext&, Span<const Tensor*> in,
                       Span<Tensor*> out, const void* options) {
    const PoolCallOptions* o = static_cast<const PoolCallOptions*>(options);
    const bool global=out[0]->size(2)==1 && out[0]->size(3)==1 && o->pool.pad[0]==0 && o->pool.pad[1]==0
        && o->kernel[0]==in[0]->size(2) && o->kernel[1]==in[0]->size(3);
    void* p = bridge::get_pipeline(spirv::pool, spirv::pool_bytes, global?4:1, 0, 4);
    require(p, "pool pipeline");
    PoolPc pc = pool_pc(*in[0], *out[0], *o);
    const VBuf bufs[4] = {vhandle(*in[0]), vhandle(*in[0]),
                          vhandle(*out[0]), vhandle(*out[0])};
    bridge::dispatch_bufs(p, bufs, 4, &pc, sizeof(pc),
                          global?unsigned(out[0]->numel()):(unsigned)grp(out[0]->numel()), 1, 1);
}

void max_pool_grad_vulkan(const Tensor& x, const Tensor& g, Tensor& dx,
                          const void* options) {
    const PoolCallOptions* o = static_cast<const PoolCallOptions*>(options);
    void* p = bridge::get_pipeline(spirv::pool, spirv::pool_bytes, 2, 0, 4);
    require(p, "pool-grad pipeline");
    PoolPc pc = pool_pc(x, g, *o);      // 输出几何取自 g(=池化输出形状)
    pc.total = (int32_t)g.numel();
    const VBuf bufs[4] = {vhandle(g), vhandle(x), vhandle(dx), vhandle(dx)};
    bridge::dispatch_bufs(p, bufs, 4, &pc, sizeof(pc),
                          (unsigned)grp(g.numel()), 1, 1);
}

void avg_pool_grad_vulkan(const Tensor& x, const Tensor& g, Tensor& dx,
                          const void* options) {
    const PoolCallOptions* o = static_cast<const PoolCallOptions*>(options);
    void* p = bridge::get_pipeline(spirv::pool, spirv::pool_bytes, 3, 0, 4);
    require(p, "pool-grad pipeline");
    PoolPc pc = pool_pc(x, g, *o);      // 输出几何取自 g(=池化输出形状)
    pc.total = (int32_t)g.numel();
    const VBuf bufs[4] = {vhandle(g), vhandle(g), vhandle(dx), vhandle(dx)};
    bridge::dispatch_bufs(p, bufs, 4, &pc, sizeof(pc),
                          (unsigned)grp(g.numel()), 1, 1);
}

// ---- attention: 每 (b,h) 头组合原生算子(linear=q@kT/softmax/matmul),
//      输出经 strided copy 写入 y 对应视图 —— 零 CPU 往返 ----
void attention_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
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
        void* p = bridge::get_pipeline(spirv::attention, spirv::attention_bytes,
                                       0, 0, 4);
        require(p, "attention pipeline");
        struct Pc { int32_t bh, L, D; float scale; int32_t causal; } pc{
            (int32_t)(B * H), (int32_t)L, (int32_t)D, scale, causal ? 1 : 0};
        const VBuf bufs[4] = {vhandle(q), vhandle(k), vhandle(v), vhandle(y)};
        bridge::dispatch_bufs(p, bufs, 4, &pc, sizeof(pc),
                              (unsigned)(((int64_t)B * H * L * D + 63) / 64), 1, 1);
        return;
    }
    const long long ss[3] = {L * D, D, 1};               // o1 [1,L,D] 连续
    const long long sh[3] = {1, L, D};
    const long long ds[3] = {0, y.strides()[2], y.strides()[3]};
    // 标量预分配到循环外(免每头一次 alloc+释放池复用)
    Tensor sc = Tensor::full(Shape{}, scale, DType::F32, q.device());
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            // 头提取 = 连续 4D 张量的带偏移 memcpy(跳过全部布局算子 ——
            // slice→reshape 三层派发在池复用下产生错误数据)
            const int64_t hoff = (b * H + h) * L * D * 4;
            Tensor qh = Tensor::empty(Shape{L, D}, DType::F32, q.device());
            Tensor kh = Tensor::empty(Shape{L, D}, DType::F32, q.device());
            Tensor vh = Tensor::empty(Shape{L, D}, DType::F32, q.device());
            copy_device_to_device_off(hdl(qh), 0, hdl(q), hoff, L * D * 4);
            copy_device_to_device_off(hdl(kh), 0, hdl(k), hoff, L * D * 4);
            copy_device_to_device_off(hdl(vh), 0, hdl(v), hoff, L * D * 4);
            Tensor p = Tensor::linear(qh, kh);           // q@kT [L,L]
            Tensor sm = Activator::softmax(Tensor::mul(p, sc));
            Tensor o1 = Tensor::matmul(sm, vh);          // [L,D]
            strided_copy_raw(Device::Vulkan, hdl(o1), 0, hdl(y),
                             (b * H + h) * L * D, 3, ss, ds, sh, L * D);
        }
    }
}

} // namespace vulkan_backend
} // namespace znn

#endif // ZNN_VULKAN
