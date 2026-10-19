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
#include "backends/vulkan/gen/unarybwd.h"
#include "backends/vulkan/gen/copy.h"
#include "backends/vulkan/gen/bcastgrad.h"
#include "backends/vulkan/gen/opt.h"
#include <stdexcept>
#include <cstring>

namespace znn {
namespace vulkan_backend {

namespace {
namespace bridge = device_bridge;

void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(std::string("znn vulkan: ") + what);
}

// Storage 裸句柄(与 CUDA 内核同模式; Vulkan 侧为 VkBuf*)
void* vhandle(const Tensor& t) {
    return t.data<Tensor::VulkanDataPtr>();   // 裸句柄(基址)
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
    void* p = gemm_pipeline(flops, 0);
    require(p, "gemm pipeline");
    struct { int32_t M, N, K, lda, ldb, ldc; } pc{M, N, K, K, N, N};
    const uint32_t gx = (uint32_t)((N + tile - 1) / tile);
    const uint32_t gy = (uint32_t)((M + tile - 1) / tile);
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
    const int tile = flops >= (int64_t)50 << 30 ? 64 : 32;
    void* p = gemm_pipeline(flops, 1);
    require(p, "gemm-t pipeline");
    struct { int32_t M, N, K, lda, ldb, ldc; } pc{M, N, K, K, K, N};
    const uint32_t gx = (uint32_t)((N + tile - 1) / tile);
    const uint32_t gy = (uint32_t)((M + tile - 1) / tile);
    bridge::dispatch(p, vhandle(x), vhandle(w), vhandle(y), &pc, sizeof(pc), gx, gy, 1);
    // bias: in[2] 存在时逐行广播加(复用 elementwise 广播通道)
    if (in.size() > 2 && in[2]) {
        // 每行加同一向量 —— b[0] 广播不适用, 由 CPU 侧 add 算子组合;
        // v1: 拷回加再上传? 否 —— 借 mul 广播不成立, 留给组合层。
        // 简明起见: linear Vulkan 版要求无 bias(组合层会拆分调用)。
        throw std::runtime_error("znn vulkan: linear bias use add op composition");
    }
}

// ---- 规约: sum/mean 沿 dim(任意维) ----
namespace {
void reduce_run(Span<const Tensor*> in, Span<Tensor*> out, int64_t dim, bool mean) {
    const Tensor& x = *in[0];
    Tensor& y = *out[0];
    require(x.is_contiguous() && y.is_contiguous(), "reduce needs contiguous");
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
    const uint32_t groups = (uint32_t)((pc.total + 255) / 256);
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

// ---- softmax(末维) ----
void softmax_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                    const void* options) {
    const Tensor& x = *in[0];
    Tensor& y = *out[0];
    require(x.is_contiguous() && y.is_contiguous(), "softmax needs contiguous");
    struct SO { int64_t dim; };
    const int64_t dim = static_cast<const SO*>(options)->dim;
    const int64_t d = dim < 0 ? dim + x.dim_count() : dim;
    if (d != x.dim_count() - 1) {
        throw std::runtime_error("znn vulkan: softmax last-dim only (fallback)");
    }
    void* p = bridge::get_pipeline(spirv::softmax, spirv::softmax_bytes, 0, 0);
    require(p, "softmax pipeline");
    struct Pc { int32_t rows; int32_t L; } pc{(int32_t)(x.numel() / x.size(d)),
                                              (int32_t)x.size(d)};
    const uint32_t groups = (uint32_t)((pc.rows + 255) / 256);
    bridge::dispatch(p, vhandle(x), vhandle(x), vhandle(y), &pc, sizeof(pc), groups, 1, 1);
}

// ---- 直连内核(unary_bwd 经注册表槽位; 其余经 backend_slots 宏直调) ----
void unary_bwd_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                      const void* options) {
    const int fn = (int)*static_cast<const unsigned char*>(options);
    void* p = bridge::get_pipeline(spirv::unarybwd, spirv::unarybwd_bytes,
                                   (uint32_t)fn, 0, 4);
    require(p, "unarybwd pipeline");
    const int32_t n = (int32_t)out[0]->numel();
    void* bufs[4] = {vhandle(*in[0]), vhandle(*in[1]), vhandle(*in[2]),
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
    void* bufs[2] = {vhandle(g), vhandle(out)};
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
    } pc{};
    pc.total = (int32_t)total;
    pc.rank = rank;
    for (int d = 0; d < rank; ++d) {
        pc.ss[d] = (int32_t)ss[d];
        pc.ds[d] = (int32_t)ds[d];
        pc.sh[d] = (int32_t)sh[d];
    }
    void* bufs[2] = {vhandle(src), vhandle(dst)};
    bridge::dispatch_bufs(p, bufs, 2, &pc, sizeof(pc),
                          (unsigned)((total + 255) / 256), 1, 1);
}

void strided_copy_raw_vk(const void* src, void* dst, int rank,
                         const long long* ss, const long long* ds,
                         const long long* sh, long long total) {
    void* p = bridge::get_pipeline(spirv::copy, spirv::copy_bytes, 0, 0, 2);
    require(p, "copy pipeline");
    struct Pc {
        int32_t total, rank;
        int32_t ss[8], ds[8], sh[8];
    } pc{};
    pc.total = (int32_t)total;
    pc.rank = rank;
    for (int d = 0; d < rank; ++d) {
        pc.ss[d] = (int32_t)ss[d];
        pc.ds[d] = (int32_t)ds[d];
        pc.sh[d] = (int32_t)sh[d];
    }
    void* bufs[2] = {const_cast<void*>(src), dst};
    bridge::dispatch_bufs(p, bufs, 2, &pc, sizeof(pc),
                          (unsigned)((total + 255) / 256), 1, 1);
}

void sgd_step_vulkan(void* p, const void* g, void* vel, long long n,
                     float lr, float momentum, float wd) {
    void* pl = bridge::get_pipeline(spirv::opt, spirv::opt_bytes, 0, 0, 4);
    require(pl, "sgd pipeline");
    struct Pc { int32_t n; float lr, p1, p2; } pc{(int32_t)n, lr, momentum, wd};
    void* bufs[4] = {p, const_cast<void*>(g), vel, vel};   // U2 占位(OPT=0 不读)
    bridge::dispatch_bufs(pl, bufs, 4, &pc, sizeof(pc),
                          (unsigned)((n + 255) / 256), 1, 1);
}

void adam_step_vulkan(void* p, const void* g, void* m, void* v,
                      long long n, float lr, float b1, float b2, float eps,
                      float wd, float bias1, float bias2) {
    void* pl = bridge::get_pipeline(spirv::opt, spirv::opt_bytes, 1, 0, 4);
    require(pl, "adam pipeline");
    struct Pc { int32_t n; float lr, p1, p2, p3, p4, p5, p6; } pc{
        (int32_t)n, lr, b1, b2, eps, wd, bias1, bias2};
    void* bufs[4] = {p, const_cast<void*>(g), m, v};
    bridge::dispatch_bufs(pl, bufs, 4, &pc, sizeof(pc),
                          (unsigned)((n + 255) / 256), 1, 1);
}

} // namespace vulkan_backend
} // namespace znn

#endif // ZNN_VULKAN
