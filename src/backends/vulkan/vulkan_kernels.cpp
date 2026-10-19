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
#include "backends/vulkan/gen/reduce.h"
#include "backends/vulkan/gen/softmax.h"
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
    return t.data<float>();
}

// 通用逐元素 host 入口: op 编码见 common.comp
void elementwise(Span<const Tensor*> in, Span<Tensor*> out, int op, int bcast) {
    require(in[0]->is_contiguous() && out[0]->is_contiguous(),
            "elementwise needs contiguous");
    const Tensor& a = *in[0];
    Tensor& y = *out[0];
    const int64_t n = y.numel();
    void* p = bridge::get_pipeline(spirv::common, spirv::common_bytes,
                                  (uint32_t)op, (uint32_t)bcast);
    require(p, "pipeline create failed");
    const int32_t pc = (int32_t)n;
    const uint32_t groups = (uint32_t)((n + 255) / 256);
    void* vb = (in.size() > 1 && in[1]) ? vhandle(*in[1]) : vhandle(a);
    // 二元恒绑 B(bcast 仅切换着色器读 b[0] 还是 b[i]); 一元绑 A 占位
    bridge::dispatch(p, vhandle(a), vb, vhandle(y), &pc, sizeof(pc), groups, 1, 1);
}
} // namespace

// ---- 逐元素 13 算子(OP 编码: 0+ 1- 2* 3/ 4neg 5exp 6log 7sqrt
//      8relu 9gelu 10sigmoid 11tanh 12silu; 二元广播 bcast=1) ----
void add_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    const bool bcast = in[0]->numel() != in[1]->numel();
    elementwise(in, out, 0, bcast ? 1 : 0);
}
void sub_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    const bool bcast = in[0]->numel() != in[1]->numel();
    elementwise(in, out, 1, bcast ? 1 : 0);
}
void mul_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    const bool bcast = in[0]->numel() != in[1]->numel();
    elementwise(in, out, 2, bcast ? 1 : 0);
}
void div_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    const bool bcast = in[0]->numel() != in[1]->numel();
    elementwise(in, out, 3, bcast ? 1 : 0);
}
void neg_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 4, 0);
}
void exp_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 5, 0);
}
void log_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 6, 0);
}
void sqrt_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 7, 0);
}
void relu_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 8, 0);
}
void gelu_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 9, 0);
}
void sigmoid_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 10, 0);
}
void tanh_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 11, 0);
}
void silu_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    elementwise(in, out, 12, 0);
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

void matmul_vulkan(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                   const void*) {
    const Tensor& a = *in[0];
    const Tensor& b = *in[1];
    Tensor& y = *out[0];
    require(a.is_contiguous() && b.is_contiguous() && y.is_contiguous(),
            "matmul needs contiguous");
    const int M = (int)a.size(0), K = (int)a.size(1), N = (int)b.size(1);
    void* p = bridge::get_pipeline(spirv::gemm, spirv::gemm_bytes, 0, 0);
    require(p, "gemm pipeline");
    struct { int32_t M, N, K, lda, ldb, ldc; } pc{M, N, K, K, N, N};
    const uint32_t gx = (uint32_t)((N + 31) / 32);
    const uint32_t gy = (uint32_t)((M + 31) / 32);
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
    void* p = bridge::get_pipeline(spirv::gemm, spirv::gemm_bytes, 1, 0);
    require(p, "gemm-t pipeline");
    struct { int32_t M, N, K, lda, ldb, ldc; } pc{M, N, K, K, K, N};
    const uint32_t gx = (uint32_t)((N + 31) / 32);
    const uint32_t gy = (uint32_t)((M + 31) / 32);
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

} // namespace vulkan_backend
} // namespace znn

#endif // ZNN_VULKAN
