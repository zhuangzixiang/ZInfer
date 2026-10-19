// shared.cpp — 逐元素类算子的共用实现
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backend_slots.hpp"
#include <Eigen/Dense>
#include <cstring>

namespace znn {

Shape broadcast_shapes(const Shape& a, const Shape& b) {
    const int64_t rank = a.rank > b.rank ? a.rank : b.rank;
    Shape out;
    out.rank = rank;
    for (int64_t d = -1; d >= -rank; --d) {               // 右对齐逐维
        const int64_t da = -d <= a.rank ? a[d] : 1;
        const int64_t db = -d <= b.rank ? b[d] : 1;
        if (da != db && da != 1 && db != 1) {
            throw std::runtime_error("znn: shapes not broadcastable");
        }
        out.dims[d + rank] = da > db ? da : db;
    }
    return out;
}

void infer_broadcast(Span<const TensorMeta*> inputs, const void*, std::vector<Shape>& outs) {
    outs.push_back(broadcast_shapes(*inputs[0]->shape, *inputs[1]->shape));
}

void infer_same(Span<const TensorMeta*> inputs, const void*, std::vector<Shape>& outs) {
    outs.push_back(*inputs[0]->shape);
}

namespace {

// 广播后的有效步长: 该维输入尺寸为 1 则步长 0
void effective_strides(const Tensor& t, const Shape& out_shape, int64_t (&es)[8]) {
    const int64_t pad = out_shape.rank - t.dim_count();
    for (int64_t d = 0; d < out_shape.rank; ++d) {
        if (d < pad || t.size(d - pad) == 1) {
            es[d] = 0;
        } else {
            es[d] = t.strides()[d - pad];
        }
    }
}

// 行主序推进输出多下标, 同时累加各输入偏移
struct Walker {
    int64_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    bool advance(const Shape& shape) {
        for (int64_t d = shape.rank - 1; d >= 0; --d) {
            if (++idx[d] < shape.dims[d]) return true;
            idx[d] = 0;
        }
        return false;
    }
};

} // namespace

void unary_kernel(Span<const Tensor*> in, Span<Tensor*> out, float (*fn)(float)) {
    const Tensor& x = *in[0];
    Tensor& y = *out[0];
    const float* xp = x.data<float>();
    float* yp = y.data<float>();
    int64_t es[8];
    effective_strides(x, y.shape(), es);
    Walker w;
    for (int64_t i = 0; i < y.numel(); ++i) {
        int64_t off = 0;
        for (int64_t d = 0; d < y.dim_count(); ++d) {
            off += w.idx[d] * es[d];
        }
        yp[i] = fn(xp[off]);
        w.advance(y.shape());
    }
}

void binary_kernel(Span<const Tensor*> in, Span<Tensor*> out, float (*fn)(float, float)) {
    const Tensor& a = *in[0];
    const Tensor& b = *in[1];
    Tensor& y = *out[0];
    const float* ap = a.data<float>();
    const float* bp = b.data<float>();
    float* yp = y.data<float>();
    int64_t ea[8], eb[8];
    effective_strides(a, y.shape(), ea);
    effective_strides(b, y.shape(), eb);
    Walker w;
    for (int64_t i = 0; i < y.numel(); ++i) {
        int64_t oa = 0, ob = 0;
        for (int64_t d = 0; d < y.dim_count(); ++d) {
            oa += w.idx[d] * ea[d];
            ob += w.idx[d] * eb[d];
        }
        yp[i] = fn(ap[oa], bp[ob]);
        w.advance(y.shape());
    }
}

void binary_grad(BackwardContext& context) {
    const Tensor& g = *context.grad_outputs[0];
    *context.grad_inputs[0] = BackwardContext::reduce_to(g, context.saved_inputs[0]->shape());
    *context.grad_inputs[1] = BackwardContext::reduce_to(g, context.saved_inputs[1]->shape());
}

// 一元导数表 d(x, y)。CUDA 端(unary_bwd_k)有同表——设备代码无法引用
// 宿主函数, 改动须两侧同步
float unary_deriv(UnaryBwd fn, float x, float y) {
    switch (fn) {
    case UnaryBwd::Neg:     return -1.f;
    case UnaryBwd::Exp:     return y;
    case UnaryBwd::Log:     return 1.f / x;
    case UnaryBwd::Sqrt:    return 0.5f / y;
    case UnaryBwd::ReLU:    return x > 0.f ? 1.f : 0.f;
    case UnaryBwd::GELU: {
        const float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
        const float t = std::tanh(inner);
        const float dt = (1.f - t * t) * 0.7978845608f * (1.f + 3.f * 0.044715f * x * x);
        return 0.5f * (1.f + t) + 0.5f * x * dt;
    }
    case UnaryBwd::Sigmoid: return y * (1.f - y);
    case UnaryBwd::Tanh:    return 1.f - y * y;
    case UnaryBwd::Silu: {
        const float sg = 1.f / (1.f + std::exp(-x));
        return sg + x * sg * (1.f - sg);
    }
    }
    return 0.f;
}

static void unary_bwd_infer(Span<const TensorMeta*> in, const void*,
                            std::vector<Shape>& outs) {
    outs.push_back(*in[0]->shape);
}

static void unary_bwd_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                          const void* options) {
    const UnaryBwd fn = *static_cast<const UnaryBwd*>(options);
    const Tensor& g = *in[0];
    const Tensor& x = *in[1];
    const Tensor& y = *in[2];
    Tensor& gi = *out[0];
    const float* gp = g.data<float>();
    const float* xp = x.data<float>();
    const float* yp = y.data<float>();
    float* ip = gi.data<float>();
    for (int64_t i = 0; i < gi.numel(); ++i) {
        ip[i] = gp[i] * unary_deriv(fn, xp[i], yp[i]);
    }
}

const Operator g_unary_bwd_operator = {
    "unary_bwd",
    unary_bwd_infer,
    nullptr,                                   // 内部算子, 无二次反向
    unary_bwd_ref,
    {nullptr, ZNN_CUDA_K(unary_bwd_cuda), ZNN_METAL_K(unary_bwd_metal), ZNN_VULKAN_K(unary_bwd_vulkan)},
    sizeof(UnaryBwd),
};

// ---- 可微深拷贝(Tensor::clone 经派发器: 建图 + 全后端原生拷贝) ----
namespace {
void clone_run(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    impl::copy_view_to(*in[0], *out[0]);       // 内部按设备派发 strided copy
}
void clone_grad(BackwardContext& c) {
    *c.grad_inputs[0] = *c.grad_outputs[0];    // 恒等: 梯度原样回传(句柄共享)
}
} // namespace

const Operator g_clone_operator = {
    "clone",
    infer_same,
    clone_grad,
    clone_run,
    {clone_run, clone_run, clone_run, clone_run},   // 同一 host 入口, 内部按设备派发
    0,
};

void unary_grad(BackwardContext& context, UnaryBwd fn) {
    // 经派发器原生执行(任意后端); 反向期间建图已全局关闭
    const Tensor* ins[3] = {context.grad_outputs[0], context.saved_inputs[0],
                            context.saved_outputs[0]};
    *context.grad_inputs[0] = OpDispatcher::instance().run(
        &g_unary_bwd_operator, Span<const Tensor*>(ins, 3), &fn);
}

// ============================================================ Eigen 快速内核
// 连续 + 无广播 → Eigen::Map(SIMD 自动分发 AVX2/AVX-512/NEON/标量)
// 非连续或广播 → 内部回退朴素路径

namespace {
bool fast_binary_ok(const Tensor& a, const Tensor& b, const Tensor& y) {
    return a.is_contiguous() && b.is_contiguous() && y.is_contiguous()
        && a.shape() == b.shape() && a.shape() == y.shape();
}
bool fast_unary_ok(const Tensor& x, const Tensor& y) {
    return x.is_contiguous() && y.is_contiguous() && x.shape() == y.shape();
}
} // namespace

void unary_kernel_fast(Span<const Tensor*> in, Span<Tensor*> out, float (*fn)(float)) {
    const Tensor& x = *in[0];
    Tensor& y = *out[0];
    if (!fast_unary_ok(x, y)) {
        unary_kernel(in, out, fn);                       // 回退朴素
        return;
    }
    const int64_t n = y.numel();
    Eigen::Map<Eigen::VectorXf> xm(x.data<float>(), n);
    Eigen::Map<Eigen::VectorXf> ym(y.data<float>(), n);
    ym = xm.unaryExpr([fn](float v) { return fn(v); }); // Eigen SIMD
}

void binary_kernel_fast(Span<const Tensor*> in, Span<Tensor*> out, float (*fn)(float, float)) {
    const Tensor& a = *in[0];
    const Tensor& b = *in[1];
    Tensor& y = *out[0];
    if (!fast_binary_ok(a, b, y)) {
        binary_kernel(in, out, fn);                      // 回退朴素
        return;
    }
    const int64_t n = y.numel();
    Eigen::Map<Eigen::VectorXf> am(a.data<float>(), n);
    Eigen::Map<Eigen::VectorXf> bm(b.data<float>(), n);
    Eigen::Map<Eigen::VectorXf> ym(y.data<float>(), n);
    ym = am.binaryExpr(bm, [fn](float u, float v) { return fn(u, v); });
}

void sum_into(const Tensor& x, int64_t dim, bool keepdim, Tensor& out) {
    // 连续末维快速路径: 行主序内层连续累加(通用路径每元素做步长算术, 慢 20 倍)
    if (x.is_contiguous() && out.is_contiguous()
        && dim == x.dim_count() - 1 && x.dim_count() >= 1) {
        const int64_t rows = out.numel();
        const int64_t d = x.size(dim);
        const float* xp = x.data<float>();
        float* op = out.data<float>();
        for (int64_t r = 0; r < rows; ++r) {
            float acc = 0.f;
            const float* xr = xp + r * d;
            for (int64_t k = 0; k < d; ++k) {
                acc += xr[k];
            }
            op[r] = acc;
        }
        return;
    }
    // 连续首维快速路径: 内层连续累加(与末维路径同理)
    if (x.is_contiguous() && out.is_contiguous() && dim == 0) {
        const int64_t outer = x.size(0);
        const int64_t inner = out.numel();
        const float* xp2 = x.data<float>();
        float* op2 = out.data<float>();
        for (int64_t o = 0; o < outer; ++o) {
            const float* xr = xp2 + o * inner;
            for (int64_t k = 0; k < inner; ++k) {
                op2[k] += xr[k];
            }
        }
        return;
    }
    const float* xp = x.data<float>();
    float* op = out.data<float>();
    const int64_t* xs = x.strides();
    const int64_t* os = out.strides();
    int64_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t i = 0; i < x.numel(); ++i) {
        int64_t ro = 0;
        for (int64_t d = 0; d < x.dim_count(); ++d) {
            if (d != dim) {
                // keepdim 时输出仍含被规约维(长 1), 映射保持 d;
                // 此前恒按 d-1 映射 —— 3D 中间维 keepdim 一直算错(存量 bug)
                const int64_t rd = keepdim ? d : (d > dim ? d - 1 : d);
                ro += idx[d] * os[rd];
            }
        }
        op[ro] += xp[i];
        for (int64_t d = x.dim_count() - 1; d >= 0; --d) {
            if (++idx[d] < x.size(d)) break;
            idx[d] = 0;
        }
    }
}

Tensor sum_along(const Tensor& x, int64_t dim, bool keepdim) {
    Shape out;
    for (int64_t d = 0; d < x.dim_count(); ++d) {
        if (d != dim) out.dims[out.rank++] = x.size(d);
    }
    Tensor r = Tensor::zeros(keepdim ? Shape{} : out, x.dtype(), x.device());
    if (keepdim) {
        Shape k;
        for (int64_t d = 0; d < x.dim_count(); ++d) {
            k.dims[k.rank++] = d == dim ? 1 : x.size(d);
        }
        r = Tensor::zeros(k, x.dtype(), x.device());
    }
    sum_into(x, dim, keepdim, r);
    return r;
}

void sum_grad_into(const Tensor& g, int64_t dim, bool keepdim,
                   const Shape& in_shape, Tensor& out) {
    // 原生: 广播维(g 尺寸 1)步长置 0; CPU 走通用宿主循环
    long long dims[8], eff[8];
    const int64_t* gs = g.strides();
    const int rank = (int)in_shape.rank;
    for (int d = 0; d < rank; ++d) {
        dims[d] = in_shape.dims[d];
        const int64_t gd = (d == dim && !keepdim) ? -1
                                : (keepdim ? d : (d > dim ? d - 1 : d));
        eff[d] = (gd < 0 || g.size(gd) == 1) ? 0 : gs[gd];
    }
    grad_bcast_dispatch(g, out, out.numel(), rank, dims, eff, 1.f);
}

// ---- 设备无关内核派发(GPU 原生 / CPU 宿主循环) ----
void grad_bcast_dispatch(const Tensor& g, Tensor& out, long long total, int rank,
                         const long long* dims, const long long* eff, float scale) {
    switch (g.device()) {
    case Device::CUDA:
        ZNN_CUDA_CALL(ZNN_CUDA_K(grad_bcast_cuda)(cuda_ptr_const(g), cuda_ptr(out),
                                                  total, rank, dims, eff, scale));
        return;
    case Device::Vulkan:
        ZNN_VULKAN_CALL(vulkan_backend::grad_bcast_vulkan(g, out, total, rank,
                                                          dims, eff, scale));
        return;
    case Device::Metal:
        ZNN_METAL_CALL(metal_backend::grad_bcast_metal(g, out, total, rank,
                                                       dims, eff, scale));
        return;
    default: break;
    }
    const float* gp = g.data<float>();
    float* op = out.data<float>();
    int64_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t i = 0; i < total; ++i) {
        int64_t go = 0;
        for (int64_t d = 0; d < rank; ++d) {
            go += idx[d] * eff[d];
        }
        op[i] = scale * gp[go];
        for (int64_t d = rank - 1; d >= 0; --d) {
            if (++idx[d] < dims[d]) break;
            idx[d] = 0;
        }
    }
}

void strided_copy_dispatch(const Tensor& src, Tensor& dst, int rank,
                           const long long* ss, const long long* ds,
                           const long long* sh, long long total) {
    switch (src.device()) {
    case Device::CUDA:
        ZNN_CUDA_CALL(ZNN_CUDA_K(strided_copy_cuda)(cuda_ptr_const(src),
                                                    cuda_ptr(dst), rank, ss, ds,
                                                    sh, total));
        return;
    case Device::Vulkan:
        ZNN_VULKAN_CALL(vulkan_backend::strided_copy_vulkan(src, dst, rank, ss,
                                                            ds, sh, total));
        return;
    case Device::Metal:
        ZNN_METAL_CALL(metal_backend::strided_copy_metal(src, dst, rank, ss, ds,
                                                         sh, total));
        return;
    default: break;
    }
    const float* sp = src.data<float>();
    float* dp = dst.data<float>();
    int64_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t i = 0; i < total; ++i) {
        int64_t so = 0, doff = 0;              // 双步长: dst 可为非连续视图(散射)
        for (int64_t d = 0; d < rank; ++d) {
            so += idx[d] * ss[d];
            doff += idx[d] * ds[d];
        }
        dp[doff] = sp[so];
        for (int64_t d = rank - 1; d >= 0; --d) {
            if (++idx[d] < sh[d]) break;
            idx[d] = 0;
        }
    }
}

void strided_copy_raw(Device dev, const void* src, int64_t soff, void* dst,
                      int64_t doff, int rank, const long long* ss,
                      const long long* ds, const long long* sh, long long total) {
    switch (dev) {
    case Device::CUDA:
        ZNN_CUDA_CALL(ZNN_CUDA_K(strided_copy_cuda)(
            static_cast<const float*>(src) + soff, static_cast<float*>(dst) + doff,
            rank, ss, ds, sh, total));
        return;
    case Device::Vulkan:
        if (soff || doff) {
            throw std::runtime_error("znn: vulkan strided copy offset unsupported");
        }
        ZNN_VULKAN_CALL(vulkan_backend::strided_copy_raw_vk(src, dst, rank, ss,
                                                            ds, sh, total));
        return;
    case Device::Metal:
        if (soff || doff) {
            throw std::runtime_error("znn: metal strided copy offset unsupported");
        }
        ZNN_METAL_CALL(metal_backend::strided_copy_raw_mt(src, dst, rank, ss, ds,
                                                          sh, total));
        return;
    default: break;
    }
    const float* sp = static_cast<const float*>(src) + soff;
    float* dp = static_cast<float*>(dst) + doff;
    int64_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t i = 0; i < total; ++i) {
        int64_t so = 0, doff2 = 0;
        for (int64_t d = 0; d < rank; ++d) {
            so += idx[d] * ss[d];
            doff2 += idx[d] * ds[d];
        }
        dp[doff2] = sp[so];
        for (int64_t d = rank - 1; d >= 0; --d) {
            if (++idx[d] < sh[d]) break;
            idx[d] = 0;
        }
    }
}

void infer_reduce(Span<const TensorMeta*> inputs, const void* options, std::vector<Shape>& outs) {
    const ReduceAttrs* a = static_cast<const ReduceAttrs*>(options);
    const Shape& s = *inputs[0]->shape;
    const int64_t dim = a->dim < 0 ? a->dim + s.rank : a->dim;
    Shape out;
    for (int64_t d = 0; d < s.rank; ++d) {
        if (a->keepdim) {
            out.dims[out.rank++] = d == dim ? 1 : s[d];
        } else if (d != dim) {
            out.dims[out.rank++] = s[d];
        }
    }
    outs.push_back(out);
}

} // namespace znn
