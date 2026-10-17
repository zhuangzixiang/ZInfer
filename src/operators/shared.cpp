// shared.cpp — 逐元素类算子的共用实现
#include "shared.hpp"
#include "op_dispatcher.hpp"
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
    {nullptr, cuda_backend::unary_bwd_cuda, nullptr, nullptr},
    sizeof(UnaryBwd),
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
    const float* xp = x.data<float>();
    float* op = out.data<float>();
    const int64_t* xs = x.strides();
    const int64_t* os = out.strides();
    int64_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t i = 0; i < x.numel(); ++i) {
        int64_t ro = 0;
        for (int64_t d = 0; d < x.dim_count(); ++d) {
            if (d != dim) {
                const int64_t rd = d > dim ? d - 1 : d;
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
    const float* gp = g.data<float>();
    float* op = out.data<float>();
    const int64_t* gs = g.strides();
    int64_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t i = 0; i < out.numel(); ++i) {
        int64_t go = 0;
        for (int64_t d = 0; d < in_shape.rank; ++d) {
            if (d == dim && !keepdim) continue;   // 被求和维不贡献梯度偏移
            const int64_t gd = keepdim ? d : (d > dim ? d - 1 : d);
            go += idx[d] * gs[gd];
        }
        op[i] = gp[go];
        for (int64_t d = in_shape.rank - 1; d >= 0; --d) {
            if (++idx[d] < in_shape.dims[d]) break;
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
