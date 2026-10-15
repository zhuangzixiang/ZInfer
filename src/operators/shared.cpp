// shared.cpp — 逐元素类算子的共用实现
#include "shared.hpp"
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

void unary_grad(BackwardContext& context, float (*d)(float, float)) {
    const Tensor& g = *context.grad_outputs[0];
    const Tensor& x = *context.saved_inputs[0];
    const Tensor& y = *context.saved_outputs[0];
    Tensor& gi = *context.grad_inputs[0];
    const float* gp = g.data<float>();
    const float* xp = x.data<float>();
    const float* yp = y.data<float>();
    float* ip = gi.data<float>();
    for (int64_t i = 0; i < gi.numel(); ++i) {
        ip[i] = gp[i] * d(xp[i], yp[i]);
    }
}

} // namespace znn
