// cat: 沿 dim 拼接(可变输入数); 梯度 = 按各输入区间切片回传
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include <vector>

namespace znn {

extern const Operator g_cat_operator;

struct CatOptions {
    int64_t dim;
};

static int64_t resolve_dim(int64_t dim, int64_t rank) {
    return dim < 0 ? dim + rank : dim;
}

static void cat_infer(Span<const TensorMeta*> in, const void* options,
                      std::vector<Shape>& outs) {
    const Shape& first = *in[0]->shape;
    const int64_t dim = resolve_dim(static_cast<const CatOptions*>(options)->dim, first.rank);
    Shape out = first;
    for (size_t i = 1; i < in.size(); ++i) {
        const Shape& s = *in[i]->shape;
        if (s.rank != out.rank) {
            throw std::runtime_error("znn: cat rank mismatch");
        }
        for (int64_t d = 0; d < out.rank; ++d) {
            if (d != dim && s[d] != out[d]) {
                throw std::runtime_error("znn: cat shape mismatch");
            }
        }
        out.dims[dim] += s[dim];
    }
    outs.push_back(out);
}

static void cat_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                    const void* options) {
    Tensor& y = *out[0];
    const int64_t dim = resolve_dim(static_cast<const CatOptions*>(options)->dim,
                                    y.dim_count());
    float* yp = y.data<float>();
    const int64_t* ys = y.strides();
    int64_t offset = 0;
    for (size_t i = 0; i < in.size(); ++i) {
        const Tensor& x = *in[i];
        const float* xp = x.data<float>();
        const int64_t* xs = x.strides();
        const int64_t rank = x.dim_count();
        int64_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int64_t e = 0; e < x.numel(); ++e) {
            int64_t xo = 0;
            int64_t yo = 0;
            for (int64_t d = 0; d < rank; ++d) {
                xo += idx[d] * xs[d];
                yo += (d == dim ? idx[d] + offset : idx[d]) * ys[d];
            }
            yp[yo] = xp[xo];
            for (int64_t d = rank - 1; d >= 0; --d) {
                if (++idx[d] < x.size(d)) break;
                idx[d] = 0;
            }
        }
        offset += x.size(dim);
    }
}

static void cat_grad(BackwardContext& c) {
    const int64_t dim = resolve_dim(static_cast<const CatOptions*>(c.options)->dim,
                                    c.grad_outputs[0]->dim_count());
    const Tensor& g = *c.grad_outputs[0];
    int64_t offset = 0;
    for (size_t i = 0; i < c.saved_inputs.size(); ++i) {
        const int64_t len = c.saved_inputs[i]->size(dim);
        Tensor slot = g.slice(dim, offset, offset + len);   // 反向期间为纯视图
        *c.grad_inputs[i] = slot.reshape(c.saved_inputs[i]->shape());
        offset += len;
    }
}

const Operator g_cat_operator = {
    "cat",
    cat_infer,
    cat_grad,
    cat_ref,
    {},                          // kernels[CPU/CUDA/Metal/Vulkan]
    sizeof(CatOptions),
};

Tensor Tensor::cat(const std::vector<Tensor>& tensors, int64_t dim) {
    if (tensors.empty()) {
        throw std::runtime_error("znn: cat needs inputs");
    }
    const CatOptions o{dim};
    std::vector<const Tensor*> ins;
    ins.reserve(tensors.size());
    for (const Tensor& t : tensors) {
        ins.push_back(&t);
    }
    return OpDispatcher::instance().run(&g_cat_operator,
                                        Span<const Tensor*>(ins.data(), ins.size()), &o);
}

Tensor Tensor::cat(const Tensor& a, const Tensor& b, int64_t dim) {
    return cat(std::vector<Tensor>{a, b}, dim);
}

} // namespace znn
