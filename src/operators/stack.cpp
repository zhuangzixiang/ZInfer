// stack: 沿新维度 dim 堆叠(所有输入同形); 梯度 = 沿 dim 取片再压回原形
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include <vector>

namespace znn {

extern const Operator g_stack_operator;

struct StackOptions {
    int64_t dim;
};

static void stack_infer(Span<const TensorMeta*> in, const void* options,
                        std::vector<Shape>& outs) {
    const Shape& first = *in[0]->shape;
    const int64_t dim = static_cast<const StackOptions*>(options)->dim < 0
        ? static_cast<const StackOptions*>(options)->dim + first.rank + 1
        : static_cast<const StackOptions*>(options)->dim;
    Shape out;
    for (int64_t d = 0; d < dim; ++d) out.dims[out.rank++] = first[d];
    out.dims[out.rank++] = static_cast<int64_t>(in.size());
    for (int64_t d = dim; d < first.rank; ++d) out.dims[out.rank++] = first[d];
    outs.push_back(out);
}

static void stack_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                      const void* options) {
    Tensor& y = *out[0];
    const int64_t dim = static_cast<const StackOptions*>(options)->dim < 0
        ? static_cast<const StackOptions*>(options)->dim + y.dim_count()
        : static_cast<const StackOptions*>(options)->dim;
    float* yp = y.data<float>();
    const int64_t* ys = y.strides();
    for (size_t i = 0; i < in.size(); ++i) {
        const Tensor& x = *in[i];
        const float* xp = x.data<float>();
        const int64_t* xs = x.strides();
        const int64_t rank = x.dim_count();
        int64_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int64_t e = 0; e < x.numel(); ++e) {
            int64_t xo = 0;
            int64_t yo = static_cast<int64_t>(i) * ys[dim];
            for (int64_t d = 0; d < rank; ++d) {
                xo += idx[d] * xs[d];
                yo += idx[d] * ys[d < dim ? d : d + 1];
            }
            yp[yo] = xp[xo];
            for (int64_t d = rank - 1; d >= 0; --d) {
                if (++idx[d] < x.size(d)) break;
                idx[d] = 0;
            }
        }
    }
}

static void stack_grad(BackwardContext& c) {
    const Tensor& g = *c.grad_outputs[0];
    const int64_t dim = g.dim_count() - c.saved_inputs[0]->dim_count() == 1
        ? 0 : 0;                                            // 见下: 由选项给出
    (void)dim;
    const StackOptions* o = static_cast<const StackOptions*>(c.options);
    for (size_t i = 0; i < c.saved_inputs.size(); ++i) {
        Tensor slot = g.slice(o->dim < 0 ? o->dim + g.dim_count() : o->dim,
                             static_cast<int64_t>(i), static_cast<int64_t>(i) + 1);
        *c.grad_inputs[i] = slot.reshape(c.saved_inputs[i]->shape());
    }
}

const Operator g_stack_operator = {
    "stack",
    stack_infer,
    stack_grad,
    stack_ref,
    {},                          // kernels[CPU/CUDA/Metal/Vulkan]
    sizeof(StackOptions),
};

Tensor Tensor::stack(const std::vector<Tensor>& tensors, int64_t dim) {
    if (tensors.empty()) {
        throw std::runtime_error("znn: stack needs inputs");
    }
    const StackOptions o{dim};
    std::vector<const Tensor*> ins;
    ins.reserve(tensors.size());
    for (const Tensor& t : tensors) {
        ins.push_back(&t);
    }
    return OpDispatcher::instance().run(&g_stack_operator,
                                        Span<const Tensor*>(ins.data(), ins.size()), &o);
}

} // namespace znn
