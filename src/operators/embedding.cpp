// embedding: 按索引取行(indices: I64 任意形状); 梯度 scatter-add 回权重
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backends/cuda/cuda_backend.hpp"

namespace znn {

extern const Operator g_embedding_operator;

static void embedding_infer(Span<const TensorMeta*> in, const void*,
                            std::vector<Shape>& outs) {
    const Shape& w = *in[0]->shape;                       // {V, D}
    const Shape& idx = *in[1]->shape;
    if (w.rank != 2) {
        throw std::runtime_error("znn: embedding weight expects [V, D]");
    }
    Shape out;
    for (int64_t d = 0; d < idx.rank; ++d) out.dims[out.rank++] = idx[d];
    out.dims[out.rank++] = w[1];
    outs.push_back(out);
}

static void embedding_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                          const void*) {
    const Tensor& w = *in[0];
    const Tensor& idx = *in[1];
    Tensor& y = *out[0];
    const int64_t D = w.size(1);
    const float* wp = w.data<float>();
    const int64_t* ip = idx.data<int64_t>();
    float* yp = y.data<float>();
    for (int64_t e = 0; e < idx.numel(); ++e) {
        const int64_t v = ip[e];
        if (v < 0 || v >= w.size(0)) {
            throw std::runtime_error("znn: embedding index out of range");
        }
        for (int64_t d = 0; d < D; ++d) {
            yp[e * D + d] = wp[v * D + d];
        }
    }
}

static void embedding_grad(BackwardContext& c) {
    const Tensor& idx = *c.saved_inputs[1];
    const Tensor& g = *c.grad_outputs[0];
    Tensor& dw = *c.grad_inputs[0];                       // 已零初始化
    const int64_t D = dw.size(1);
    const int64_t* ip = idx.data<int64_t>();
    const float* gp = g.data<float>();
    float* dwp = dw.data<float>();
    for (int64_t e = 0; e < idx.numel(); ++e) {
        const int64_t v = ip[e];
        for (int64_t d = 0; d < D; ++d) {
            dwp[v * D + d] += gp[e * D + d];
        }
    }
}

const Operator g_embedding_operator = {
    "embedding",
    embedding_infer,
    embedding_grad,
    embedding_ref,
    {nullptr, cuda_backend::embedding_cuda, nullptr, nullptr},
};

Tensor Tensor::embedding(const Tensor& weight, const Tensor& indices) {
    const Tensor* ins[2] = {&weight, &indices};
    return OpDispatcher::instance().run(&g_embedding_operator,
                                        Span<const Tensor*>(ins, 2), nullptr);
}

} // namespace znn
