// cross_entropy: logits[N,C] + targets(I64)[N] → 标量;
// 前向 = 稳定 log_softmax 的负对数似然取平均; 梯度 = (softmax - onehot)/N
#include "shared.hpp"
#include "op_dispatcher.hpp"

namespace znn {

extern const Operator g_cross_entropy_operator;

static void ce_infer(Span<const TensorMeta*>, const void*, std::vector<Shape>& outs) {
    outs.push_back(Shape{});                             // 标量
}

static void ce_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    const Tensor& x = *in[0];
    const Tensor& t = *in[1];
    const int64_t n = x.size(0);
    const int64_t c = x.size(1);
    const float* xp = x.data<float>();
    const int64_t* tp = t.data<int64_t>();
    float loss = 0.f;
    for (int64_t i = 0; i < n; ++i) {
        const float* row = xp + i * c;
        float m = row[0];
        for (int64_t j = 1; j < c; ++j) m = m < row[j] ? row[j] : m;
        float lse = 0.f;
        for (int64_t j = 0; j < c; ++j) lse += expf(row[j] - m);
        lse = m + logf(lse);
        loss += lse - row[tp[i]];
    }
    out[0]->data<float>()[0] = loss / static_cast<float>(n);
}

static void ce_grad(BackwardContext& c) {
    const Tensor& x = *c.saved_inputs[0];
    const Tensor& t = *c.saved_inputs[1];
    Tensor& gi = *c.grad_inputs[0];
    const int64_t n = x.size(0);
    const int64_t cl = x.size(1);
    const float* xp = x.data<float>();
    const int64_t* tp = t.data<int64_t>();
    float* ip = gi.data<float>();
    const float scale = *c.grad_outputs[0]->data<float>() / static_cast<float>(n);
    for (int64_t i = 0; i < n; ++i) {
        const float* row = xp + i * cl;
        float* irow = ip + i * cl;
        float m = row[0];
        for (int64_t j = 1; j < cl; ++j) m = m < row[j] ? row[j] : m;
        float s = 0.f;
        for (int64_t j = 0; j < cl; ++j) {
            irow[j] = expf(row[j] - m);
            s += irow[j];
        }
        for (int64_t j = 0; j < cl; ++j) {
            irow[j] = (irow[j] / s - (j == tp[i] ? 1.f : 0.f)) * scale;
        }
    }
}

const Operator g_cross_entropy_operator = {
    "cross_entropy",
    ce_infer,
    ce_grad,
    ce_ref,
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Loss::cross_entropy(const Tensor& logits, const Tensor& targets) {
    const Tensor* ins[2] = {&logits, &targets};
    return OpDispatcher::instance().run(&g_cross_entropy_operator, Span<const Tensor*>(ins, 2), nullptr);
}

} // namespace znn
