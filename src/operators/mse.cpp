// mse: 均方误差(同形) → 标量; 梯度 = 2(pred-target)/N
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backend_slots.hpp"

namespace znn {

extern const Operator g_mse_operator;

static void mse_infer(Span<const TensorMeta*>, const void*, std::vector<Shape>& outs) {
    outs.push_back(Shape{});
}

static void mse_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    const Tensor& p = *in[0];
    const Tensor& t = *in[1];
    const float* pp = p.data<float>();
    const float* tp = t.data<float>();
    float acc = 0.f;
    for (int64_t i = 0; i < p.numel(); ++i) {
        const float d = pp[i] - tp[i];
        acc += d * d;
    }
    out[0]->data<float>()[0] = acc / static_cast<float>(p.numel());
}

static void mse_grad(BackwardContext& c) {
    const Tensor& p = *c.saved_inputs[0];
    const Tensor& t = *c.saved_inputs[1];
    // 组合式: g_pred = (2/N) × g × (pred − target), 全部经算子派发(任意后端)
    Tensor coef = Tensor::mul(*c.grad_outputs[0],
                              Tensor::full(Shape{}, 2.0 / (double)p.numel(),
                                           DType::F32, p.device()));
    *c.grad_inputs[0] = Tensor::mul(Tensor::sub(p, t), coef);
}

const Operator g_mse_operator = {
    "mse",
    mse_infer,
    mse_grad,
    mse_ref,
    {nullptr, ZNN_CUDA_K(mse_cuda), nullptr, nullptr},
};

Tensor Loss::mse(const Tensor& pred, const Tensor& target) {
    const Tensor* ins[2] = {&pred, &target};
    return OpDispatcher::instance().run(&g_mse_operator, Span<const Tensor*>(ins, 2), nullptr);
}

} // namespace znn
