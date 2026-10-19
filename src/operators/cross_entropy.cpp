// cross_entropy: logits[N,C] + targets(I64)[N] → 标量;
// 前向 = 稳定 log_softmax 的负对数似然取平均; 梯度 = (softmax - onehot)/N
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backend_slots.hpp"

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
    // 组合实现: dx = (softmax(x) - onehot(t)) * g / N
    // 全部经算子(softmax/mul/sub), 自动走当前后端内核
    const Tensor& x = *c.saved_inputs[0];
    const Tensor& t = *c.saved_inputs[1];
    const Tensor& g = *c.grad_outputs[0];
    const int64_t n = x.size(0);
    const int64_t cl = x.size(1);
    // softmax(x): 走 softmax 算子(有 CUDA 内核)
    Tensor sm = Activator::softmax(x);
    // onehot(t): host 构造后按目标设备上传(数据准备, 非 CPU 桥接)
    Tensor t_cpu = t.to_device(Device::CPU);
    std::vector<float> oh_host(static_cast<size_t>(n * cl), 0.f);
    for (int64_t i = 0; i < n; ++i) {
        oh_host[i * cl + t_cpu.data<int64_t>()[i]] = 1.f;
    }
    Tensor oh = Tensor::from_host(oh_host.data(), Shape{n, cl}, DType::F32, x.device());
    // (sm - oh) * g / N: 全算子组合
    Tensor diff = Tensor::sub(sm, oh);
    Tensor scaled = Tensor::mul(diff, g);                  // g 标量广播
    *c.grad_inputs[0] = Tensor::mul(scaled,
        Tensor::full(Shape{}, 1.0 / static_cast<float>(n), DType::F32, x.device()));
}

const Operator g_cross_entropy_operator = {
    "cross_entropy",
    ce_infer,
    ce_grad,
    ce_ref,
    {nullptr, ZNN_CUDA_K(cross_entropy_cuda), nullptr, nullptr},
};

Tensor Loss::cross_entropy(const Tensor& logits, const Tensor& targets) {
    const Tensor* ins[2] = {&logits, &targets};
    return OpDispatcher::instance().run(&g_cross_entropy_operator, Span<const Tensor*>(ins, 2), nullptr);
}

} // namespace znn
