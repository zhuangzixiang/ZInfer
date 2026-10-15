// matmul: 2D 矩阵乘(v1 仅 rank-2); 梯度 dA=dY@Bᵀ, dB=dYᵀ@A 经算子组合
#include "shared.hpp"
#include "op_dispatcher.hpp"

namespace znn {

extern const Operator g_matmul_operator;

static void matmul_infer(Span<const TensorMeta*> in, const void*, std::vector<Shape>& outs) {
    const Shape& a = *in[0]->shape;
    const Shape& b = *in[1]->shape;
    if (a.rank != 2 || b.rank != 2 || a[1] != b[0]) {
        throw std::runtime_error("znn: matmul expects [M,K]x[K,N]");
    }
    outs.push_back(Shape{a[0], b[1]});
}

static void matmul_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    const Tensor& a = *in[0];
    const Tensor& b = *in[1];
    Tensor& y = *out[0];
    const int64_t M = a.size(0), K = a.size(1), N = b.size(1);
    const float* ap = a.data<float>();
    const float* bp = b.data<float>();
    float* yp = y.data<float>();
    const int64_t* as = a.strides();
    const int64_t* bs = b.strides();
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int64_t k = 0; k < K; ++k) {
                acc += ap[m * as[0] + k * as[1]] * bp[k * bs[0] + n * bs[1]];
            }
            yp[m * N + n] = acc;
        }
    }
}

static void matmul_grad(BackwardContext& c) {
    const Tensor& g = *c.grad_outputs[0];
    const Tensor& a = *c.saved_inputs[0];
    const Tensor& b = *c.saved_inputs[1];
    *c.grad_inputs[0] = Tensor::matmul(g, b.transpose(0, 1));
    *c.grad_inputs[1] = Tensor::matmul(a.transpose(0, 1), g);
}

const Operator g_matmul_operator = {
    "matmul",
    matmul_infer,
    matmul_grad,
    matmul_ref,
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::matmul(const Tensor& a, const Tensor& b) {
    const Tensor* ins[2] = {&a, &b};
    return OpDispatcher::instance().run(&g_matmul_operator, Span<const Tensor*>(ins, 2), nullptr);
}

} // namespace znn
