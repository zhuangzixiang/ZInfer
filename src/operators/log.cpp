// log: 逐元素一元运算()
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backends/cuda/cuda_backend.hpp"

namespace znn {

extern const Operator g_log_operator;

static void log_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*);

static void log_cpu(KernelContext& c, Span<const Tensor*> in, Span<Tensor*> out,
                        const void* o) {
    if (!onednn::eltwise(in, out, (int)dnnl_eltwise_log, 0.f, 0.f)) {
        log_ref(c, in, out, o);   // 缺 DLL 回落(Eigen 快速路径)
    }
}


static void log_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void*) {
    unary_kernel_fast(in, out, [](float x) { return logf(x); });
}

static void log_grad(BackwardContext& c) {
    unary_grad(c, UnaryBwd::Log);
}

const Operator g_log_operator = {
    "log",
    infer_same,
    log_grad,
    log_ref,
    {log_cpu, nullptr, nullptr},
};

Tensor Tensor::log(const Tensor& x) {
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_log_operator, Span<const Tensor*>(ins, 1), nullptr);
}

} // namespace znn
