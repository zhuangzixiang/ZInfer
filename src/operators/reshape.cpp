// reshape: 训练态形状变换走本算子(材料化拷贝 + 梯度逆变换);
// 纯推理态 Tensor::reshape/view/flatten/squeeze/unsqueeze 直接返回零拷贝视图。
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backend_slots.hpp"

namespace znn {

extern const Operator g_reshape_operator;

struct ReshapeOptions {
    Shape target;
};

static void reshape_infer(Span<const TensorMeta*> in, const void* options,
                          std::vector<Shape>& outs) {
    (void)in;
    const ReshapeOptions* o = static_cast<const ReshapeOptions*>(options);
    outs.push_back(o->target);
}

static void reshape_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                        const void*) {
    // 紧凑化(逻辑行主序)后按目标形状视图, 句柄赋值共享存储
    const bool was = Runtime::grad_enabled();
    Runtime::set_grad_enabled(false);                       // 防递归
    Tensor flat = in[0]->clone();
    *out[0] = flat.reshape(out[0]->shape());
    Runtime::set_grad_enabled(was);
}

static void reshape_grad(BackwardContext& c) {
    // 梯度逆变换 = 上游梯度变回输入形状(反向期间 reshape 为纯视图)
    *c.grad_inputs[0] = c.grad_outputs[0]->reshape(c.saved_inputs[0]->shape());
}

const Operator g_reshape_operator = {
    "reshape",
    reshape_infer,
    reshape_grad,
    reshape_ref,
    {nullptr, ZNN_CUDA_K(reshape_cuda), ZNN_METAL_K(reshape_metal), ZNN_VULKAN_K(reshape_vulkan)},
    sizeof(ReshapeOptions),
};

Tensor reshape_for_autograd(const Tensor& x, Shape target) {
    const ReshapeOptions o{target};
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_reshape_operator,
                                        Span<const Tensor*>(ins, 1), &o);
}

} // namespace znn
