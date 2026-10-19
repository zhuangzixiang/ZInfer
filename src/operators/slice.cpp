// slice: 训练态切片走本算子(材料化 + 梯度散射回原位置);
// 纯推理态 Tensor::slice 直接返回零拷贝视图
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backend_slots.hpp"
#include <cstring>

namespace znn {

extern const Operator g_slice_operator;

struct SliceOptions {
    int64_t dim;
    int64_t begin;
    int64_t end;
};

static void slice_infer(Span<const TensorMeta*> in, const void* options,
                        std::vector<Shape>& outs) {
    const SliceOptions* o = static_cast<const SliceOptions*>(options);
    Shape s = *in[0]->shape;
    const int64_t dim = o->dim < 0 ? o->dim + s.rank : o->dim;
    s.dims[dim] = o->end - o->begin;
    outs.push_back(s);
}

static void slice_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                      const void* options) {
    const SliceOptions* o = static_cast<const SliceOptions*>(options);
    const Tensor& x = *in[0];
    Tensor& y = *out[0];
    const int64_t dim = o->dim < 0 ? o->dim + x.dim_count() : o->dim;
    // 内核内禁用建图: 防止 x.slice() 在 requires_grad 输入上
    // 递归调回 slice_for_autograd 造成无限循环
    const bool was = Runtime::grad_enabled();
    Runtime::set_grad_enabled(false);
    Tensor view = x.slice(dim, o->begin, o->end);
    Runtime::set_grad_enabled(was);
    impl::copy_view_to(view, y);
}

static void slice_grad(BackwardContext& c) {
    const SliceOptions* o = static_cast<const SliceOptions*>(c.options);
    const Tensor& g = *c.grad_outputs[0];
    const Tensor& x = *c.saved_inputs[0];
    Tensor& dx = *c.grad_inputs[0];                       // 已零初始化
    const int64_t dim = o->dim < 0 ? o->dim + x.dim_count() : o->dim;
    // 梯度散射: dgrad 的切片区 = g(经视图写入共享存储)
    Tensor slot = dx.slice(dim, o->begin, o->end);        // 反向期间为纯视图
    impl::copy_view_to(g, slot);
}

const Operator g_slice_operator = {
    "slice",
    slice_infer,
    slice_grad,
    slice_ref,
    {nullptr, ZNN_CUDA_K(slice_cuda), nullptr, nullptr},
    sizeof(SliceOptions),
};

Tensor slice_for_autograd(const Tensor& x, int64_t dim, int64_t begin, int64_t end) {
    const SliceOptions o{dim, begin, end};
    const Tensor* ins[1] = {&x};
    return OpDispatcher::instance().run(&g_slice_operator,
                                        Span<const Tensor*>(ins, 1), &o);
}

} // namespace znn
