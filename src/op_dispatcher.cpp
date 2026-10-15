// op_dispatcher.cpp — 派发器实现(索引构建为结构性实现, 派发链为占位)
#include "op_dispatcher.hpp"
#include "autograd.hpp"
#include "znn_internal.hpp"
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace znn {

OpDispatcher& OpDispatcher::instance() {
    static OpDispatcher dispatcher;
    return dispatcher;
}

// 遍历注册名单(operator_list.cpp)构建只读索引
OpDispatcher::OpDispatcher() {
    for (const Operator* const* it = g_operator_list; *it != nullptr; ++it) {
        table_.emplace((*it)->name, *it);
    }
}

const Operator* OpDispatcher::find(const char* name) const {
    auto it = table_.find(name);
    return it == table_.end() ? nullptr : it->second;
}

Tensor OpDispatcher::run(const Operator* op, Span<const Tensor*> inputs, const void* options) {
    if (inputs.size() == 0 || !op->infer) {
        throw std::runtime_error("znn: operator needs inputs and infer");
    }
    // 1. 名片与形状推理(契约收指针数组)
    std::vector<TensorMeta> meta_storage(inputs.size());
    std::vector<const TensorMeta*> metas(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        meta_storage[i] = TensorMeta{&inputs[i]->shape(), inputs[i]->dtype(), inputs[i]->device()};
        metas[i] = &meta_storage[i];
    }
    std::vector<Shape> out_shapes;
    op->infer(Span<const TensorMeta*>(metas.data(), metas.size()), options, out_shapes);
    if (out_shapes.empty()) {
        throw std::runtime_error("znn: infer produced no output");
    }
    // 2. 分配输出(v1 约定: 单输出, dtype/device 随首个输入)
    Tensor out = Tensor::empty(out_shapes[0], inputs[0]->dtype(), inputs[0]->device());
    Tensor* outs[1] = {&out};
    // 3. 内核选择: 后端槽位 → ref 兜底
    const Device dev = inputs[0]->device();
    KernelFunction kernel = op->kernels[static_cast<int>(dev)];
    static const bool force_ref = std::getenv("ZNN_FORCE_REF") != nullptr;   // 对拍/调试
    if (!kernel || force_ref) {
        kernel = op->ref;
    }
    if (!kernel) {
        throw std::runtime_error(std::string("znn: no kernel for operator ") + op->name);
    }
    KernelContext ctx{dev, nullptr};
    kernel(ctx, inputs, Span<Tensor*>(outs, 1), options);
    // 4. 训练时建图: 任一输入 requires_grad 则输出也标记
    if (Runtime::grad_enabled() && op->grad) {
        bool any = false;
        for (size_t i = 0; i < inputs.size(); ++i) {
            any = any || inputs[i]->requires_grad();
        }
        if (any) {
            out.requires_grad(true);
            const Tensor* out_c[1] = {&out};
            AutogradEngine::instance().record(op, inputs, Span<const Tensor*>(out_c, 1), options);
        }
    }
    return out;
}

} // namespace znn
