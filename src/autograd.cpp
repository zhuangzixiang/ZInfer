// autograd.cpp — 自动求导引擎实现
// 铁律: 梯度规则经算子组合自动走当前后端, 无 CPU 桥接
#include "autograd.hpp"
#include "znn_internal.hpp"
#include <stdexcept>
#include <vector>

namespace znn {

AutogradEngine& AutogradEngine::instance() {
    static AutogradEngine engine;
    return engine;
}

GraphNode* AutogradEngine::record(const Operator* op,
                                  Span<const Tensor*> inputs,
                                  Span<const Tensor*> outputs,
                                  const void* options) {
    GraphNode* n = new GraphNode();       // 节点池: v1.5 性能项
    n->op = op;
    if (op->options_size > 0 && options) {
        n->options_copy.resize(op->options_size);
        std::memcpy(n->options_copy.data(), options, op->options_size);
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        n->saved_inputs.push_back(*inputs[i]);   // 句柄共享, 不拷贝数据
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        n->saved_outputs.push_back(*outputs[i]);
    }
    tape_.push_back(n);                   // 创建序 = 拓扑序
    return n;
}

void AutogradEngine::backward(const Tensor& loss) {
    if (loss.numel() != 1) {
        throw std::runtime_error("znn: backward on non-scalar");
    }
    Tensor& target = const_cast<Tensor&>(loss);   // 种子梯度必须写入
    target.grad() = Tensor::ones(loss.shape(), loss.dtype(), loss.device());

    // 反向期间暂停建图: 梯度规则内组合的前向算子不再记录
    const bool was_enabled = Runtime::grad_enabled();
    Runtime::set_grad_enabled(false);

    for (auto it = tape_.rbegin(); it != tape_.rend(); ++it) {   // 逆拓扑序
        GraphNode* n = *it;
        if (!n->op->grad) {
            continue;
        }
        // 1. 分配本节点各输入的梯度缓冲(跟随输入设备)
        const size_t nin = n->saved_inputs.size();
        std::vector<Tensor> in_grads(nin);
        for (size_t i = 0; i < nin; ++i) {
            in_grads[i] = Tensor::zeros(n->saved_inputs[i].shape(),
                                        n->saved_inputs[i].dtype(),
                                        n->saved_inputs[i].device());
        }
        // 2. 装配上下文(直接用原始张量; 梯度规则经算子组合自动走当前后端)
        std::vector<const Tensor*> si(nin), so(n->saved_outputs.size()),
                                    go(n->saved_outputs.size());
        std::vector<Tensor*> gi(nin);
        for (size_t i = 0; i < nin; ++i) {
            si[i] = &n->saved_inputs[i];
            gi[i] = &in_grads[i];
        }
        for (size_t i = 0; i < so.size(); ++i) {
            so[i] = &n->saved_outputs[i];
            go[i] = &n->saved_outputs[i].grad();
        }
        BackwardContext ctx;
        ctx.saved_inputs  = Span<const Tensor*>(si.data(), si.size());
        ctx.saved_outputs = Span<const Tensor*>(so.data(), so.size());
        ctx.grad_outputs  = Span<const Tensor*>(go.data(), go.size());
        ctx.grad_inputs   = Span<Tensor*>(gi.data(), gi.size());
        ctx.options       = n->options_copy.empty() ? nullptr : n->options_copy.data();
        n->op->grad(ctx);
        // 3. 累加到各输入(叶子得到最终梯度; 中间量等待其生产节点读取)
        for (size_t i = 0; i < nin; ++i) {
            if (n->saved_inputs[i].requires_grad()) {
                Tensor& acc = n->saved_inputs[i].grad();
                acc = Tensor::add(acc, in_grads[i]);
            }
        }
    }

    Runtime::set_grad_enabled(was_enabled);

    // 用毕释放整图(v1 不支持 retain_graph)
    for (GraphNode* n : tape_) {
        delete n;
    }
    tape_.clear();
}

namespace {

// 沿 dim 求和(keepdim 控制保留长度 1 的维), CPU 直写
Tensor sum_along(const Tensor& x, int64_t dim, bool keepdim) {
    Shape out;
    for (int64_t d = 0; d < x.dim_count(); ++d) {
        if (d != dim) {
            out.dims[out.rank++] = x.size(d);
        }
    }
    Tensor r = Tensor::zeros(out, x.dtype(), x.device());
    const float* xp = x.data<float>();
    float* rp = r.data<float>();
    const int64_t* xs = x.strides();
    const int64_t* rs = r.strides();
    int64_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int64_t i = 0; i < x.numel(); ++i) {
        int64_t ro = 0;
        for (int64_t d = 0; d < x.dim_count(); ++d) {
            if (d != dim) {
                const int64_t rd = d > dim ? d - 1 : d;
                ro += idx[d] * rs[rd];
            }
        }
        rp[ro] += xp[i];
        for (int64_t d = x.dim_count() - 1; d >= 0; --d) {
            if (++idx[d] < x.size(d)) break;
            idx[d] = 0;
        }
    }
    return keepdim ? r.unsqueeze(dim) : r;
}

} // namespace

Tensor BackwardContext::reduce_to(const Tensor& grad, const Shape& target_shape) {
    if (grad.shape() == target_shape) {
        return grad;
    }
    Tensor g = grad;
    while (g.dim_count() > target_shape.rank) {            // 去掉前导维
        g = sum_along(g, 0, false);
    }
    for (int64_t d = 0; d < target_shape.rank; ++d) {      // 尺寸 1 的维求和
        if (target_shape.dims[d] == 1 && g.size(d) != 1) {
            g = sum_along(g, d, true);
        }
    }
    if (!(g.shape() == target_shape)) {
        throw std::runtime_error("znn: reduce_to shape mismatch");
    }
    return g;
}

} // namespace znn
