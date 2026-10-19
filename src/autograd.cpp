// autograd.cpp — 可微计算图(TensorTape)实现
// 铁律: 梯度规则经算子组合自动走当前后端, 无 CPU 桥接
#include "autograd.hpp"
#include <cstdlib>
#include "znn_internal.hpp"
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace znn {

namespace {
const bool trace_bw = std::getenv("ZNN_TRACE_BW") != nullptr;   // 反向轨迹开关
} // namespace

void TensorTape::attach(Tensor& t, std::unique_ptr<TensorTape> node) {
    if (!t.impl_) {
        throw std::runtime_error("znn: tape attach on empty tensor");
    }
    t.impl_->tape = std::move(node);
}

TensorTape* TensorTape::of(const Tensor& t) {
    return t.impl_ ? t.impl_->tape.get() : nullptr;
}

void record_tape(Tensor& out, const Operator* op, Span<const Tensor*> inputs,
                 const void* options) {
    auto node = std::make_unique<TensorTape>();
    node->op = op;
    node->inputs.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        node->inputs.push_back(*inputs[i]);   // 句柄共享(引用计数+1)
    }
    if (op->options_size > 0 && options) {
        const char* p = static_cast<const char*>(options);
        node->options.assign(p, p + op->options_size);
    }
    TensorTape::attach(out, std::move(node));
}

namespace {

// DFS 收集本图节点(按 Tape 地址去重); 后序 = 拓扑序(叶子在前, 损失在尾)
void collect(const Tensor& t, std::unordered_set<const void*>& seen,
             std::vector<std::pair<const Tensor*, TensorTape*>>& order) {
    TensorTape* tape = TensorTape::of(t);
    if (!tape || !seen.insert(tape).second) {
        return;                               // 叶子或已收集(多下游共享)
    }
    for (const Tensor& in : tape->inputs) {
        collect(in, seen, order);
    }
    order.emplace_back(&t, tape);
}

} // namespace

void run_backward(const Tensor& loss) {
    if (loss.numel() != 1) {
        throw std::runtime_error("znn: backward on non-scalar");
    }
    Tensor& target = const_cast<Tensor&>(loss);   // 种子梯度必须写入
    target.grad() = Tensor::ones(loss.shape(), loss.dtype(), loss.device());

    // 反向期间暂停建图: 梯度规则内组合的前向算子不再记录
    // (它们产生新张量/新 Tape, 不属于本图 —— 结构上无法干扰遍历)
    const bool was_enabled = Runtime::grad_enabled();
    Runtime::set_grad_enabled(false);

    std::vector<std::pair<const Tensor*, TensorTape*>> order;
    std::unordered_set<const void*> seen;
    collect(loss, seen, order);

    for (auto it = order.rbegin(); it != order.rend(); ++it) {   // 逆拓扑
        const Tensor& owner = *it->first;
        TensorTape* n = it->second;
        if (!n->op->grad) {
            continue;
        }
        // 1. 分配本节点各输入的梯度缓冲(跟随输入设备)
        const size_t nin = n->inputs.size();
        std::vector<Tensor> in_grads(nin);
        for (size_t i = 0; i < nin; ++i) {
            in_grads[i] = Tensor::zeros(n->inputs[i].shape(),
                                        n->inputs[i].dtype(),
                                        n->inputs[i].device());
        }
        // 2. 装配上下文(saved_outputs = Tape 的宿主张量本身)
        std::vector<const Tensor*> si(nin), so(1), go(1);
        std::vector<Tensor*> gi(nin);
        for (size_t i = 0; i < nin; ++i) {
            si[i] = &n->inputs[i];
            gi[i] = &in_grads[i];
        }
        so[0] = &owner;
        go[0] = &const_cast<Tensor&>(owner).grad();
        BackwardContext ctx;
        ctx.saved_inputs  = Span<const Tensor*>(si.data(), si.size());
        ctx.saved_outputs = Span<const Tensor*>(so.data(), so.size());
        ctx.grad_outputs  = Span<const Tensor*>(go.data(), go.size());
        ctx.grad_inputs   = Span<Tensor*>(gi.data(), gi.size());
        ctx.options       = n->options.empty() ? nullptr : n->options.data();
        if (trace_bw) {
            std::fprintf(stderr, "[bw:%s nin=%d] ", n->op->name, (int)nin);
        }
        n->op->grad(ctx);
        if (trace_bw) {
            std::fprintf(stderr, "[done] ");
        }
        // 3. 累加到各输入(叶子得到最终梯度; 中间量等待其生产节点读取)
        //    非 F32 输入(如 embedding 的 I64 indices)不可训练, 跳过
        for (size_t i = 0; i < nin; ++i) {
            if (n->inputs[i].requires_grad()
                && n->inputs[i].dtype() == DType::F32) {
                Tensor& acc = n->inputs[i].grad();
                acc = Tensor::add(acc, in_grads[i]);
            }
        }
    }
    Runtime::set_grad_enabled(was_enabled);
}

Tensor BackwardContext::reduce_to(const Tensor& grad, const Shape& target_shape) {
    if (grad.shape() == target_shape) {
        return grad;
    }
    Tensor g = grad;
    while (g.dim_count() > target_shape.rank) {            // 去掉前导维
        g = Tensor::sum(g, 0, false);
    }
    for (int64_t d = 0; d < target_shape.rank; ++d) {      // 尺寸 1 的维求和
        if (target_shape.dims[d] == 1 && g.size(d) != 1) {
            g = Tensor::sum(g, d, true);
        }
    }
    if (!(g.shape() == target_shape)) {
        throw std::runtime_error("znn: reduce_to shape mismatch");
    }
    return g;
}

} // namespace znn
