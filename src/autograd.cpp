// autograd.cpp — 自动求导引擎实现
#include "autograd.hpp"
#include "operators/shared.hpp"
#include "znn_internal.hpp"
#include "backends/cuda/cuda_backend.hpp"
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
        // 1.5 GPU 梯度规则桥接: 梯度规则当前为 CPU 实现, CUDA 张量
        //     先暂存到 CPU 副本, 规则跑完后结果拷回原设备。
        //     (架构不变: GradientFunction 签名不动; CUDA 原生梯度内核
        //      按算子槽位模式后续补齐后此桥接自动短路)
        bool has_cuda = false;
        for (size_t i = 0; i < nin; ++i) {
            has_cuda = has_cuda || n->saved_inputs[i].device() == Device::CUDA;
        }
        std::vector<Tensor> cpu_saved_in, cpu_saved_out, cpu_grad_out, cpu_grad_in;
        if (has_cuda) {
            cpu_saved_in.resize(nin);
            cpu_saved_out.resize(n->saved_outputs.size());
            cpu_grad_out.resize(n->saved_outputs.size());
            cpu_grad_in.resize(nin);
            for (size_t i = 0; i < nin; ++i) {
                cpu_saved_in[i] = n->saved_inputs[i].to_device(Device::CPU);
                cpu_grad_in[i] = Tensor::zeros(n->saved_inputs[i].shape(),
                                               n->saved_inputs[i].dtype(), Device::CPU);
            }
            for (size_t i = 0; i < n->saved_outputs.size(); ++i) {
                cpu_saved_out[i] = n->saved_outputs[i].to_device(Device::CPU);
                cpu_grad_out[i] = n->saved_outputs[i].grad().to_device(Device::CPU);
            }
        }
        // 2. 装配上下文
        std::vector<const Tensor*> si(nin), so(n->saved_outputs.size()), go(n->saved_outputs.size());
        std::vector<Tensor*> gi(nin);
        for (size_t i = 0; i < nin; ++i) {
            si[i] = has_cuda ? &cpu_saved_in[i] : &n->saved_inputs[i];
            gi[i] = has_cuda ? &cpu_grad_in[i] : &in_grads[i];
        }
        for (size_t i = 0; i < so.size(); ++i) {
            so[i] = has_cuda ? &cpu_saved_out[i] : &n->saved_outputs[i];
            go[i] = has_cuda ? &cpu_grad_out[i] : &n->saved_outputs[i].grad();
        }
        BackwardContext ctx;
        ctx.saved_inputs  = Span<const Tensor*>(si.data(), si.size());
        ctx.saved_outputs = Span<const Tensor*>(so.data(), so.size());
        ctx.grad_outputs  = Span<const Tensor*>(go.data(), go.size());
        ctx.grad_inputs   = Span<Tensor*>(gi.data(), gi.size());
        ctx.options       = n->options_copy.empty() ? nullptr : n->options_copy.data();
        n->op->grad(ctx);
        // 2.5 GPU 桥接回拷: CPU 梯度结果 → 原设备
        if (has_cuda) {
            for (size_t i = 0; i < nin; ++i) {
                const Device dev = n->saved_inputs[i].device();
                if (dev == Device::CUDA && n->saved_inputs[i].requires_grad()) {
                    Tensor& acc = n->saved_inputs[i].grad();
                    Tensor cpu_sum = Tensor::add(acc.to_device(Device::CPU), cpu_grad_in[i]);
                    acc = cpu_sum.to_device(Device::CUDA);
                }
            }
        }
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
