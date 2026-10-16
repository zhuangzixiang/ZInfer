// op_dispatcher.cpp — 派发器实现(索引构建为结构性实现, 派发链为占位)
#include "op_dispatcher.hpp"
#include "autograd.hpp"
#include "backends/cuda/cuda_backend.hpp"
#include "znn_internal.hpp"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace znn {

namespace {
// CPU 中间结果拷回原设备输出(经 Storage 直写, 保持输出张量身份)
void Impl_copy_back(const Tensor& cpu_src, Tensor& dst) {
    cuda_backend::copy_host_to_device(dst.data<float>(), cpu_src.data<float>(),
                                      static_cast<size_t>(cpu_src.numel()) * sizeof(float));
}
} // namespace

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

void OpDispatcher::set_precision(Precision p) { precision_ = p; }
Precision OpDispatcher::precision() const { return precision_; }

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
    KernelContext ctx{dev, nullptr, precision_};
    const bool fallback = dev != Device::CPU && kernel == op->ref;
    if (!fallback) {
        kernel(ctx, inputs, Span<Tensor*>(outs, 1), options);
    } else {
        // 透明降级: 拷回 CPU 执行 ref, 结果拷回原设备(带一次性告警)
        static bool warned = false;
        if (!warned) {
            std::fprintf(stderr, "znn: op '%s' has no native kernel, falling back to CPU",
                         op->name);
            warned = true;
        }
        std::vector<Tensor> cpu_ins(inputs.size());
        std::vector<const Tensor*> cpu_ptrs(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            cpu_ins[i] = inputs[i]->to_device(Device::CPU);
            cpu_ptrs[i] = &cpu_ins[i];
        }
        Tensor cpu_out = Tensor::empty(out.shape(), out.dtype(), Device::CPU);
        Tensor* cpu_outs[1] = {&cpu_out};
        KernelContext cpu_ctx{Device::CPU, nullptr, precision_};
        kernel(cpu_ctx, Span<const Tensor*>(cpu_ptrs.data(), cpu_ptrs.size()),
               Span<Tensor*>(cpu_outs, 1), options);
        Impl_copy_back(cpu_out, out);
    }
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
