// op_dispatcher.cpp — 派发器实现(索引构建为结构性实现, 派发链为占位)
#include "op_dispatcher.hpp"
#include "autograd.hpp"
#include "backend_slots.hpp"
#include "znn_internal.hpp"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace znn {

namespace {

// 算子调用栈(异常可追溯): 入栈/出栈零分配, 仅异常路径打印。
// thread_local 是必要例外——嵌套算子调用(梯度规则组合前向)的深度
// 与归属线程一一对应, 全局栈在多线程下会串线
thread_local std::vector<const char*> t_op_stack;

struct OpScope {
    const char* name;
    explicit OpScope(const char* n) : name(n) { t_op_stack.push_back(n); }
    ~OpScope() { t_op_stack.pop_back(); }
};

// 打印"算子 ← 调用者 ← ..."轨迹到 stderr(抛出前调用, 未捕获异常
// 也已在崩溃前留下现场 —— 0xC0000409 fastfail 会吞掉异常文本)
void print_trace(const char* op, const char* what) {
    std::fprintf(stderr, "znn ERROR: op '%s' failed: %s\n", op, what);
    for (size_t i = t_op_stack.size() - 1; i > 0; --i) {
        std::fprintf(stderr, "  at '%s' <- '%s'\n", t_op_stack[i],
                     t_op_stack[i - 1]);
    }
}

// CPU 中间结果拷回原设备输出(经 Storage 直写, 保持输出张量身份)
void Impl_copy_back(const Tensor& cpu_src, Tensor& dst) {
    ZNN_CUDA_CALL(cuda_backend::copy_host_to_device(dst.data<float>(),
                                     cpu_src.data<float>(),
                                     static_cast<size_t>(cpu_src.numel()) * sizeof(float)));
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
    // 0. 设备一致性校验(混合设备输入此前直达内核 = 越界读崩溃, 存量缺口)
    for (size_t i = 1; i < inputs.size(); ++i) {
        if (inputs[i]->device() != inputs[0]->device()) {
            throw std::runtime_error(
                std::string("znn: op '") + op->name
                + "' inputs on different devices (input#0 vs input#"
                + std::to_string(i) + "); use to_device() to align");
        }
    }
    OpScope scope(op->name);
    try {
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
    } catch (const std::exception& e) {
        print_trace(op->name, e.what());
        throw std::runtime_error(std::string("znn: op '") + op->name
                                 + "' failed: " + e.what());
    }
}

} // namespace znn
