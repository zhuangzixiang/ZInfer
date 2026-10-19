// arange: 等差序列 [0, n)
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backend_slots.hpp"

namespace znn {

extern const Operator g_arange_operator;

static void arange_ref(KernelContext&, Span<const Tensor*>, Span<Tensor*> out, const void*) {
    float* p = out[0]->data<float>();
    for (int64_t i = 0; i < out[0]->numel(); ++i) p[i] = static_cast<float>(i);
}

const Operator g_arange_operator = {
    "arange",
    nullptr,   // 前端直接构造, 不经派发
    nullptr,   // 无需求导
    arange_ref,
    {nullptr, ZNN_CUDA_K(arange_cuda), nullptr, nullptr},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::arange(int64_t n, DType dtype, Device device) {
    // 无输入算子: 以输出张量身份构造后直接调 ref / CUDA 内核
    Tensor t = Tensor::empty(Shape{n}, dtype, device);
    Tensor* outs[1] = {&t};
    KernelContext ctx{device, nullptr};
    if (device == Device::CUDA) {
        ZNN_CUDA_CALL(ZNN_CUDA_K(arange_cuda)(ctx, Span<const Tensor*>(nullptr, 0),
                                  Span<Tensor*>(outs, 1), nullptr));
    } else {
        float* p = t.data<float>();
        for (int64_t i = 0; i < n; ++i) p[i] = static_cast<float>(i);
    }
    return t;
}

} // namespace znn
