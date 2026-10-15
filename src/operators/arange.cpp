// arange: 等差序列 [0, n)
#include "shared.hpp"
#include "op_dispatcher.hpp"

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
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::arange(int64_t n, DType dtype, Device device) {
    // 无输入算子不经派发(派发器要求输入提供 dtype/device), 直接构造
    Tensor t = Tensor::empty(Shape{n}, dtype, device);
    float* p = t.data<float>();
    for (int64_t i = 0; i < n; ++i) p[i] = static_cast<float>(i);
    return t;
}

} // namespace znn
