// sqrt: 逐元素一元运算
#include "operator.hpp"
#include "znn_internal.hpp"

namespace znn {

extern const Operator g_sqrt_operator;
const Operator g_sqrt_operator = {
    "sqrt",
    nullptr,   // infer: 实现阶段补
    nullptr,   // grad
    nullptr,   // ref(对拍基准)
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::sqrt(const Tensor& x) {
    ZNN_TODO();
}

} // namespace znn
