// gelu: 激活函数
#include "operator.hpp"
#include "znn_internal.hpp"

namespace znn {

extern const Operator g_gelu_operator;
const Operator g_gelu_operator = {
    "gelu",
    nullptr,   // infer: 实现阶段补
    nullptr,   // grad
    nullptr,   // ref(对拍基准)
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Activator::gelu(const Tensor& x) {
    ZNN_TODO();
}

} // namespace znn
