// softmax: 沿 dim 归一化指数
#include "operator.hpp"
#include "znn_internal.hpp"

namespace znn {

extern const Operator g_softmax_operator;
const Operator g_softmax_operator = {
    "softmax",
    nullptr,   // infer: 实现阶段补
    nullptr,   // grad: 实现阶段补
    nullptr,   // ref(对拍基准)
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Activator::softmax(const Tensor& x, int64_t dim) {
    ZNN_TODO();
}

} // namespace znn
