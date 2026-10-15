// sigmoid: 激活函数
#include "znn_ext.hpp"

namespace znn {

extern const OpDef g_sigmoid_def;
const OpDef g_sigmoid_def = {
    "sigmoid",
    nullptr,   // infer: 实现阶段补
    nullptr,   // grad
    nullptr,   // ref(对拍基准)
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Activator::sigmoid(const Tensor& x) {
    ZNN_TODO();
}

} // namespace znn
