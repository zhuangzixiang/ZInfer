// neg: 逐元素一元运算
#include "znn_ext.hpp"

namespace znn {

extern const OpDef g_neg_def;
const OpDef g_neg_def = {
    "neg",
    nullptr,   // infer: 实现阶段补
    nullptr,   // grad
    nullptr,   // ref(对拍基准)
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::neg(const Tensor& x) {
    ZNN_TODO();
}

} // namespace znn
