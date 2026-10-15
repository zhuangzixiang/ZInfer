// cross_entropy: softmax + 负对数似然(分类标准损失), 返回标量
#include "znn_ext.hpp"

namespace znn {

extern const OpDef g_cross_entropy_def;
const OpDef g_cross_entropy_def = {
    "cross_entropy",
    nullptr,   // infer: 实现阶段补
    nullptr,   // grad: 实现阶段补
    nullptr,   // ref(对拍基准)
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Loss::cross_entropy(const Tensor& logits, const Tensor& targets) {
    ZNN_TODO();
}

} // namespace znn
