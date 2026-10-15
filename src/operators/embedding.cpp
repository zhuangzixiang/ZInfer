// embedding: 按索引取行(indices: I64)
#include "operator.hpp"
#include "znn_internal.hpp"

namespace znn {

extern const Operator g_embedding_operator;
const Operator g_embedding_operator = {
    "embedding",
    nullptr,   // infer: 实现阶段补
    nullptr,   // grad: 实现阶段补( scatter-add 到权重)
    nullptr,   // ref(对拍基准)
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::embedding(const Tensor& weight, const Tensor& indices) {
    ZNN_TODO();
}

} // namespace znn
