// embedding: 按索引取行(indices: I64)
#include "op_def.hpp"
#include "znn_internal.hpp"

namespace znn {

extern const OpDef g_embedding_def;
const OpDef g_embedding_def = {
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
