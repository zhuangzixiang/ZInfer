// attention: scaled_dot_product_attention(q/k/v: [B,H,L,D]; causal 屏蔽未来位)
#include "op_def.hpp"
#include "znn_internal.hpp"

namespace znn {

extern const OpDef g_attention_def;
const OpDef g_attention_def = {
    "attention",
    nullptr,   // infer: 实现阶段补
    nullptr,   // grad: 实现阶段补
    nullptr,   // ref(对拍基准)
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::attention(const Tensor& q, const Tensor& k,
                         const Tensor& v, bool causal) {
    ZNN_TODO();
}

} // namespace znn
