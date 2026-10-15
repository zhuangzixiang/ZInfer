// attention: scaled_dot_product_attention(q/k/v: [B,H,L,D]; causal 屏蔽未来位)
#include "operator.hpp"
#include "znn_internal.hpp"

namespace znn {

extern const Operator g_attention_operator;
const Operator g_attention_operator = {
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
