// linear: x @ w.T + b(w 形状 [out, in])
#include "op_def.hpp"
#include "znn_internal.hpp"

namespace znn {

extern const OpDef g_linear_def;
const OpDef g_linear_def = {
    "linear",
    nullptr,   // infer: 实现阶段补
    nullptr,   // grad: 实现阶段补
    nullptr,   // ref(对拍基准)
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::linear(const Tensor& x, const Tensor& w, const Tensor* b) {
    ZNN_TODO();
}

} // namespace znn
