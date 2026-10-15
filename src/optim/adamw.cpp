// AdamW: 与 Adam 仅解耦衰减语义(衰减直接作用于参数, 不经动量/梯度)
#include "optim.hpp"
#include "znn_internal.hpp"

namespace znn {

void AdamWImpl::step() {
    // 1. 解耦衰减: p *= (1 - lr·wd)
    const float d = 1.f - static_cast<float>(param_.lr) * static_cast<float>(param_.weight_decay);
    if (param_.weight_decay > 0) {
        for (Tensor& p : params_) {
            float* pp = p.data<float>();
            for (int64_t k = 0; k < p.numel(); ++k) pp[k] *= d;
        }
    }
    // 2. 动量更新复用 Adam 步, 但梯度中不再含 L2 项
    const double saved = param_.weight_decay;
    param_.weight_decay = 0.0;
    AdamImpl::step();
    param_.weight_decay = saved;
}

} // namespace znn
