// Adam: 动量 + 偏差校正(L2 衰减并入梯度); AdamW 见 adamw.cpp
#include "optim.hpp"
#include "znn_internal.hpp"
#include <cmath>

namespace znn {

AdamImpl::AdamImpl(std::vector<Tensor> params, const OptimParam& param)
    : params_(std::move(params)), param_(param) {
    moments_.resize(params_.size());
}

void AdamImpl::step() {
    ++step_count_;
    const float b1 = static_cast<float>(param_.beta1);
    const float b2 = static_cast<float>(param_.beta2);
    const float bias1 = 1.f - std::pow(b1, static_cast<float>(step_count_));
    const float bias2 = 1.f - std::pow(b2, static_cast<float>(step_count_));
    for (size_t i = 0; i < params_.size(); ++i) {
        Tensor& p = params_[i];
        if (!p.grad()) {
            continue;
        }
        if (!moments_[i].m) {                     // 惰性分配动量缓冲
            moments_[i].m = Tensor::zeros(p.shape());
            moments_[i].v = Tensor::zeros(p.shape());
        }
        const int64_t n = p.numel();
        const float* g = p.grad().data<float>();
        float* pp = p.data<float>();
        float* m = moments_[i].m.data<float>();
        float* v = moments_[i].v.data<float>();
        for (int64_t k = 0; k < n; ++k) {
            const float grad = g[k] + static_cast<float>(param_.weight_decay) * pp[k];
            m[k] = b1 * m[k] + (1.f - b1) * grad;
            v[k] = b2 * v[k] + (1.f - b2) * grad * grad;
            const float mh = m[k] / bias1;
            const float vh = v[k] / bias2;
            pp[k] -= static_cast<float>(param_.lr) * mh / (std::sqrt(vh) + static_cast<float>(param_.eps));
        }
    }
}

void AdamImpl::clear() {
    for (Tensor& p : params_) {
        if (p.grad()) {
            std::memset(p.grad().data<float>(), 0,
                        static_cast<size_t>(p.numel()) * sizeof(float));
        }
    }
}

} // namespace znn
