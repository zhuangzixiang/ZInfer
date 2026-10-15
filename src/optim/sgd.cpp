// SGD: p -= lr * (grad + weight_decay*p) [+ 动量]
#include "optim.hpp"
#include "znn_internal.hpp"

namespace znn {

SgdImpl::SgdImpl(std::vector<Tensor> params, const OptimParam& param)
    : params_(std::move(params)), param_(param) {
    if (param_.momentum > 0) {
        velocity_.resize(params_.size());
        for (Tensor& v : velocity_) {
            v = Tensor::zeros(Tensor::zeros({1}).shape());    // 占位, 首步按参数形状分配
        }
    }
}

void SgdImpl::step() {
    for (size_t i = 0; i < params_.size(); ++i) {
        Tensor& p = params_[i];
        if (!p.grad()) {
            continue;
        }
        const float* g = p.grad().data<float>();
        float* pp = p.data<float>();
        const int64_t n = p.numel();
        if (param_.momentum > 0) {
            if (!velocity_[i]) {             // 惰性分配速度缓冲
                velocity_[i] = Tensor::zeros(p.shape());
            }
            float* v = velocity_[i].data<float>();
            for (int64_t k = 0; k < n; ++k) {
                v[k] = param_.momentum * v[k] + g[k] + param_.weight_decay * pp[k];
                pp[k] -= param_.lr * v[k];
            }
        } else {
            for (int64_t k = 0; k < n; ++k) {
                pp[k] -= param_.lr * (g[k] + param_.weight_decay * pp[k]);
            }
        }
    }
}

void SgdImpl::clear() {
    for (Tensor& p : params_) {
        if (p.grad()) {
            std::memset(p.grad().data<float>(), 0,
                        static_cast<size_t>(p.numel()) * sizeof(float));
        }
    }
}

} // namespace znn
