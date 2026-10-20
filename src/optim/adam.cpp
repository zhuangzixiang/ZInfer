// Adam: 动量 + 偏差校正(L2 衰减并入梯度); AdamW 见 adamw.cpp
#include "optim.hpp"
#include "znn_internal.hpp"
#include "backend_slots.hpp"
#include <cmath>

namespace znn {

AdamImpl::AdamImpl(std::vector<Tensor> params, const OptimParam& param)
    : param_(param) {
    params_ = std::move(params);   // 基类成员(GradScaler 共享)
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
        const int64_t n = p.numel();
        if (!moments_[i].m) {                         // 惰性分配(跟随参数设备)
            moments_[i].m = Tensor::zeros(p.shape(), DType::F32, p.device());
            moments_[i].v = Tensor::zeros(p.shape(), DType::F32, p.device());
        }
        if (p.device() != Device::CPU) {              // GPU: 原生内核更新
            switch (p.device()) {
            case Device::CUDA:
                ZNN_CUDA_CALL(ZNN_CUDA_K(adam_step_cuda)(
                    cuda_ptr(p), cuda_ptr(p.grad()),
                    cuda_ptr(moments_[i].m), cuda_ptr(moments_[i].v),
                    n, param_.lr, b1, b2, param_.eps, param_.weight_decay,
                    bias1, bias2));
                break;
            case Device::Vulkan:
                ZNN_VULKAN_CALL(vulkan_backend::adam_step_vulkan(
                    p, p.grad(), moments_[i].m, moments_[i].v,
                    n, param_.lr, b1, b2, param_.eps, param_.weight_decay,
                    bias1, bias2));
                break;
            case Device::Metal:
                ZNN_METAL_CALL(metal_backend::adam_step_metal(
                    p, p.grad(), moments_[i].m, moments_[i].v,
                    n, param_.lr, b1, b2, param_.eps, param_.weight_decay,
                    bias1, bias2));
                break;
            default: break;
            }
            continue;
        }
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
            device_memset0(p.grad().device(), p.grad().data<float>(),
                           static_cast<size_t>(p.numel()) * sizeof(float));
        }
    }
}

} // namespace znn
