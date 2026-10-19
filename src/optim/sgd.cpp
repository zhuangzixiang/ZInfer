// SGD: p -= lr * (grad + weight_decay*p) [+ 动量]
#include "optim.hpp"
#include "znn_internal.hpp"
#include "backend_slots.hpp"

namespace znn {

SgdImpl::SgdImpl(std::vector<Tensor> params, const OptimParam& param)
    : param_(param) {
    params_ = std::move(params);   // 基类成员(GradScaler 共享)
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
        const int64_t n = p.numel();
        if (p.device() != Device::CPU) {                 // GPU: 原生内核更新
            if (param_.momentum > 0 && !velocity_[i]) {
                velocity_[i] = Tensor::zeros(p.shape(), DType::F32, p.device());
            }
            Tensor& velv = velocity_[i];   // momentum>0 时已按参数设备分配
            switch (p.device()) {
            case Device::CUDA:
                ZNN_CUDA_CALL(ZNN_CUDA_K(sgd_step_cuda)(
                    cuda_ptr(p), cuda_ptr(p.grad()),
                    param_.momentum > 0 ? cuda_ptr(velv) : cuda_ptr(p), n,
                    param_.lr, param_.momentum, param_.weight_decay));
                break;
            case Device::Vulkan:
                ZNN_VULKAN_CALL(vulkan_backend::sgd_step_vulkan(
                    p.data<Tensor::VulkanDataPtr>(),
                    p.grad().data<Tensor::VulkanDataPtr>(),
                    param_.momentum > 0 ? velv.data<Tensor::VulkanDataPtr>()
                                        : p.data<Tensor::VulkanDataPtr>(), n,
                    param_.lr, param_.momentum, param_.weight_decay));
                break;
            case Device::Metal:
                ZNN_METAL_CALL(metal_backend::sgd_step_metal(
                    p.data<Tensor::MetalDataPtr>(),
                    p.grad().data<Tensor::MetalDataPtr>(),
                    param_.momentum > 0 ? velv.data<Tensor::MetalDataPtr>()
                                        : p.data<Tensor::MetalDataPtr>(), n,
                    param_.lr, param_.momentum, param_.weight_decay));
                break;
            default: break;
            }
            continue;
        }
        const float* g = p.grad().data<float>();
        float* pp = p.data<float>();
        if (param_.momentum > 0) {
            if (!velocity_[i]) {
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
            device_memset0(p.grad().device(), p.grad().data<float>(),
                           static_cast<size_t>(p.numel()) * sizeof(float));
        }
    }
}

} // namespace znn
