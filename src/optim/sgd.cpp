// SGD: p -= lr * (grad + weight_decay*p) [+ 动量]
#include "optim.hpp"
#include "znn_internal.hpp"
#include "backend_slots.hpp"
#include "operators/shared.hpp"

namespace znn {
void SgdImpl::export_state(std::vector<Tensor>& ts,int64_t& step,OptimParam& p,int& kind) const {
    step=0;p=param_;kind=0;ts=velocity_;ts.resize(params_.size());
}
void SgdImpl::import_state(const std::vector<Tensor>& ts,int64_t step,const OptimParam& p,int kind) {
    if(kind!=0||step!=0||ts.size()!=params_.size())throw std::runtime_error("SGD checkpoint mismatch");
    std::vector<Tensor> next(ts.size());for(size_t i=0;i<ts.size();++i)if(ts[i])next[i]=ts[i].to_device(params_[i].device());
    velocity_=std::move(next);param_=p;
}

SgdImpl::SgdImpl(std::vector<Tensor> params, const OptimParam& param)
    : param_(param) {
    params_ = std::move(params);   // 基类成员(GradScaler 共享)
    velocity_.resize(params_.size());
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
                    p, p.grad(), param_.momentum > 0 ? velv : p, n,
                    param_.lr, param_.momentum, param_.weight_decay));
                break;
            case Device::Metal:
                ZNN_METAL_CALL(metal_backend::sgd_step_metal(
                    p, p.grad(), param_.momentum > 0 ? velv : p, n,
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
            const Tensor& gr = p.grad();
            void* gh = gr.device() == Device::CPU
                         ? static_cast<void*>(gr.data<float>())   // CPU: 宿主缓冲
                         : gpu_handle(gr);                        // GPU: 裸句柄
            device_memset0(gr.device(), gh,
                           static_cast<size_t>(p.numel()) * sizeof(float));
        }
    }
}

} // namespace znn
