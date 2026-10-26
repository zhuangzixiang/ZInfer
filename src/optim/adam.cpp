// Adam: 动量 + 偏差校正(L2 衰减并入梯度); AdamW 见 adamw.cpp
#include "optim.hpp"
#include "znn_internal.hpp"
#include "backend_slots.hpp"
#include "operators/shared.hpp"
#include <cmath>

namespace znn {
void AdamImpl::export_state(std::vector<Tensor>& ts,int64_t& step,OptimParam& p,int& kind) const {
    step=step_count_;p=param_;kind=decoupled_?2:1;
    for(auto& m:moments_) {ts.push_back(m.m);ts.push_back(m.v);}
}
void AdamImpl::import_state(const std::vector<Tensor>& ts,int64_t step,const OptimParam& p,int kind) {
    if(kind!=(decoupled_?2:1)||ts.size()!=params_.size()*2||step<0) throw std::runtime_error("Adam checkpoint mismatch");
    std::vector<Moment> next(params_.size());
    for(size_t i=0;i<params_.size();++i) {
        if(bool(ts[2*i])!=bool(ts[2*i+1]))throw std::runtime_error("Adam incomplete moments");
        if(ts[2*i]) {next[i].m=ts[2*i].to_device(params_[i].device());next[i].v=ts[2*i+1].to_device(params_[i].device());}
    }
    moments_=std::move(next);param_=p;step_count_=step;
}

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
    const float wd = decoupled_ ? 0.f : static_cast<float>(param_.weight_decay);
    const float decay = decoupled_ ? static_cast<float>(param_.weight_decay) : 0.f;
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
                    n, param_.lr, b1, b2, param_.eps, wd, decay,
                    bias1, bias2));
                break;
            case Device::Vulkan:
                ZNN_VULKAN_CALL(vulkan_backend::adam_step_vulkan(
                    p, p.grad(), moments_[i].m, moments_[i].v,
                    n, param_.lr, b1, b2, param_.eps, wd, decay,
                    bias1, bias2));
                break;
            case Device::Metal:
                ZNN_METAL_CALL(metal_backend::adam_step_metal(
                    p, p.grad(), moments_[i].m, moments_[i].v,
                    n, param_.lr, b1, b2, param_.eps, wd, decay,
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
            const float grad = g[k] + wd * pp[k];
            m[k] = b1 * m[k] + (1.f - b1) * grad;
            v[k] = b2 * v[k] + (1.f - b2) * grad * grad;
            const float mh = m[k] / bias1;
            const float vh = v[k] / bias2;
            pp[k] = pp[k] * (1.f - static_cast<float>(param_.lr) * decay)
                  - static_cast<float>(param_.lr) * mh / (std::sqrt(vh) + static_cast<float>(param_.eps));
        }
    }
}

void AdamImpl::clear() {
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
