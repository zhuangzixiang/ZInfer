// Optimizer 工厂: layers 重载自动收集各层参数
#include "optim.hpp"
#include "znn_internal.hpp"
#include <stdexcept>
#include <fstream>
#include <cmath>

namespace znn {
void Optimizer::export_state(std::vector<Tensor>&,int64_t&,OptimParam&,int&) const {throw std::runtime_error("Optimizer does not support checkpoints");}
void Optimizer::import_state(const std::vector<Tensor>&,int64_t,const OptimParam&,int) {throw std::runtime_error("Optimizer does not support checkpoints");}
void Optimizer::save_state(const std::string& path) const {
    std::vector<Tensor> ts;int64_t step;OptimParam p;int kind;export_state(ts,step,p,kind);
    std::ofstream f(path,std::ios::binary|std::ios::trunc);
    auto put=[&](const auto& v){f.write(reinterpret_cast<const char*>(&v),sizeof(v));};
    uint32_t magic=0x31504f5a;put(magic);put(kind);put(step);
    for(double v:{p.lr,p.beta1,p.beta2,p.eps,p.momentum,p.weight_decay})put(v);
    uint64_t count=ts.size();put(count);
    for(auto& t:ts) {int64_t rank=t?t.dim_count():-1;put(rank);if(rank<0)continue;
        for(int64_t d=0;d<rank;++d)put(t.size(d));auto cpu=t.to_device(Device::CPU);
        f.write(reinterpret_cast<const char*>(cpu.data<float>()),t.numel()*sizeof(float));}
    f.flush();if(!f)throw std::runtime_error("Cannot save optimizer state");
}
void Optimizer::load_state(const std::string& path) {
    std::ifstream f(path,std::ios::binary);auto get=[&](auto& v){if(!f.read(reinterpret_cast<char*>(&v),sizeof(v)))throw std::runtime_error("Truncated optimizer state");};
    uint32_t magic;int kind;int64_t step;OptimParam p;uint64_t count;
    get(magic);get(kind);get(step);get(p.lr);get(p.beta1);get(p.beta2);get(p.eps);get(p.momentum);get(p.weight_decay);get(count);
    if(magic!=0x31504f5a||kind<0||kind>2||step<0||count!=params_.size()*(kind?2:1))throw std::runtime_error("Invalid optimizer checkpoint");
    for(double v:{p.lr,p.beta1,p.beta2,p.eps,p.momentum,p.weight_decay})if(!std::isfinite(v))throw std::runtime_error("Nonfinite optimizer configuration");
    if(p.lr<0||p.eps<=0||p.beta1<0||p.beta1>=1||p.beta2<0||p.beta2>=1||p.momentum<0)throw std::runtime_error("Invalid optimizer configuration");
    std::vector<Tensor> ts(count);
    for(size_t i=0;i<count;++i) {int64_t rank;get(rank);if(rank==-1)continue;const auto& expected=params_[i/(kind?2:1)];
        if(rank!=expected.dim_count())throw std::runtime_error("Optimizer rank mismatch");
        for(int64_t d=0;d<rank;++d){int64_t n;get(n);if(n!=expected.size(d))throw std::runtime_error("Optimizer shape mismatch");}
        ts[i]=Tensor::empty(expected.shape(),DType::F32,Device::CPU);
        if(!f.read(reinterpret_cast<char*>(ts[i].data<float>()),ts[i].numel()*sizeof(float)))throw std::runtime_error("Truncated optimizer tensor");
        for(int64_t j=0;j<ts[i].numel();++j)if(!std::isfinite(ts[i].data<float>()[j]))throw std::runtime_error("Nonfinite optimizer tensor");}
    if(f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Trailing optimizer data");
    import_state(ts,step,p,kind);
}

Optimizer* Optimizer::create(OptimType type, const std::vector<Layer*>& layers,
                             const OptimParam& param) {
    std::vector<Tensor> params;
    for (Layer* layer : layers) {
        for (Tensor& t : layer->params()) {
            params.push_back(t);
        }
    }
    return create(type, std::move(params), param);
}

Optimizer* Optimizer::create(OptimType type, std::vector<Tensor> params,
                             const OptimParam& param) {
    switch (type) {
        case OptimType::Sgd:   return new SgdImpl(std::move(params), param);
        case OptimType::Adam:  return new AdamImpl(std::move(params), param);
        case OptimType::AdamW: return new AdamWImpl(std::move(params), param);
        default: throw std::runtime_error("znn: unknown optimizer type");
    }
}

void Optimizer::destroy(Optimizer* opt) {
    delete opt;
}

} // namespace znn
