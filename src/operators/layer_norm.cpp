// layer_norm: 沿指定 dim 归一化 + 仿射(默认 -1 末维); 梯度按标准 LN 反传公式
// (全广播组合, 任意维通用; dim 参数服务 NCHW 通道维归一化 —— ConvNeXt 式用法,
//  免去消费者 transpose+clone 往返)
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backend_slots.hpp"
#include <cmath>

namespace znn {

extern const Operator g_layer_norm_operator;

static void layer_norm_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out, const void* options);


struct LayerNormOptions {
    double eps;
    int64_t dim;                   // -1 = 末维(默认); 否则归一化轴
};

static void layer_norm_cpu(KernelContext& c, Span<const Tensor*> in,
                           Span<Tensor*> out, const void* o) {
    const LayerNormOptions* a = static_cast<const LayerNormOptions*>(o);
    const int64_t dim = a->dim < 0 ? in[0]->dim_count() - 1 : a->dim;
    const Tensor& x = *in[0];
    int64_t inner=1, outer=1;
    for(int64_t d=0;d<x.dim_count();++d) {
        if(d<dim) outer*=x.size(d);
        if(d>dim) inner*=x.size(d);
    }
    if(inner>1 && x.is_contiguous() && in[1]->is_contiguous() && in[2]->is_contiguous()) {
        const int64_t D=x.size(dim), tiles=(inner+127)/128;
        const float* xp=x.data<float>(); const float* wp=in[1]->data<float>();
        const float* bp=in[2]->data<float>(); float* yp=out[0]->data<float>();
        // Adjacent spatial locations share cache lines; reduce channels across
        // a tile instead of striding through the entire tensor per pixel.
        #pragma omp parallel for if(outer*tiles*D>4096)
        for(int64_t t=0;t<outer*tiles;++t) {
            const int64_t base=(t/tiles)*D*inner+(t%tiles)*128;
            const int n=static_cast<int>(std::min<int64_t>(128,inner-(t%tiles)*128));
            double sum[128]={}, sq[128]={}; float mean[128],inv[128];
            for(int64_t d=0;d<D;++d) for(int i=0;i<n;++i) {
                const double v=xp[base+d*inner+i]; sum[i]+=v; sq[i]+=v*v;
            }
            for(int i=0;i<n;++i) {const double m=sum[i]/D;mean[i]=float(m);inv[i]=float(1/std::sqrt(std::max(0.,sq[i]/D-m*m)+a->eps));}
            for(int64_t d=0;d<D;++d) for(int i=0;i<n;++i)
                yp[base+d*inner+i]=(xp[base+d*inner+i]-mean[i])*inv[i]*wp[d]+bp[d];
        }
        return;
    }
    if (dim == in[0]->dim_count() - 1
        && onednn::layer_norm(*in[0], *in[1], *in[2], *out[0], a->eps)) {
        return;
    }
    layer_norm_ref(c, in, out, o);           // 非末维: oneDNN 不适用, 走 ref
}



static void layer_norm_infer(Span<const TensorMeta*> in, const void*,
                             ShapeList& outs) {
    outs.push_back(*in[0]->shape);
}

// 通用: x 看作 [outer, D, inner](contiguous), 归一化组 = 固定 (o,i) 扫 d
static void layer_norm_ref(KernelContext&, Span<const Tensor*> in, Span<Tensor*> out,
                           const void* options) {
    const LayerNormOptions* a = static_cast<const LayerNormOptions*>(options);
    const Tensor& x = *in[0];
    const Tensor& w = *in[1];
    const Tensor& b = *in[2];
    Tensor& y = *out[0];
    const int64_t rank = x.dim_count();
    const int64_t dim = a->dim < 0 ? rank - 1 : a->dim;
    const int64_t D = x.size(dim);
    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < rank; ++d) {
        if (d < dim) outer *= x.size(d);
        else if (d > dim) inner *= x.size(d);
    }
    const float* xp = x.data<float>();
    const float* wp = w.data<float>();
    const float* bp = b.data<float>();
    float* yp = y.data<float>();
    const int64_t blk = D * inner;
    for (int64_t o = 0; o < outer; ++o)
        for (int64_t i = 0; i < inner; ++i) {
            const float* xr = xp + o * blk + i;
            float* yr = yp + o * blk + i;
            double mu = 0, ex2 = 0;
            for (int64_t d = 0; d < D; ++d) {
                const float v = xr[d * inner];
                mu += v; ex2 += double(v) * v;
            }
            mu /= D;
            const double var = ex2 / D - mu * mu;
            const float inv = 1.f / std::sqrt(std::max(var, 0.0) + a->eps);
            const float fmu = float(mu);
            for (int64_t d = 0; d < D; ++d)
                yr[d * inner] = (xr[d * inner] - fmu) * inv * wp[d] + bp[d];
        }
}

static void layer_norm_grad(BackwardContext& c) {
    // 全算子组合(自动走当前后端), 广播语义对任意 dim 通用:
    //   mu/mean(x,dim), inv=1/sqrt(var+eps), x̂=(x-μ)·inv
    //   dγ=Σ g·x̂, dβ=Σ g(除 dim 全约), dx=inv·(gγ - mean(gγ) - x̂·mean(gγ·x̂))
    const LayerNormOptions* a = static_cast<const LayerNormOptions*>(c.options);
    const Tensor& x = *c.saved_inputs[0];
    const Tensor& w = *c.saved_inputs[1];
    const Tensor& g = *c.grad_outputs[0];
    const int64_t rank = x.dim_count();
    const int64_t dim = a->dim < 0 ? rank - 1 : a->dim;
    const Device dev = x.device();

    Tensor mu = Tensor::mean(x, dim, true);                     // [..,1,..]
    Tensor xc = Tensor::sub(x, mu);
    Tensor var = Tensor::mean(Tensor::mul(xc, xc), dim, true);
    Tensor ones = Tensor::full(var.shape(), 1.0, DType::F32, dev);
    Tensor inv = Tensor::div(ones, Tensor::sqrt(Tensor::add(
        var, Tensor::full(var.shape(), a->eps, DType::F32, dev))));
    Tensor xh = Tensor::mul(xc, inv);

    Tensor dgamma = Tensor::mul(g, xh);                // Σ(除 dim 外全约, keepdim
    Tensor dbeta = g;                                  // 保形 → 索引不漂移) → [D]
    for (int64_t d = 0; d < rank; ++d) {
        if (d == dim) continue;
        dgamma = Tensor::sum(dgamma, d, true);
        dbeta = Tensor::sum(dbeta, d, true);
    }
    *c.grad_inputs[1] = dgamma.reshape(Shape{w.size(0)});
    *c.grad_inputs[2] = dbeta.reshape(Shape{w.size(0)});

    Shape ws;
    for (int64_t d = 0; d < rank; ++d) ws.dims[d] = d == dim ? w.size(0) : 1;
    ws.rank = rank;
    Tensor wb = w.reshape(ws);
    Tensor gg = Tensor::mul(g, wb);
    Tensor m2 = Tensor::mean(gg, dim, true);
    Tensor m1 = Tensor::mean(Tensor::mul(gg, xh), dim, true);
    Tensor inner_dx = Tensor::sub(Tensor::sub(gg, m2), Tensor::mul(xh, m1));
    *c.grad_inputs[0] = Tensor::mul(inner_dx, inv);
}

const Operator g_layer_norm_operator = {
    "layer_norm",
    layer_norm_infer,
    layer_norm_grad,
    layer_norm_ref,
    {layer_norm_cpu, ZNN_CUDA_K(layer_norm_cuda), ZNN_METAL_K(layer_norm_metal), ZNN_VULKAN_K(layer_norm_vulkan)},
    sizeof(LayerNormOptions),
};

Tensor layer_norm_apply(const Tensor& x, const Tensor& w, const Tensor& b,
                        double eps, int64_t dim) {
    const LayerNormOptions o{eps, dim};
    const Tensor* ins[3] = {&x, &w, &b};
    return OpDispatcher::instance().run(&g_layer_norm_operator,
                                        Span<const Tensor*>(ins, 3), &o);
}

} // namespace znn
