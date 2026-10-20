// randn: 标准正态随机张量(seed=0 用全局引擎)
#include "shared.hpp"
#include "op_dispatcher.hpp"
#include "backend_slots.hpp"
#include <random>

namespace znn {

extern const Operator g_randn_operator;

static void randn_ref(KernelContext&, Span<const Tensor*>, Span<Tensor*> out, const void* options) {
    const unsigned seed = *static_cast<const unsigned*>(options);
    static std::mt19937 global_engine(42);
    std::mt19937 local(7);
    std::mt19937& engine = seed != 0 ? local : global_engine;
    if (seed != 0) {
        local.seed(seed);
    }
    std::normal_distribution<float> dist(0.f, 1.f);
    float* p = out[0]->data<float>();
    for (int64_t i = 0; i < out[0]->numel(); ++i) p[i] = dist(engine);
}

const Operator g_randn_operator = {
    "randn",
    nullptr,   // 输出形状即前端分配目标, 无需推导
    nullptr,   // 无需求导
    randn_ref,
    {},        // kernels[CPU/CUDA/Metal/Vulkan]
};

Tensor Tensor::randn(Shape shape, unsigned seed, DType dtype, Device device) {
    Tensor t = Tensor::empty(shape, dtype, device);
    Tensor* outs[1] = {&t};
    KernelContext ctx{device, nullptr};
    if (device == Device::CUDA) {
        ZNN_CUDA_CALL(ZNN_CUDA_K(randn_cuda)(ctx, Span<const Tensor*>(nullptr, 0),
                                 Span<Tensor*>(outs, 1), &seed));
    } else if (device == Device::Vulkan) {
        ZNN_VULKAN_CALL(vulkan_backend::randn_vulkan(ctx,
                Span<const Tensor*>(nullptr, 0), Span<Tensor*>(outs, 1), &seed));
    } else if (device == Device::Metal) {
        ZNN_METAL_CALL(metal_backend::randn_metal(ctx,
                Span<const Tensor*>(nullptr, 0), Span<Tensor*>(outs, 1), &seed));
    } else {
        randn_ref(ctx, Span<const Tensor*>(nullptr, 0), Span<Tensor*>(outs, 1), &seed);
    }
    return t;
}

} // namespace znn
