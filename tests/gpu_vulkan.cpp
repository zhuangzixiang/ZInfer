// gpu_vulkan — Vulkan 后端冒烟与对拍(设备枚举 → 内存 → 内核 → CPU 对拍)
#include "znn.hpp"
#include <cstdio>
#include <cmath>
using namespace znn;

static int fails = 0;
static void check(bool ok, const char* what) {
    if (!ok) { ++fails; std::printf("FAIL: %s\n", what); }
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (!Runtime::device_available(Device::Vulkan)) {
        std::printf("Vulkan not available\n");
        return 1;
    }
    std::printf("Vulkan device OK\n");

    // 1) 内存往返
    float src[8] = {1,2,3,4,5,6,7,8};
    float dst[8] = {0};
    Tensor t = Tensor::from_host(src, {8}).to_device(Device::Vulkan);
    Tensor back = t.to_device(Device::CPU);
    bool rt_ok = true;
    for (int i = 0; i < 8; ++i) rt_ok = rt_ok && back.data<float>()[i] == src[i];
    check(rt_ok, "vulkan memory roundtrip");
    std::printf("rt done\n");

    // 2) 逐元素对拍: add/mul/relu
    {
        Tensor a = Tensor::randn({256, 64}, 1);
        Tensor b = Tensor::randn({256, 64}, 2);
        Tensor c_add = Tensor::add(a, b);
        Tensor g_add = Tensor::add(a.to_device(Device::Vulkan),
                                   b.to_device(Device::Vulkan)).to_device(Device::CPU);
        float m = 0;
        for (int64_t i = 0; i < c_add.numel(); ++i)
            m = fmaxf(m, fabsf(c_add.data<float>()[i] - g_add.data<float>()[i]));
        check(m == 0, "vulkan add parity");
        std::printf("add done\n");

        Tensor c_relu = Activator::relu(a);
        Tensor g_relu = Activator::relu(a.to_device(Device::Vulkan)).to_device(Device::CPU);
        m = 0;
        for (int64_t i = 0; i < a.numel(); ++i)
            m = fmaxf(m, fabsf(c_relu.data<float>()[i] - g_relu.data<float>()[i]));
        check(m == 0, "vulkan relu parity");
    }

    // 3) GEMM 对拍(NN 与 NT)
    {
        Tensor a = Tensor::randn({64, 128}, 3);
        Tensor b = Tensor::randn({128, 96}, 4);
        Tensor c = Tensor::matmul(a, b);
        Tensor g = Tensor::matmul(a.to_device(Device::Vulkan),
                                  b.to_device(Device::Vulkan)).to_device(Device::CPU);
        float m = 0;
        for (int64_t i = 0; i < c.numel(); ++i)
            m = fmaxf(m, fabsf(c.data<float>()[i] - g.data<float>()[i]));
        check(m < 1e-3f, "vulkan matmul parity");
        std::printf("matmul done\n");

        Tensor x = Tensor::randn({32, 64}, 5);
        Tensor w = Tensor::randn({48, 64}, 6);
        Tensor cl = Tensor::linear(x, w);
        Tensor gl = Tensor::linear(x.to_device(Device::Vulkan),
                                   w.to_device(Device::Vulkan)).to_device(Device::CPU);
        m = 0;
        for (int64_t i = 0; i < cl.numel(); ++i)
            m = fmaxf(m, fabsf(cl.data<float>()[i] - gl.data<float>()[i]));
        check(m < 1e-3f, "vulkan linear(NT) parity");
        std::printf("linear done\n");
    }

    // 4) 规约 + softmax
    {
        Tensor x = Tensor::randn({16, 128}, 7);
        Tensor cs = Tensor::sum(x, -1);
        Tensor gs = Tensor::sum(x.to_device(Device::Vulkan), -1).to_device(Device::CPU);
        float m = 0;
        for (int64_t i = 0; i < cs.numel(); ++i)
            m = fmaxf(m, fabsf(cs.data<float>()[i] - gs.data<float>()[i]));
        check(m < 1e-2f, "vulkan sum parity");

        Tensor cm = Tensor::mean(x, 0);
        Tensor gm = Tensor::mean(x.to_device(Device::Vulkan), 0).to_device(Device::CPU);
        m = 0;
        for (int64_t i = 0; i < cm.numel(); ++i)
            m = fmaxf(m, fabsf(cm.data<float>()[i] - gm.data<float>()[i]));
        check(m < 1e-3f, "vulkan mean parity");

        Tensor csm = Activator::softmax(x);
        Tensor gsm = Activator::softmax(x.to_device(Device::Vulkan)).to_device(Device::CPU);
        m = 0;
        for (int64_t i = 0; i < x.numel(); ++i)
            m = fmaxf(m, fabsf(csm.data<float>()[i] - gsm.data<float>()[i]));
        check(m < 1e-4f, "vulkan softmax parity");
    }

    if (fails == 0) std::printf("VULKAN SMOKE TEST PASSED\n");
    else std::printf("%d FAILURES\n", fails);
    return fails ? 1 : 0;
}
