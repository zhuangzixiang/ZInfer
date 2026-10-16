// GPU 基准测试: 张量驻留 GPU, 纯计算性能(排除 PCIe 开销)
#include "znn.hpp"
#include <chrono>
#include <cstdio>

using namespace znn;

static double now_sec() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main() {
    if (!Runtime::device_available(Device::CUDA)) {
        std::printf("CUDA not available, abort\n");
        return 1;
    }
    std::printf("=== GPU 基准(数据驻留 GPU, 纯计算) ===\n");

    // matmul: 创建在 CPU → 一次搬到 GPU → 循环计算(不搬回)
    struct { int64_t M, K, N; int iters; } sizes[] = {
        {128, 128, 128, 100},
        {256, 256, 256, 50},
        {512, 512, 512, 30},
        {1024, 1024, 1024, 20},
        {2048, 2048, 2048, 10},
        {4096, 4096, 4096, 5},
    };
    for (auto& s : sizes) {
        Tensor a = Tensor::randn({s.M, s.K}, 1).to_device(Device::CUDA);
        Tensor b = Tensor::randn({s.K, s.N}, 2).to_device(Device::CUDA);
        Tensor warm = Tensor::matmul(a, b);       // 预热(内核编译 + 缓存)
        (void)warm;
        const double t0 = now_sec();
        for (int i = 0; i < s.iters; ++i) {
            Tensor c = Tensor::matmul(a, b);
            (void)c;
        }
        // 不拷回, 纯 GPU 计算
        const double dt = (now_sec() - t0) / s.iters;
        const double flops = 2.0 * s.M * s.K * s.N;
        std::printf("matmul %4lld^3 : %8.3f ms  %8.1f GFLOP/s\n",
                    (long long)s.M, dt * 13, flops / dt / 1e9);
    }

    // conv2d: 数据驻留 GPU
    {
        Tensor x = Tensor::randn({8, 16, 32, 32}, 3).to_device(Device::CUDA);
        Tensor w = Tensor::randn({32, 16, 3, 3}, 4).to_device(Device::CUDA);
        Conv2dOptions o;
        o.pad = {1, 1};
        Tensor warm = Tensor::conv2d(x, w, o);
        (void)warm;
        const double t0 = now_sec();
        for (int i = 0; i < 20; ++i) {
            Tensor y = Tensor::conv2d(x, w, o);
            (void)y;
        }
        const double dt = (now_sec() - t0) / 20;
        const double flops = 2.0 * 8 * 32 * 16 * 3 * 3 * 32 * 32;
        std::printf("conv2d 8x16x32x32  : %8.3f ms  %8.1f GFLOP/s\n", dt * 1e3, flops / dt / 1e9);
    }

    // MLP 推理(数据驻留 GPU): 纯 GPU 计算循环
    {
        Tensor x = Tensor::randn({64, 784}, 5).to_device(Device::CUDA);
        Tensor w1 = Tensor::randn({256, 784}, 6).to_device(Device::CUDA);
        Tensor w2 = Tensor::randn({10, 256}, 7).to_device(Device::CUDA);
        // 预热
        { Tensor h = Tensor::linear(x, w1, nullptr); Tensor y = Tensor::linear(h, w2, nullptr); }
        const double t0 = now_sec();
        for (int i = 0; i < 100; ++i) {
            Tensor h = Tensor::linear(x, w1, nullptr);
            Tensor y = Tensor::linear(h, w2, nullptr);
            (void)y;
        }
        const double dt = (now_sec() - t0) / 100;
        std::printf("MLP 784-256-10 infer (GPU resident) : %8.3f ms\n", dt * 1e3);
    }

    std::printf("=== GPU 基准完成 ===\n");
    return 0;
}
