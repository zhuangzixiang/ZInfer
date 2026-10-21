// bench.cpp — 三后端全算子吞吐表(ztestkit 生态, 仅依赖公共 API)
// 运行时探测可用设备, 逐设备逐算子测吞吐(GF/s 或 ms)
#include "znn.hpp"
#include <chrono>
#include <cstdio>
#include <vector>
#include <algorithm>
#if defined(ZNN_CUDA)
#include <cuda_runtime_api.h>
#endif

using namespace znn;

template <class F>
static double ms_of(F&& fn, Device dev, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) {
        Tensor r = fn();
        if (dev == Device::CUDA) {
#if defined(ZNN_CUDA)
            cudaDeviceSynchronize();
#endif
        }
    }
    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        Tensor r = fn();
        if (dev == Device::CUDA) {
#if defined(ZNN_CUDA)
            cudaDeviceSynchronize();
#endif
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

static void bench_device(Device dev, const char* name) {
    if (!Runtime::device_available(dev)) return;
    std::printf("\n===== %s =====\n", name);
    const auto D = [&](const Tensor& t) { return t.to_device(dev); };

    // ---- GEMM ----
    for (auto [M, K, N] : {std::tuple{128, 128, 128}, {512, 512, 512},
                           {1024, 1024, 1024}}) {
        Tensor a = D(Tensor::randn({M, K}, 1));
        Tensor b = D(Tensor::randn({K, N}, 2));
        const double gflop = 2.0 * M * K * N / 1e9;
        const double ms = ms_of([&] { return Tensor::matmul(a, b); }, dev, 3, 10);
        std::printf("  matmul %4lld×%4lld×%4lld  %8.3f ms  %7.1f GF/s\n",
                    (long long)M, (long long)K, (long long)N, ms, gflop / (ms / 1000.0));
    }

    // ---- linear(含 ×w^T) ----
    {
        Tensor x = D(Tensor::randn({256, 512}, 3));
        Tensor w = D(Tensor::randn({256, 512}, 4));
        const double gflop = 2.0 * 256 * 512 * 256 / 1e9;
        const double ms = ms_of([&] { return Tensor::linear(x, w); }, dev, 3, 10);
        std::printf("  linear 256×512×256     %8.3f ms  %7.1f GF/s\n",
                    ms, gflop / (ms / 1000.0));
    }

    // ---- conv2d ----
    {
        Tensor x = D(Tensor::randn({4, 3, 32, 32}, 5));
        Tensor w = D(Tensor::randn({16, 3, 3, 3}, 6));
        const double ms = ms_of([&] { return Tensor::conv2d(x, w); }, dev, 2, 5);
        // ~4*30*30*16*3*3*3*2*2 / 1e9 GF/s(近似)
        std::printf("  conv2d 4×3×32×32→16    %8.3f ms\n", ms);
    }

    // ---- attention ----
    {
        Tensor q = D(Tensor::randn({1, 4, 128, 64}, 7));
        Tensor k = D(Tensor::randn({1, 4, 128, 64}, 8));
        Tensor v = D(Tensor::randn({1, 4, 128, 64}, 9));
        const double ms = ms_of([&] { return Tensor::attention(q, k, v); }, dev, 2, 5);
        std::printf("  attention 1×4×128×64   %8.3f ms\n", ms);
    }

    // ---- 逐元素(带宽) ----
    {
        const int64_t n = 4 << 20;
        Tensor a = D(Tensor::randn({n}, 10));
        Tensor b = D(Tensor::randn({n}, 11));
        const double ms = ms_of([&] { return Tensor::add(a, b); }, dev, 3, 20);
        std::printf("  add %lld elems           %8.3f ms  %6.1f GB/s\n",
                    (long long)n, ms, 3.0 * n * 4 / 1e9 / (ms / 1000.0));
    }

    // ---- softmax ----
    {
        Tensor x = D(Tensor::randn({256, 1024}, 12));
        const double ms = ms_of([&] { return Activator::softmax(x); }, dev, 3, 10);
        std::printf("  softmax 256×1024       %8.3f ms\n", ms);
    }

    // ---- embedding ----
    {
        Tensor w = D(Tensor::randn({10000, 128}, 13));
        Tensor idx = Tensor::from_host(std::vector<int64_t>{1, 5, 3, 99, 42, 7, 1, 5}.data(),
                                       {8}, DType::I64).to_device(dev);
        const double ms = ms_of([&] { return Tensor::embedding(w, idx); }, dev, 3, 20);
        std::printf("  embedding 8×128         %8.3f ms\n", ms);
    }

    // ---- LN(经 Layer 管道; 无独立静态方法) ----
    {
        Tensor x = D(Tensor::randn({64, 512}, 14));
        LayerParam p;
        p.normalized_dim = 512;
        p.eps = 1e-5;
        Layer* ln = Layer::create(LayerType::LayerNorm, p);
        const double ms = ms_of([&] { return ln->forward(x); }, dev, 3, 10);
        std::printf("  layer_norm 64×512       %8.3f ms\n", ms);
        Layer::destroy(ln);
    }
}

int main() {
    std::printf("=== znn 三后端吞吐表(ztestkit bench) ===\n");
    bench_device(Device::CPU, "CPU");
    bench_device(Device::CUDA, "CUDA");
    bench_device(Device::Vulkan, "Vulkan");
    bench_device(Device::Metal, "Metal");
    std::printf("\n=== 完成 ===\n");
    return 0;
}
