// 基准测试: matmul / conv2d / MLP 训练步 / 推理步(当前内核的性能快照)
#include "znn.hpp"
#include <chrono>
#include <cstdio>
#include <vector>

using namespace znn;

static double now_sec() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void bench_matmul(int64_t M, int64_t K, int64_t N, int iters) {
    Tensor a = Tensor::randn({M, K}, 1);
    Tensor b = Tensor::randn({K, N}, 2);
    Tensor warm = Tensor::matmul(a, b);                     // 预热
    (void)warm;
    const double t0 = now_sec();
    for (int i = 0; i < iters; ++i) {
        Tensor c = Tensor::matmul(a, b);
        (void)c;
    }
    const double dt = (now_sec() - t0) / iters;
    const double flops = 2.0 * M * K * N;
    std::printf("matmul %lldx%lld @ %lldx%lld : %8.3f ms  %6.2f GFLOP/s\n",
                (long long)M, (long long)K, (long long)K, (long long)N, dt * 1e3, flops / dt / 1e9);
}

static void bench_conv2d(int iters) {
    Tensor x = Tensor::randn({8, 16, 32, 32}, 3);           // NCHW
    Tensor w = Tensor::randn({32, 16, 3, 3}, 4);
    Conv2dOptions o;
    o.pad = {1, 1};
    Tensor warm = Tensor::conv2d(x, w, o);
    (void)warm;
    const double t0 = now_sec();
    for (int i = 0; i < iters; ++i) {
        Tensor y = Tensor::conv2d(x, w, o);
        (void)y;
    }
    const double dt = (now_sec() - t0) / iters;
    const double flops = 2.0 * 8 * 32 * 16 * 3 * 3 * 32 * 32;
    std::printf("conv2d 8x16x32x32 x 32x3x3(pad1): %8.3f ms  %6.2f GFLOP/s\n",
                dt * 1e3, flops / dt / 1e9);
}

static void bench_attention(int iters) {
    Tensor q = Tensor::randn({1, 4, 128, 64}, 5);
    Tensor k = Tensor::randn({1, 4, 128, 64}, 6);
    Tensor v = Tensor::randn({1, 4, 128, 64}, 7);
    Tensor warm = Tensor::attention(q, k, v);
    (void)warm;
    const double t0 = now_sec();
    for (int i = 0; i < iters; ++i) {
        Tensor y = Tensor::attention(q, k, v);
        (void)y;
    }
    const double dt = (now_sec() - t0) / iters;
    std::printf("attention 1x4x128x64 : %8.3f ms\n", dt * 1e3);
}

static void bench_mlp_train(int iters) {
    std::vector<Layer*> net = {
        Layer::create(LayerType::Linear, LayerParam{784, 256}),
        Layer::create(LayerType::ReLU, LayerParam{}),
        Layer::create(LayerType::Linear, LayerParam{256, 10}),
    };
    Optimizer* opt = Optimizer::create(OptimType::Adam, net, OptimParam{1e-3});
    std::vector<float> x(64 * 784, 0.5f);
    std::vector<int64_t> y(64, 3);
    const double t0 = now_sec();
    for (int i = 0; i < iters; ++i) {
        Tensor xb = Tensor::from_host(x.data(), {64, 784});
        Tensor yb = Tensor::from_host(y.data(), {64}, DType::I64);
        Tensor h = xb;
        for (Layer* l : net) h = l->forward(h);
        Tensor loss = Loss::cross_entropy(h, yb);
        opt->clear();
        loss.backward();
        opt->step();
    }
    const double dt = (now_sec() - t0) / iters;
    std::printf("MLP 784-256-10 train step (B=64) : %8.3f ms  (=> %5.1f s/epoch, 938 steps)\n",
                dt * 1e3, dt * 938);
    Runtime::set_grad_enabled(false);
    const double t1 = now_sec();
    for (int i = 0; i < iters; ++i) {
        Tensor xb = Tensor::from_host(x.data(), {64, 784});
        Tensor h = xb;
        for (Layer* l : net) h = l->forward(h);
    }
    std::printf("MLP 784-256-10 infer step (B=64) : %8.3f ms\n",
                (now_sec() - t1) / iters * 1e3);
    Runtime::set_grad_enabled(true);
    for (Layer* l : net) Layer::destroy(l);
    Optimizer::destroy(opt);
}

int main() {
    std::printf("=== znn benchmark (CPU, naive reference kernels) ===\n");
    bench_matmul(128, 128, 128, 50);
    bench_matmul(512, 512, 512, 10);
    bench_matmul(1024, 1024, 1024, 3);
    bench_conv2d(5);
    bench_attention(5);
    bench_mlp_train(20);
    return 0;
}
