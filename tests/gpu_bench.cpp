#include "znn.hpp"
#include <chrono>
#include <cstdio>
using namespace znn;
static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static void bench(const char* label, Precision mode, int64_t M, int iters) {
    Runtime::set_precision(mode);
    Tensor a = Tensor::randn({M, M}, 1).to_device(Device::CUDA);
    Tensor b = Tensor::randn({M, M}, 2).to_device(Device::CUDA);
    Tensor w = Tensor::matmul(a, b);
    w.to_device(Device::CPU);                    // 预热+同步
    double t0 = now();
    Tensor last;
    for (int i = 0; i < iters; ++i) last = Tensor::matmul(a, b);
    last.to_device(Device::CPU);                 // 同步(真实 GPU 完成时间)
    double dt = (now() - t0) / iters;
    double tf = 2.0 * M * M * M / dt / 1e12;
    fprintf(stderr, "%s %4lld^3: %8.3f ms  %6.2f TF/s\n", label, (long long)M, dt*1e3, tf);
}

// 通用算子基准: 创建一次输入, 循环调用, 首尾同步
template <class Make, class Call>
static void bench_op(const char* label, double flops, int iters,
                     Make make, Call call) {
    auto in = make();
    Tensor y = call(in);                          // 预热
    y.to_device(Device::CPU);
    double t0 = now();
    Tensor last;
    for (int i = 0; i < iters; ++i) last = call(in);
    last.to_device(Device::CPU);
    double dt = (now() - t0) / iters;
    fprintf(stderr, "%-26s: %8.3f ms  %6.2f TF/s\n", label, dt*1e3, flops/dt/1e12);
}

int main() {
    if (!Runtime::device_available(Device::CUDA)) { fprintf(stderr, "no cuda\n"); return 1; }
    Runtime::set_precision(Precision::TF32);
    fprintf(stderr, "=== FP32 vs TF32 matmul (RTX 4070 Ti SUPER) ===\n");
    const int64_t S[] = {512, 1024, 2048, 4096};
    for (int64_t M : S) {
        int n = (M >= 4096) ? 5 : (M >= 2048) ? 10 : (M >= 1024) ? 20 : 30;
        bench("FP32", Precision::FP32, M, n);
        bench("TF32", Precision::TF32, M, n);
        bench("FP16", Precision::FP16, M, n);
    }
    fprintf(stderr, "=== small matmul (synchronized, allocation included) ===\n");
    const int64_t small[] = {16, 32, 64, 128, 256};
    for (int64_t M : small) {
        bench("FP32", Precision::FP32, M, 50);
        bench("TF32", Precision::TF32, M, 50);
        bench("FP16", Precision::FP16, M, 50);
    }

    fprintf(stderr, "=== GEMM-class ops (TF32, data-resident) ===\n");
    // linear: y[N,OUT] = x[N,IN] × w[OUT,IN]^T (+bias)
    {
        const int N = 4096, IN = 2048, OUT = 2048;
        Tensor x = Tensor::randn({N, IN}, 1).to_device(Device::CUDA);
        Tensor w = Tensor::randn({OUT, IN}, 2).to_device(Device::CUDA);
        Tensor b = Tensor::randn({OUT}, 3).to_device(Device::CUDA);
        bench_op("linear 4096x2048x2048+bias",
                 2.0 * N * IN * OUT, 30,
                 [&] { return std::make_tuple(x, w, b); },
                 [](auto& t) { return Tensor::linear(std::get<0>(t), std::get<1>(t), &std::get<2>(t)); });
    }
    // conv2d: N8 C16→32 3x3, 32x32 输入(im2col+GEMM 的典型负载)
    {
        const int Nn = 8, C = 16, O = 32, HW = 32, K = 3;
        Tensor x = Tensor::randn({Nn, C, HW, HW}, 1).to_device(Device::CUDA);
        Tensor w = Tensor::randn({O, C, K, K}, 2).to_device(Device::CUDA);
        bench_op("conv2d 8x16x32x32 -> 32x3x3",
                 2.0 * Nn * O * C * K * K * HW * HW, 30,
                 [&] { return std::make_tuple(x, w); },
                 [](auto& t) { return Tensor::conv2d(std::get<0>(t), std::get<1>(t)); });
    }
    // attention: B4 H8 L256 D64(QK^T + softmax + PV)
    {
        const int B = 4, H = 8, L = 256, D = 64;
        Tensor q = Tensor::randn({B, H, L, D}, 1).to_device(Device::CUDA);
        Tensor k = Tensor::randn({B, H, L, D}, 2).to_device(Device::CUDA);
        Tensor v = Tensor::randn({B, H, L, D}, 3).to_device(Device::CUDA);
        bench_op("attention 4x8x256x64",
                 2.0 * 2 * B * H * L * L * D, 20,
                 [&] { return std::make_tuple(q, k, v); },
                 [](auto& t) { return Tensor::attention(std::get<0>(t), std::get<1>(t), std::get<2>(t)); });
    }
    Runtime::set_precision(Precision::AUTO);
    fprintf(stderr, "=== done ===\n");
    return 0;
}
