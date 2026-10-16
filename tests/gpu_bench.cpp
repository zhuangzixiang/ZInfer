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
int main() {
    if (!Runtime::device_available(Device::CUDA)) { fprintf(stderr, "no cuda\n"); return 1; }
    fprintf(stderr, "=== FP32 vs TF32 (RTX 4070 Ti SUPER) ===\n");
    const int64_t S[] = {512, 1024, 2048, 4096};
    for (int64_t M : S) {
        int n = (M >= 4096) ? 5 : (M >= 2048) ? 10 : (M >= 1024) ? 20 : 30;
        bench("FP32", Precision::FP32, M, n);
        bench("TF32", Precision::TF32, M, n);
    }
    Runtime::set_precision(Precision::AUTO);
    fprintf(stderr, "=== done ===\n");
    return 0;
}
