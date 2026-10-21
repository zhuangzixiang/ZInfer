// hello_dll -- minimal external consumer: public header + znn.dll only
// Verifies DLL loads and runs 3-backend compute from a standalone EXE
#include "znn.hpp"
#include <cstdio>

int main() {
    std::printf("=== znn.dll smoke (external consumer) ===\n");

    // CPU
    Tensor a = Tensor::randn({64, 64}, 1);
    Tensor b = Tensor::randn({64, 64}, 2);
    Tensor y = Tensor::matmul(a, b);
    std::printf("CPU matmul 64x64: %f\n", y.sum(0).sum(0).item<float>());

    // CUDA (runtime probe)
    if (Runtime::device_available(Device::CUDA)) {
        Tensor ga = a.to_device(Device::CUDA);
        Tensor gb = b.to_device(Device::CUDA);
        Tensor gy = Tensor::matmul(ga, gb).to_device(Device::CPU);
        std::printf("CUDA matmul 64x64: %f\n", gy.sum(0).sum(0).item<float>());
    } else {
        std::printf("CUDA: not available\n");
    }

    // Vulkan
    if (Runtime::device_available(Device::Vulkan)) {
        Tensor va = a.to_device(Device::Vulkan);
        Tensor vb = b.to_device(Device::Vulkan);
        Tensor vy = Tensor::matmul(va, vb).to_device(Device::CPU);
        std::printf("Vulkan matmul 64x64: %f\n", vy.sum(0).sum(0).item<float>());
    } else {
        std::printf("Vulkan: not available\n");
    }

    std::printf("=== PASS ===\n");
    return 0;
}
