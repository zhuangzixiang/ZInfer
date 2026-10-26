// hello_dll -- minimal external consumer: public header + znn.dll only
// Verifies DLL loads and runs 3-backend compute from a standalone EXE
#include "znn.hpp"
#include <cmath>
#include <cstdio>

using namespace znn;

static void check(bool ok, const char* what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) std::exit(1);
}

int main() {
    std::printf("=== znn.dll smoke (external consumer) ===\n");

    // CPU
    Tensor a = Tensor::randn({64, 64}, 1);
    Tensor b = Tensor::randn({64, 64}, 2);
    Tensor y = Tensor::matmul(a, b);
    std::printf("CPU matmul 64x64: %f\n", Tensor::sum(Tensor::sum(y, 0), 0).item<float>());

    // CUDA (runtime probe)
    if (Runtime::device_available(Device::CUDA)) {
        Tensor ga = a.to_device(Device::CUDA);
        Tensor gb = b.to_device(Device::CUDA);
        Tensor gy = Tensor::matmul(ga, gb).to_device(Device::CPU);
        std::printf("CUDA matmul 64x64: %f\n", Tensor::sum(Tensor::sum(gy, 0), 0).item<float>());
    } else {
        std::printf("CUDA: not available\n");
    }

    // Vulkan
    if (Runtime::device_available(Device::Vulkan)) {
        Tensor va = a.to_device(Device::Vulkan);
        Tensor vb = b.to_device(Device::Vulkan);
        Tensor vy = Tensor::matmul(va, vb).to_device(Device::CPU);
        std::printf("Vulkan matmul 64x64: %f\n", Tensor::sum(Tensor::sum(vy, 0), 0).item<float>());
    } else {
        std::printf("Vulkan: not available\n");
    }

    // upsample 梯度探针
    {
        try {
            Tensor x = Tensor::ones({2, 3, 4, 6});
            x.requires_grad(true);
            Tensor y = Tensor::upsample2x_nearest(x);
            Tensor l = y; while (l.dim_count()) l = Tensor::sum(l, 0); l.backward();
            std::printf("[upgrad] ok grad0=%g numel=%lld\n",
                        x.grad().data<float>()[0], (long long)x.grad().numel());
        } catch (const std::exception& e) {
            std::printf("[upgrad] FAIL: %s\n", e.what());
        }
    }

    // DLL 消费者回归: 此前缺失的导出(Shape 方法 / data<T> / item<T>)
    {
        Conv2dOptions o;                            // 触发 Shape(initializer_list)
        o.pad = {1, 1};
        check(o.pad == Shape{1, 1}, "Shape ctor/operator==");
        Tensor t = Tensor::ones({4, 4});
        check(t.data<float>()[0] == 1.f, "Tensor::data<float>");
        check(Tensor::full({}, 2.5).item<float>() == 2.5f, "Tensor::item<float>");
    }

    std::printf("=== PASS ===\n");
    return 0;
}
