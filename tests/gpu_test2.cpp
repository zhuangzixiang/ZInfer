#include "znn.hpp"
#include <cstdio>
using namespace znn;
int main() {
    fprintf(stderr, "A: zeros\n");
    Tensor x = Tensor::zeros({4, 4});
    fprintf(stderr, "B: randn\n");
    Tensor a = Tensor::randn({64, 64}, 1);
    Tensor b = Tensor::randn({64, 64}, 2);
    fprintf(stderr, "C: cpu matmul\n");
    Tensor cpu = Tensor::matmul(a, b);
    fprintf(stderr, "D: cpu done=%f\n", cpu.data<float>()[0]);
    fprintf(stderr, "E: to_gpu\n");
    Tensor ga = a.to_device(Device::CUDA);
    Tensor gb = b.to_device(Device::CUDA);
    fprintf(stderr, "F: gpu matmul\n");
    Tensor g = Tensor::matmul(ga, gb);
    fprintf(stderr, "G: back\n");
    Tensor back = g.to_device(Device::CPU);
    fprintf(stderr, "H: result=%f\n", back.data<float>()[0]);
    fprintf(stderr, "PASS\n");
    return 0;
}
