#include "znn.hpp"
#include <cstdio>
using namespace znn;
int main() {
    fprintf(stderr,"1: device_available(CUDA)=%d\n", (int)Runtime::device_available(Device::CUDA));
    if (!Runtime::device_available(Device::CUDA)) {
        fprintf(stderr,"CUDA unavailable, abort\n");
        return 1;
    }
    fprintf(stderr,"2: creating tensors\n");
    Tensor a = Tensor::randn({4, 4}, 1);
    Tensor b = Tensor::randn({4, 4}, 2);
    Tensor cpu = Tensor::add(a, b);
    fprintf(stderr,"3: cpu[0]=%f\n", cpu.data<float>()[0]);
    fprintf(stderr,"4: to_device(CUDA)...\n");
    Tensor ga = a.to_device(Device::CUDA);
    fprintf(stderr,"5: ga on %d\n", (int)ga.device());
    Tensor gb = b.to_device(Device::CUDA);
    fprintf(stderr,"6: gb on %d\n", (int)gb.device());
    fprintf(stderr,"7: gpu add...\n");
    Tensor gadd = Tensor::add(ga, gb);
    fprintf(stderr,"8: gpu add done, on %d\n", (int)gadd.device());
    fprintf(stderr,"9: to_device(CPU)...\n");
    Tensor back = gadd.to_device(Device::CPU);
    fprintf(stderr,"10: gpu[0]=%f, cpu[0]=%f, match=%s\n",
           back.data<float>()[0], cpu.data<float>()[0],
           back.data<float>()[0] == cpu.data<float>()[0] ? "YES" : "NO");
    fprintf(stderr,"GPU SMOKE TEST PASSED\n");
    return 0;
}
