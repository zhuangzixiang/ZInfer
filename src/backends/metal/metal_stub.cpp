// ============================================================================
// metal_stub.cpp — 非 Metal 构建的同名实现(明确报错, 派发器回落 CPU)
// ============================================================================
#include "operator.hpp"
#include "backends/metal/metal_backend.hpp"
#include <stdexcept>

namespace znn {
namespace metal_backend {

namespace {
void unavailable() {
    throw std::runtime_error("znn: Metal backend not built (ZNN_METAL=OFF)");
}
}

bool available() { return false; }
void* alloc(size_t) { unavailable(); return nullptr; }
void free(void*) {}
void memset0(void*, size_t) { unavailable(); }
void copy_host_to_device(void*, const void*, size_t) { unavailable(); }
void copy_device_to_host(void*, const void*, size_t) { unavailable(); }
void copy_device_to_device(void*, const void*, size_t) { unavailable(); }

#define MT_STUB(name)                                                        \
    void name(KernelContext&, Span<const Tensor*>, Span<Tensor*>,            \
              const void*) {                                                 \
        unavailable();                                                       \
    }
MT_STUB(add_metal) MT_STUB(sub_metal) MT_STUB(mul_metal) MT_STUB(div_metal)
MT_STUB(neg_metal) MT_STUB(exp_metal) MT_STUB(log_metal) MT_STUB(sqrt_metal)
MT_STUB(relu_metal) MT_STUB(gelu_metal) MT_STUB(sigmoid_metal)
MT_STUB(tanh_metal) MT_STUB(silu_metal)
MT_STUB(matmul_metal) MT_STUB(linear_metal)
MT_STUB(softmax_metal) MT_STUB(sum_metal) MT_STUB(mean_metal)
#undef MT_STUB

} // namespace metal_backend
} // namespace znn
