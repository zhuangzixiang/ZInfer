// ztestkit.cpp — 运行器: 收集用例 → 枚举可用设备 → 逐组合执行 → 报告
#include "ztestkit.hpp"

namespace ztestkit {

int g_fails = 0;
int g_checks = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fails;
        std::printf("    FAIL: %s\n", what.c_str());
    }
}

std::vector<Case>& registry() {
    static std::vector<Case> r;
    return r;
}

Registrar::Registrar(const char* n, void (*f)(znn::Device), bool grad) {
    registry().push_back(Case{n, f, grad});
}

float max_diff(const znn::Tensor& a, const znn::Tensor& b) {
    if (a.numel() != b.numel() || a.numel() == 0) {
        return 1e30f;   // 形状不符按无穷大处理
    }
    float m = 0.f;
    for (int64_t i = 0; i < a.numel(); ++i) {
        const float d = std::fabs(a.data<float>()[i] - b.data<float>()[i]);
        m = m < d ? d : m;
    }
    return m;
}

znn::Tensor on(znn::Device d, const znn::Tensor& t) {
    return t.to_device(d);
}

} // namespace ztestkit

int main() {
    using namespace znn;
    const Device devs[] = {Device::CPU, Device::CUDA, Device::Vulkan, Device::Metal};
    for (const ztestkit::Case& c : ztestkit::registry()) {
        for (Device d : devs) {
            if (d != Device::CPU && !Runtime::device_available(d)) {
                continue;   // 运行时探测: 未编入/不可用设备跳过
            }
            std::printf("[ztk] %-28s dev=%-2d ... ", c.name, (int)d);
            const int before = ztestkit::g_fails;
            c.fn(d);
            std::printf("%s\n", ztestkit::g_fails == before ? "ok" : "FAIL");
        }
    }
    if (ztestkit::g_fails == 0) {
        std::printf("ZTESTKIT ALL GREEN (%d checks)\n", ztestkit::g_checks);
        return 0;
    }
    std::printf("ZTESTKIT %d FAILURES (%d checks)\n", ztestkit::g_fails,
                ztestkit::g_checks);
    return 1;
}
