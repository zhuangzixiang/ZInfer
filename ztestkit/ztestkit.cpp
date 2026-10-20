// ztestkit.cpp — 运行器: 收集用例 → 枚举可用设备 → 逐组合执行 → 报告
#include "ztestkit.hpp"

// ---- 崩溃自打印栈(dbghelp; 崩溃时直接给出符号化调用栈) ----
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
namespace ztestkit {
LONG crash_stack(EXCEPTION_POINTERS* ep) {
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_LOAD_LINES);
    SymInitialize(GetCurrentProcess(), nullptr, TRUE);
    char sympath[MAX_PATH]; GetModuleFileNameA(nullptr, sympath, MAX_PATH);
    *strrchr(sympath, 0x5C) = 0; SymSetSearchPath(GetCurrentProcess(), sympath);
    void* frames[32];
    const USHORT n = CaptureStackBackTrace(0, 32, frames, nullptr);
    std::printf("==== CRASH code=0x%08X addr=%p stack: ====",
                (unsigned)ep->ExceptionRecord->ExceptionCode,
                ep->ExceptionRecord->ExceptionAddress);
    char buf[sizeof(SYMBOL_INFO) + 128] = {};
    for (USHORT i = 0; i < n; ++i) {
        SYMBOL_INFO* si = reinterpret_cast<SYMBOL_INFO*>(buf);
        si->SizeOfStruct = sizeof(SYMBOL_INFO);
        si->MaxNameLen = 127;
        DWORD64 off = 0;
        if (SymFromAddr(GetCurrentProcess(), (DWORD64)frames[i], &off, si)) {
            std::printf("  #%02d %s+0x%llx", i, si->Name, (unsigned long long)off);
        } else {
            std::printf("  #%02d %p", i, frames[i]);
        }
    }
    std::fflush(stdout);
    return EXCEPTION_CONTINUE_SEARCH;
}
void install_crash_handler() { SetUnhandledExceptionFilter(crash_stack); }
} // namespace ztestkit


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
    setvbuf(stdout, NULL, _IONBF, 0);   // 崩溃时进度不丢
    ztestkit::install_crash_handler();  // 崩溃自打印栈
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
