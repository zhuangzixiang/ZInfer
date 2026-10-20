// ============================================================================
// ztestkit.hpp — 独立测试工具包(唯一依赖: 被测库公共头 znn.hpp)
//
// 剥离性: 本目录不 include 任何 src/ 内部头, 整目录拷走即可测试任何
// 提供同接口的库。设备枚举运行时探测 —— 同一套用例自动跑全部可用后端。
//
// 用例注册(ZTK_CASE): 函数体收 Device 参数, CPU 恒跑(兼作基准),
// 其余设备按可用性自动展开。对拍基准 = 同用例在 CPU 上的新鲜图。
// ============================================================================
#pragma once
#include "znn.hpp"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace ztestkit {

// ---- 断言与计数(runner 汇总) ----
extern int g_fails;
extern int g_checks;
void check(bool ok, const std::string& what);

// ---- 用例注册表 ----
struct Case {
    const char* name;
    void (*fn)(znn::Device);   // 函数体内在 ztk_dev 上执行并内部 check
    bool needs_grad;           // 标记(供报告分类; 不影响执行)
};
std::vector<Case>& registry();
struct Registrar {
    Registrar(const char* n, void (*f)(znn::Device), bool grad);
};

#define ZTK_CASE(name, grad)                                              \
    static void ztk_fn_##name(znn::Device ztk_dev);                       \
    static ztestkit::Registrar ztk_reg_##name(#name, ztk_fn_##name, grad);\
    static void ztk_fn_##name(znn::Device ztk_dev)

// ---- 对拍助手(两侧须已位于 CPU) ----
float max_diff(const znn::Tensor& a, const znn::Tensor& b);
// 同值种子在目标设备生成(先 CPU 生成再搬运, 保证逐位一致)
znn::Tensor on(znn::Device d, const znn::Tensor& t);
// 简写: 拷回 CPU
inline znn::Tensor back(const znn::Tensor& t) { return t.to_device(znn::Device::CPU); }

} // namespace ztestkit
