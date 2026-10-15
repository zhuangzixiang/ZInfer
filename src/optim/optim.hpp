// ============================================================================
// optim.hpp — 优化器实现类声明 (一个优化器一文件; 仅库内使用)
// 优化器没有后端维度: 参数更新是逐元素数学, 经算子派发选择设备。
// ============================================================================
#pragma once
#include "znn.hpp"
#include <vector>

namespace znn {

class SgdImpl final : public Optimizer {
public:
    SgdImpl(std::vector<Tensor*> params, const OptimParam& param);
    void step() override;
    void clear() override;
private:
    std::vector<Tensor*> params_;
    OptimParam param_;
};

class AdamImpl : public Optimizer {
public:
    AdamImpl(std::vector<Tensor*> params, const OptimParam& param);
    void step() override;
    void clear() override;
protected:
    // AdamW 复用动量逻辑, 仅 weight_decay 语义不同(解耦衰减)
    struct Moment {
        Tensor m;      // 一阶动量
        Tensor v;      // 二阶动量
    };
    std::vector<Tensor*> params_;
    std::vector<Moment>  moments_;
    OptimParam param_;
};

class AdamWImpl final : public AdamImpl {
public:
    using AdamImpl::AdamImpl;
    void step() override;    // 与 Adam 仅解耦衰减语义不同
};

} // namespace znn
