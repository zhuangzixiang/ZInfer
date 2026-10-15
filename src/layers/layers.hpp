// ============================================================================
// layers.hpp — 内置层实现类声明 (一层一文件, 实现在本目录; 仅库内使用)
//
// 层没有后端维度: forward 与设备无关, 设备选择发生在算子派发。
// ReLU/GELU/Sigmoid/Tanh 共用 ActivationLayer, Max/Avg 池化共用 Pool2dLayer。
// ============================================================================
#pragma once
#include "znn.hpp"
#include <random>
#include <vector>

namespace znn {

class LinearLayer final : public Layer {
public:
    LinearLayer(int64_t in, int64_t out, bool bias);
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor> params() const override;
    void train(bool) override {}
private:
    Tensor w_, b_;
    bool has_bias_;
};

class Conv2dLayer final : public Layer {
public:
    Conv2dLayer(int64_t in_ch, int64_t out_ch, Shape kernel,
                const Conv2dOptions& options, bool bias);
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor> params() const override;
    void train(bool) override {}
private:
    Tensor w_, b_;
    Conv2dOptions options_;
    bool has_bias_;
};

class BatchNorm2dLayer final : public Layer {
public:
    BatchNorm2dLayer(int64_t channels, double eps, double momentum);
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor> params() const override;
    void train(bool mode) override { training_ = mode; }
private:
    Tensor w_, b_;          // 仿射参数 γ/β
    Tensor mean_, var_;     // running 统计(非训练参数)
    double eps_, momentum_;
    bool training_ = true;
};

class LayerNormLayer final : public Layer {
public:
    LayerNormLayer(int64_t normalized_dim, double eps);
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor> params() const override;
    void train(bool) override {}
private:
    Tensor w_, b_;
    double eps_;
};

class EmbeddingLayer final : public Layer {
public:
    EmbeddingLayer(int64_t num_embeddings, int64_t embed_dim);
    Tensor forward(const Tensor& indices) override;
    std::vector<Tensor> params() const override;
    void train(bool) override {}
private:
    Tensor w_;
};

// ReLU/GELU/Sigmoid/Tanh 共用: 无参数, 仅按类型分发到 Activator
class ActivationLayer final : public Layer {
public:
    explicit ActivationLayer(LayerType type);
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor> params() const override { return {}; }
    void train(bool) override {}
private:
    LayerType type_;
};

class FlattenLayer final : public Layer {
public:
    explicit FlattenLayer(int64_t start_dim);
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor> params() const override { return {}; }
    void train(bool) override {}
private:
    int64_t start_dim_;
};

// MaxPool2d/AvgPool2d 共用
class Pool2dLayer final : public Layer {
public:
    Pool2dLayer(LayerType type, Shape kernel, const Pool2dOptions& options);
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor> params() const override { return {}; }
    void train(bool) override {}
private:
    LayerType type_;
    Shape kernel_;
    Pool2dOptions options_;
};

class DropoutLayer final : public Layer {
public:
    DropoutLayer(double rate, unsigned seed);
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor> params() const override { return {}; }
    void train(bool mode) override { training_ = mode; }
private:
    double rate_;
    unsigned seed_;
    std::mt19937 rng_;                                      // 前向掩码随机源
    bool training_ = true;
};

} // namespace znn
