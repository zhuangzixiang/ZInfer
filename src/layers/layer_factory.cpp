// Layer 工厂: 按 LayerType 分发到各层实现并校验必填项
// (自定义层直接继承 Layer, 不经此工厂)
#include "layers/layers.hpp"

#include <stdexcept>

namespace znn {

static void require(bool ok, const char* what) {
    if (!ok) {
        throw std::runtime_error(std::string("znn: create layer: ") + what);
    }
}

Layer* Layer::create(LayerType type, const LayerParam& p) {
    switch (type) {
        case LayerType::Linear:
            require(p.in > 0 && p.out > 0, "linear needs in/out");
            return new LinearLayer(p.in, p.out, p.bias);
        case LayerType::Conv2d:
            require(p.in > 0 && p.out > 0 && (p.kernel.rank == 2 && p.kernel[0] > 0 && p.kernel[1] > 0), "conv2d needs in/out/kernel");
            return new Conv2dLayer(p.in, p.out, p.kernel, p.conv, p.bias);
        case LayerType::BatchNorm2d:
            require(p.channels > 0, "batchnorm2d needs channels");
            return new BatchNorm2dLayer(p.channels, p.eps, p.momentum);
        case LayerType::LayerNorm:
            require(p.normalized_dim > 0, "layernorm needs normalized_dim");
            return new LayerNormLayer(p.normalized_dim, p.eps);
        case LayerType::Embedding:
            require(p.num_embeddings > 0 && p.embed_dim > 0, "embedding needs dims");
            return new EmbeddingLayer(p.num_embeddings, p.embed_dim);
        case LayerType::Dropout:
            require(p.dropout_rate >= 0 && p.dropout_rate < 1, "dropout rate in [0,1)");
            return new DropoutLayer(p.dropout_rate, p.seed);
        case LayerType::ReLU:
        case LayerType::GELU:
        case LayerType::Sigmoid:
        case LayerType::Tanh:
            return new ActivationLayer(type);
        case LayerType::Flatten:
            return new FlattenLayer(p.flatten_start);
        case LayerType::MaxPool2d:
        case LayerType::AvgPool2d:
            require((p.kernel.rank == 2 && p.kernel[0] > 0 && p.kernel[1] > 0), "pool needs kernel");
            return new Pool2dLayer(type, p.kernel, p.pool);
        default:
            throw std::runtime_error("znn: unknown layer type");
    }
}

void Layer::destroy(Layer* layer) {
    delete layer;
}

} // namespace znn
