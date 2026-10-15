// Embedding: 按索引取行
#include "layers/layers.hpp"
#include "znn_internal.hpp"
#include <cmath>

namespace znn {

EmbeddingLayer::EmbeddingLayer(int64_t num_embeddings, int64_t embed_dim)
    : w_(Tensor::randn({num_embeddings, embed_dim})) {
    const float scale = 1.f / std::sqrt(static_cast<float>(embed_dim));
    float* wp = w_.data<float>();
    for (int64_t i = 0; i < w_.numel(); ++i) wp[i] *= scale;
    w_.requires_grad(true);
}

Tensor EmbeddingLayer::forward(const Tensor& indices) {
    return Tensor::embedding(w_, indices);
}

std::vector<Tensor> EmbeddingLayer::params() const {
    return {w_};
}

} // namespace znn
