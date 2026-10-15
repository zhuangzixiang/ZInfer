// Embedding: 按索引取行
#include "layers/layers.hpp"
#include "operator.hpp"
#include "znn_internal.hpp"
#include "znn_internal.hpp"

namespace znn {

EmbeddingLayer::EmbeddingLayer(int64_t, int64_t) {
    // TODO: w_ 按 [num, dim] 分配
}

Tensor EmbeddingLayer::forward(const Tensor& indices) {
    // TODO: Tensor::embedding(w_, indices)
    ZNN_TODO();
}

std::vector<Tensor> EmbeddingLayer::params() const {
    return {w_};
}

} // namespace znn
