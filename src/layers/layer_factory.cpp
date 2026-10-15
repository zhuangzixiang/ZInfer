// Layer 工厂: 按 LayerType 分发到各层实现并校验必填项
// (自定义层直接继承 Layer, 不经此工厂)
#include "layers/layers.hpp"
#include "znn_ext.hpp"

namespace znn {

Layer* Layer::create(LayerType type, const LayerParam& param) {
    // TODO: switch(type) 分发 + 必填项校验(缺失抛异常, 不返回 nullptr)
    ZNN_TODO();
}

void Layer::destroy(Layer* layer) {
    delete layer;
}

} // namespace znn
