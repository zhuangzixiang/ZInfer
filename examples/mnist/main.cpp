// MNIST 里程碑 1 验收例程(骨架): 训练循环已就位, 数据加载与算子待实现。
// 验收标准: 5 epoch 内测试集准确率 > 97%(见 DESIGN.md §10)
#include "znn.hpp"
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace znn;

namespace {

constexpr int64_t kImagePixels = 784;

struct Batch {
    std::vector<float>   imgs;     // [B, 784], 归一化到 [0,1]
    std::vector<int64_t> labels;   // [B]
};

struct Mnist {
    std::vector<Batch> batches;
};

// IDX 二进制格式解析(train-images-idx3-ubyte 等 4 个文件)
void load_mnist(Mnist&, const char*) {
    throw std::runtime_error("TODO: IDX loader (实现阶段)");
}

// 验证集准确率
float evaluate(const std::vector<Layer*>&, const Mnist&) {
    throw std::runtime_error("TODO: evaluate (实现阶段)");
}

} // namespace

int main() {
    try {
        Mnist train, test;
        load_mnist(train, "data/mnist/train");
        load_mnist(test,  "data/mnist/test");

        std::vector<Layer*> layers = {
            Layer::create(LayerType::Linear, LayerParam{784, 256}),
            Layer::create(LayerType::ReLU,   LayerParam{}),
            Layer::create(LayerType::Linear, LayerParam{256, 10}),
        };
        Optimizer* opt = Optimizer::create(OptimType::Adam, layers, OptimParam{1e-3});

        for (int epoch = 0; epoch < 5; ++epoch) {
            for (const Batch& b : train.batches) {
                Tensor x = Tensor::from_host(b.imgs.data(),
                                             {static_cast<int64_t>(b.labels.size()), kImagePixels});
                Tensor y = Tensor::from_host(b.labels.data(),
                                             {static_cast<int64_t>(b.labels.size())}, DType::I64);

                Tensor h = x;
                for (Layer* l : layers) h = l->forward(h);
                Tensor loss = Loss::cross_entropy(h, y);

                opt->clear();
                loss.backward();
                opt->step();

                std::printf("epoch %d loss %.4f\n", epoch, loss.item<float>());
            }
            std::printf("epoch %d acc %.4f\n", epoch, evaluate(layers, test));
        }

        for (Layer* l : layers) Layer::destroy(l);
        Optimizer::destroy(opt);
        return 0;
    } catch (const std::exception& e) {          // 骨架阶段: 未实现以明确消息退出
        std::fprintf(stderr, "mnist: %s\n", e.what());
        return 1;
    }
}
