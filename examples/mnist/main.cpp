// MNIST 里程碑 1 验收例程(骨架): 训练循环已就位, 数据加载与算子待实现。
// 验收标准: 5 epoch 内测试集准确率 > 97%(见 DESIGN.md §10)
#include "znn.hpp"
#include <cstdio>
#include <stdexcept>
#include <string>
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

// ---- IDX 二进制格式解析 ----
// 文件: <dir>-images-idx3-ubyte 与 <dir>-labels-idx1-ubyte
// 布局: 大端 magic(4B) + 维度(每个 4B) + 数据; 图像为 ubyte 像素
struct IdxFile {
    std::vector<uint8_t> bytes;
    size_t pos = 0;

    uint32_t read_u32() {
        const uint32_t v = (uint32_t(bytes[pos]) << 24) | (uint32_t(bytes[pos + 1]) << 16)
                         | (uint32_t(bytes[pos + 2]) << 8) | uint32_t(bytes[pos + 3]);
        pos += 4;
        return v;
    }
};

IdxFile read_file(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    IdxFile file;
    uint8_t chunk[65536];
    size_t got = 0;
    while ((got = std::fread(chunk, 1, sizeof(chunk), f)) > 0) {
        file.bytes.insert(file.bytes.end(), chunk, chunk + got);
    }
    std::fclose(f);
    return file;
}

constexpr int64_t kBatchSize = 64;

void load_mnist(Mnist& out, const char* prefix) {
    IdxFile images = read_file(std::string(prefix) + "-images-idx3-ubyte");
    IdxFile labels = read_file(std::string(prefix) + "-labels-idx1-ubyte");
    if (images.read_u32() != 2051 || labels.read_u32() != 2049) {
        throw std::runtime_error("bad IDX magic");
    }
    const int64_t n = images.read_u32();
    const int64_t rows = images.read_u32();
    const int64_t cols = images.read_u32();
    labels.read_u32();                                   // 标签数(重复)
    if (rows * cols != kImagePixels) {
        throw std::runtime_error("unexpected image size");
    }
    out.batches.clear();
    for (int64_t b = 0; b < n; b += kBatchSize) {
        const int64_t m = (b + kBatchSize <= n) ? kBatchSize : n - b;
        Batch batch;
        batch.imgs.reserve(static_cast<size_t>(m) * kImagePixels);
        batch.labels.reserve(static_cast<size_t>(m));
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t p = 0; p < kImagePixels; ++p) {
                batch.imgs.push_back(images.bytes[images.pos++] / 255.f);
            }
            batch.labels.push_back(labels.bytes[labels.pos++]);
        }
        out.batches.push_back(std::move(batch));
    }
    std::printf("loaded %lld samples from %s (batches=%lld)",
                (long long)n, prefix, (long long)out.batches.size());
}

// 验证集准确率
float evaluate(const std::vector<Layer*>& layers, const Mnist& data) {
    Runtime::set_grad_enabled(false);
    int64_t correct = 0;
    int64_t total = 0;
    for (const Batch& batch : data.batches) {
        Tensor x = Tensor::from_host(batch.imgs.data(),
                                     {static_cast<int64_t>(batch.labels.size()), kImagePixels});
        Tensor h = x;
        for (Layer* l : layers) h = l->forward(h);
        const float* logits = h.data<float>();
        for (size_t i = 0; i < batch.labels.size(); ++i) {
            int best = 0;
            for (int c = 1; c < 10; ++c) {
                if (logits[i * 10 + c] > logits[i * 10 + best]) best = c;
            }
            correct += best == batch.labels[i];
            ++total;
        }
    }
    Runtime::set_grad_enabled(true);
    return total ? static_cast<float>(correct) / static_cast<float>(total) : 0.f;
}

} // namespace

int main() {
    try {
        Mnist train, test;
        load_mnist(train, "data/mnist/train");
        load_mnist(test,  "data/mnist/t10k");

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
