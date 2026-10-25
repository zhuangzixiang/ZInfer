// ============================================================================
// znn_defines.hpp — ZInfer 结构定义 (v0.6 讨论稿)
//
// 与 znn.hpp(接口定义)配对。本文件只放: 枚举 / 轻量结构 / 构造参数体。
// 不放任何类方法。两文件合起来构成对外全部 API。
//
// 命名法: *Param  = create() 的构造参数体
//         *Options= 运算选项(卷积步长/补零等)
// ============================================================================
#pragma once
#include <cstdint>
#include <initializer_list>

// ---- DLL 导出/导入宏(仅 Windows 需要显式标注; 静态库/非 Windows = 空) ----
#ifdef ZNN_SHARED
  #ifdef _WIN32
    #ifdef ZNN_BUILDING
      #define ZNN_API __declspec(dllexport)
    #else
      #define ZNN_API __declspec(dllimport)
    #endif
  #else
    #define ZNN_API __attribute__((visibility("default")))
  #endif
#else
  #define ZNN_API
#endif

namespace znn {

// ---------------------------------------------------------------- 基础枚举

enum class Device : uint8_t {
    CPU = 0,
    CUDA = 1,
    Metal = 2,
    Vulkan = 3,
    kDeviceCount,                // 哨兵 = 后端数量; 新后端在此追加
};

enum class DType : uint8_t {
    F32 = 0,                     // v1 实现 F32(权重/激活)
    F16,
    BF16,                        // 半精度随 GPU 里程碑启用
    I32,
    I64,                         // I64: 标签/索引
    U8,
    kDTypeCount,
};

// ---------------------------------------------------------------- 形状

// 张量形状 = 简单的维度记录: dims[0..rank) 有效, 最多 8 维(超出抛异常)。
// n()/c()/h()/w() 是固定位置的命名访问(按 NCHW 惯例), 仅在图像类张量上
// 有语义; 一般张量(如权重 {out,in})请用索引访问(支持负索引, -1=最末维)。
struct Shape {
    Shape() = default;
    ZNN_API Shape(std::initializer_list<int64_t> dims);   // {B,C,H,W} / {out,in} / {} 标量

    int64_t dims[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t rank    = 0;                          // 维数, 0 = 标量

    ZNN_API int64_t& operator[](int64_t i);
    ZNN_API int64_t  operator[](int64_t i) const;

    int64_t n() const { return dims[0]; }         // 图像类张量便捷访问
    int64_t c() const { return dims[1]; }
    int64_t h() const { return dims[2]; }
    int64_t w() const { return dims[3]; }

    ZNN_API bool operator==(const Shape& other) const;
};

// ---------------------------------------------------------------- 运算选项

struct Conv2dOptions {
    Shape pad      = {0, 0};     // 输入四周补零: {上下, 左右} 行/列数
    Shape stride   = {1, 1};     // 滑动窗口每次移动: {纵向, 横向}
    Shape dilation = {1, 1};     // 空洞卷积: 核内元素间隔
    int64_t groups = 1;          // 分组卷积组数(1=普通卷积)
};

struct Pool2dOptions {
    Shape pad    = {0, 0};       // 输入四周补零: {上下, 左右} 行/列数
    Shape stride = {1, 1};       // 滑动窗口每次移动: {纵向, 横向}
};

// ---------------------------------------------------------------- Layer 构造

enum class LayerType : uint8_t {
    Linear = 0,
    Conv2d,
    BatchNorm2d,
    LayerNorm,
    Embedding,
    Dropout,
    ReLU,
    GELU,
    Sigmoid,
    Tanh,
    Flatten,
    MaxPool2d,
    AvgPool2d,
    kLayerTypeCount,
};

// 单一构造参数体(≈ AVCodecContext 风格): 按 LayerType 填必填项,
// create() 按类型校验, 缺失/非法字段抛异常。
struct LayerParam {
    // Linear:      in, out, bias
    // Conv2d:      in, out, kernel, conv, bias
    // BatchNorm2d: channels, eps, momentum
    // LayerNorm:   normalized_dim, eps
    // Embedding:   num_embeddings, embed_dim
    // Dropout:     dropout_rate
    // Flatten:     flatten_start
    // MaxPool2d / AvgPool2d: kernel, pool
    // ReLU/GELU/Sigmoid/Tanh: 无必填项
    int64_t       in  = -1;         // 输入特征数 / 输入通道数
    int64_t       out = -1;         // 输出特征数 / 输出通道数
    int64_t       channels = -1;    // BatchNorm2d 通道数
    int64_t       normalized_dim = -1;
    int64_t       norm_dim = -1;      // LayerNorm 归一化轴(-1=末维; NCHW 通道维填 1)
    int64_t       num_embeddings = -1;
    int64_t       embed_dim = -1;
    Shape         kernel = {0, 0};  // 卷积核 / 池化窗口: {kh, kw}
    Conv2dOptions conv  = {};       // 卷积选项
    Pool2dOptions pool  = {};       // 池化选项
    double        dropout_rate = 0.5;
    double        eps      = 1e-5;
    double        momentum = 0.1;
    int64_t       flatten_start = 1;
    bool          bias = true;
    unsigned      seed  = 0;        // Dropout/初始化随机种子
};

// ---------------------------------------------------------------- Optimizer 构造

enum class OptimType : uint8_t {
    Sgd = 0,
    Adam,
    AdamW,
    kOptimTypeCount,
};

struct OptimParam {
    // Sgd:   lr, momentum, weight_decay
    // Adam:  lr, beta1, beta2, eps, weight_decay(L2)
    // AdamW: lr, beta1, beta2, eps, weight_decay(解耦)
    double lr           = 1e-3;   // 学习率
    double beta1        = 0.9;
    double beta2        = 0.999;
    double eps          = 1e-8;
    double momentum     = 0.0;
    double weight_decay = 0.0;
};

} // namespace znn
