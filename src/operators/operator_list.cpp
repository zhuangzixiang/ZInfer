// ============================================================================
// operator_list.cpp — 算子注册数据(≈ ffmpeg codec_list.c)
//
// 注册"数据"集中于此; 派发器(执行机制)在 op_dispatcher。
// 新增算子三步: operators/<name>.cpp 定义 Operator → 此处 extern 与数组
// 各加一行 → CMake 源清单加一行。
// ============================================================================
#include "operator.hpp"

namespace znn {

extern const Operator g_arange_operator;
extern const Operator g_randn_operator;
extern const Operator g_add_operator, g_sub_operator, g_mul_operator, g_div_operator;
extern const Operator g_neg_operator, g_exp_operator, g_log_operator, g_sqrt_operator;
extern const Operator g_relu_operator, g_gelu_operator, g_sigmoid_operator, g_tanh_operator;
extern const Operator g_silu_operator, g_softmax_operator;
extern const Operator g_sum_operator, g_mean_operator;
extern const Operator g_matmul_operator, g_linear_operator;
extern const Operator g_cat_operator, g_stack_operator;
extern const Operator g_conv2d_operator, g_max_pool2d_operator, g_avg_pool2d_operator;
extern const Operator g_embedding_operator, g_attention_operator;
extern const Operator g_cross_entropy_operator, g_mse_operator;
extern const Operator g_transpose_operator;
extern const Operator g_reshape_operator;
extern const Operator g_conv2d_operator, g_max_pool2d_operator, g_avg_pool2d_operator;
extern const Operator g_cat_operator, g_stack_operator, g_embedding_operator;
extern const Operator g_layer_norm_operator, g_batch_norm2d_operator, g_attention_operator;

// OpDispatcher 首次使用时读取的算子清单, 以 nullptr 结尾
const Operator* g_operator_list[] = {
    &g_arange_operator,
    &g_randn_operator,
    &g_add_operator,
    &g_sub_operator,
    &g_mul_operator,
    &g_div_operator,
    &g_neg_operator,
    &g_exp_operator,
    &g_log_operator,
    &g_sqrt_operator,
    &g_relu_operator,
    &g_gelu_operator,
    &g_sigmoid_operator,
    &g_tanh_operator,
    &g_silu_operator,
    &g_softmax_operator,
    &g_sum_operator,
    &g_mean_operator,
    &g_matmul_operator,
    &g_linear_operator,
    &g_cat_operator,
    &g_stack_operator,
    &g_conv2d_operator,
    &g_max_pool2d_operator,
    &g_avg_pool2d_operator,
    &g_embedding_operator,
    &g_attention_operator,
    &g_cross_entropy_operator,
    &g_mse_operator,
    &g_transpose_operator,
    &g_reshape_operator,
    &g_conv2d_operator,
    &g_max_pool2d_operator,
    &g_avg_pool2d_operator,
    &g_cat_operator,
    &g_stack_operator,
    &g_embedding_operator,
    &g_layer_norm_operator,
    &g_batch_norm2d_operator,
    &g_attention_operator,
    nullptr,
};

} // namespace znn
