// ============================================================================
// operator_list.cpp — 算子注册数据(≈ ffmpeg codec_list.c)
//
// 注册"数据"集中于此; 派发器(执行机制)在 op_dispatcher。
// 新增算子三步: operators/<name>.cpp 定义 OpDef → 此处 extern 与数组
// 各加一行 → CMake 源清单加一行。
// ============================================================================
#include "op_def.hpp"

namespace znn {

extern const OpDef g_arange_def;
extern const OpDef g_randn_def;
extern const OpDef g_add_def, g_sub_def, g_mul_def, g_div_def;
extern const OpDef g_neg_def, g_exp_def, g_log_def, g_sqrt_def;
extern const OpDef g_relu_def, g_gelu_def, g_sigmoid_def, g_tanh_def;
extern const OpDef g_silu_def, g_softmax_def;
extern const OpDef g_sum_def, g_mean_def;
extern const OpDef g_matmul_def, g_linear_def;
extern const OpDef g_cat_def, g_stack_def;
extern const OpDef g_conv2d_def, g_max_pool2d_def, g_avg_pool2d_def;
extern const OpDef g_embedding_def, g_attention_def;
extern const OpDef g_cross_entropy_def, g_mse_def;

// OpDispatcher 首次使用时读取的算子清单, 以 nullptr 结尾
const OpDef* g_znn_op_list[] = {
    &g_arange_def,
    &g_randn_def,
    &g_add_def,
    &g_sub_def,
    &g_mul_def,
    &g_div_def,
    &g_neg_def,
    &g_exp_def,
    &g_log_def,
    &g_sqrt_def,
    &g_relu_def,
    &g_gelu_def,
    &g_sigmoid_def,
    &g_tanh_def,
    &g_silu_def,
    &g_softmax_def,
    &g_sum_def,
    &g_mean_def,
    &g_matmul_def,
    &g_linear_def,
    &g_cat_def,
    &g_stack_def,
    &g_conv2d_def,
    &g_max_pool2d_def,
    &g_avg_pool2d_def,
    &g_embedding_def,
    &g_attention_def,
    &g_cross_entropy_def,
    &g_mse_def,
    nullptr,
};

} // namespace znn
