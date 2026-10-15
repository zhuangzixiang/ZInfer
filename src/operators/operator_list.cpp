// ============================================================================
// operator_list.cpp — 算子注册表数据(≈ ffmpeg codec_list.c)
//
// 注册"数据"集中于此; 注册"机制"(查找/派发/兜底)在 OpRegistry(znn_internal)。
// 新增算子三步: operators/<name>.cpp 定义 OpDef → 此处 extern 与 ZNN_OP
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

ZNN_OP_LIST_BEGIN
    ZNN_OP(g_arange_def)
    ZNN_OP(g_randn_def)
    ZNN_OP(g_add_def)
    ZNN_OP(g_sub_def)
    ZNN_OP(g_mul_def)
    ZNN_OP(g_div_def)
    ZNN_OP(g_neg_def)
    ZNN_OP(g_exp_def)
    ZNN_OP(g_log_def)
    ZNN_OP(g_sqrt_def)
    ZNN_OP(g_relu_def)
    ZNN_OP(g_gelu_def)
    ZNN_OP(g_sigmoid_def)
    ZNN_OP(g_tanh_def)
    ZNN_OP(g_silu_def)
    ZNN_OP(g_softmax_def)
    ZNN_OP(g_sum_def)
    ZNN_OP(g_mean_def)
    ZNN_OP(g_matmul_def)
    ZNN_OP(g_linear_def)
    ZNN_OP(g_cat_def)
    ZNN_OP(g_stack_def)
    ZNN_OP(g_conv2d_def)
    ZNN_OP(g_max_pool2d_def)
    ZNN_OP(g_avg_pool2d_def)
    ZNN_OP(g_embedding_def)
    ZNN_OP(g_attention_def)
    ZNN_OP(g_cross_entropy_def)
    ZNN_OP(g_mse_def)
ZNN_OP_LIST_END

} // namespace znn
