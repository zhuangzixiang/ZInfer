/*
 * @Description: 计算图模块
 * @Author: ZhuangZixiang
 * @Date: 2025-04-28 23:28:22
 * @LastEditTime: 2025-05-03 23:56:12
 * @LastEditors: ZhuangZixiang
 */
#ifndef __COMPUTAYIONAL_GRAPH_H__
#define __COMPUTAYIONAL_GRAPH_H__

#include <vector>
#include <memory>
#include "tensor.h"
#include "ComputationalGraph_defines.h"


namespace zinfer
{

struct OpKernalConfig
{
    uint32_t stride[2] = { 1, 1 };  /*** stride[0]:行stride  stride[1]:列stride ***/

    uint32_t padding[2] = {0, 0};  /*** padding[0]:上下各padding的行数  padding[1]:左右各padding的列数 ***/ 

    uint32_t kernal_size = 0;
    uint32_t* kernal_ptr = nullptr;

    uint32_t dilation_size = 0;
    uint32_t* dilation_ptr = nullptr;


    uint32_t channels_in = 0;
    uint32_t channels_out = 0;
    uint32_t groups = 1;

    bool bias_use = false;

    PaddingMode padding_mode = PaddingMode::PMODE_VALID;
};

template <typename T>
struct OpKernalCalcuFactor
{
    //Tensor<T> factor_weights;  /*** 卷积运算的系数：权重 ***/
    //Tensor<T> factor_bias;  /*** 卷积运算的系数：偏移 ***/

      /*** 卷积层中的每个滤波器都有权重和偏置 ***/
    std::vector<std::shared_ptr<Tensor<T>>> factor_weights_;  /*** 权重负责提取特征 ***/
    std::vector<std::shared_ptr<Tensor<T>>> factor_bias_;  /*** 偏置负责调整输出的激活值 ***/
};

template <typename T>
class OperationKernal;
/**
 * @description: 操作数: 算子的输入/输出
 */
template <typename T>
class Operand
{
public:
    virtual ~Operand()
    {}

private:
    Tensor<T> data;  /*** 操作数中实际的数据 ***/
    OperationKernal<T>* owner_kernal;  /*** 产生本操作数的算子 ***/
    OperationKernal<T>* sink_kernal;  /*** 使用本操作数的算子 ***/
};


/**
 * @description: 算子: 输入+输出+计算逻辑
 */
template <typename T>
class OperationKernal
{
public:
    using ArmaRowType = typename ArmaType<T>::rowtype;
    virtual ~OperationKernal() = default;

    virtual int Forward() = 0;
    virtual int GetConfig(OpKernalConfig& config) = 0;

    virtual int SetInput(std::vector<Operand<T>*>& inputs)
    {
        ZASSERT_RETURN_WITHSTR(!inputs.empty(), -1, "input is empty!")
        for (auto& item : inputs)
        {
            inputs_.push_back(item);
        }
        return 0;
    }

    //virtual int SetOutput(std::vector<Operand<T>*>& outputs)
    //{
    //    ZASSERT_RETURN_WITHSTR(!outputs.empty(), -1, "outputs is empty!")
    //    for (auto& item : outputs)
    //    {
    //        outputs_.push_back(item);
    //    }
    //    return 0;
    //}

private:
    OperatorType type_;  /*** 算子的类型 ***/
    std::vector<Operand<T>*> inputs_;  /*** 算子的输入 ***/
    std::vector<Operand<T>*> outputs_;  /*** 算子的输出 ***/
};
//using OpKernal_sptr<T> = std::shared_ptr<OperationKernal<T>>;

/**
 * @description: 计算图: 处理各个算子(operator/kernal)之间的有序计算
 */
template <typename T>
class ComputatinalGraph
{
public:
    virtual ~ComputatinalGraph()
    {}

private:
    std::vector<std::shared_ptr<OperationKernal<T>> > kernal_nodes_;
};
//using CmGraph_sptr = std::shared_ptr<ComputatinalGraph>;

}
#endif  //__COMPUTAYIONAL_GRAPH_H__

