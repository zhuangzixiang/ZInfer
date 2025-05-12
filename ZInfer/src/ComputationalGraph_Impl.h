/*
 * @Description: 计算图模块
 * @Author: ZhuangZixiang
 * @Date: 2025-04-28 23:28:22
 * @LastEditTime: 2025-05-04 01:50:29
 * @LastEditors: ZhuangZixiang
 */
#ifndef __COMPUTAYIONAL_GRAPH_IMPL_H__
#define __COMPUTAYIONAL_GRAPH_IMPL_H__

#include "ComputationalGraph.h"
#include "zcom_trace.h"

/**
 * @description: 卷积算子: 输入 --> 卷积(*权重系数*偏移) --> 输出
 */
template <typename T>
class OpKernal_Convalutoin : public OperationKernal
{
public:
    OpKernal_Convalutoin(OpKernalConfig& cfg, OpKernalCalcuFactor<T>& param): 
            type_(OperatorType::OPTYPE_CONV),
            factor_weights_(std::move(param.factor_weights_)),
            factor_bias_(std::move(param.factor_bias_))
    {
        ZASSERT_RETURN_WITHSTR(cfg.dilation_size == 2 && cfg.dilation_ptr != nullptr, NULL, "dilation invalid!");
        ZASSERT_RETURN_WITHSTR(cfg.stride_size == 2 && cfg.stride_ptr != nullptr, NULL, "stride invalid!");
        ZASSERT_RETURN_WITHSTR(cfg.kernal_size == 2 && cfg.kernal_ptr != nullptr, NULL, "kernel invalid!");
        ZASSERT_RETURN_WITHSTR(cfg.channels_in > 0 && cfg.channels_out > 0, NULL, "channel invalid!");
        ZASSERT_RETURN_WITHSTR(cfg.groups > 0, NULL, "dilation invalid!");
        INFO_TRACE_THIS("bias:" << cfg.bias_use << " padding_mode:" << ToString(cfg.padding_mode));

        cfg_ = std::move(cfg);
        Img2Row(this->factor_weights_, this->factor_weights_row_);  
    }

    virtual ~OpKernal_Convalutoin()
    {}

    virtual int Forward() override
    {
        if (outputs_.size() < inputs_.size()) outputs_.resize(inputs_.size());


        const uint32_t convkernel_num = this->factor_weights_.size();
        const uint32_t convkernel_row = this->factor_weights_[0]->rows();
        const uint32_t convkernel_clo = this->factor_weights_[0]->cols();
        const uint32_t convkernel_chn = this->factor_weights_[0]->channels();
        const uint32_t convkernel_2row_len = convkernel_row * convkernel_clo;



        const uint32_t output_chns = convkernel_num;  /*** 每个卷积核对应输出的一个chnnel ***/
        ZASSERT_RETURN(output_chns % cfg_.groups == 0, -1);

        const uint32_t batch_size = inputs_.size();
        for (uint32_t i = 0; i < batch_size; ++i)
        {
            const std::shared_ptr<Tensor<T>>& input = inputs_[i];
            ZASSERT_RETURN_WITHSTR(input != nullptr && !input->empty(), -1, "input tensor " << i << "th invalid");

            const uint32_t input_chns = input->channels();
            ZASSERT_RETURN(input_chns % cfg_.groups == 0, -1);
            const uint32_t input_padded_row = input->rows() + 2 * cfg_.padding[0];
            const uint32_t input_padded_clo = input->cols() + 2 * cfg_.padding[1];
            ZASSERT_RETURN_WITHSTR(cfg_.padding[0] < this->factor_weights_[0]->rows() && cfg_.padding[1] < this->factor_weights_[0]->cols(), -1, "padding is to large!");

            const uint32_t output_row = std::floor((int(input_padded_row) - int(convkernel_row)) / stride[1] + 1);
            const uint32_t output_clo = std::floor((int(input_padded_clo) - int(convkernel_clo)) / stride[0] + 1);
            ZASSERT_RETURN_WITHSTR((output_row > 0 && output_clo > 0), -1, i << " output_size invalid!");

            if (outputs_[i] == nullptr || outputs_[i]->empty())
            {
                outputs_[i] = std::make_shared<Tensor<T>>(output_row, output_clo, output_chns);
            }

            uint32_t col_len = output_h * output_w;
            CHECK(col_len > 0) << "Output_h x output_w for the convolution layer "
                "should be greater than zero "
                << i << " th";

            //uint32_t input_chns_eachgroup = input_chns / cfg_.groups;
            //CHECK(input_chns_eachgroup == kernel_c) << "The number of channel for the kernel "
            //    "matrix and input tensor do not match";
            ZASSERT_RETURN(convkernel_chn == input_chns, -1);

            for (uint32_t group_idx = 0; group_idx < cfg_.groups; group_idx++)
            {
                const auto& input_matrix = Im2Col(input, kernel_w, kernel_h, input->cols(), input->rows(), input_chns_eachgroup, group_idx, row_len, col_len);

                std::shared_ptr<Tensor<float>> output_tensor = outputs.at(i);
                CHECK(output_tensor->rows() == output_h && output_tensor->cols() == output_w && output_tensor->channels() == kernel_count)
                    << "The output tensor array in the convolution layer has an incorrectly sized tensor " << i << "th";

                const uint32_t kernel_count_group_start = kernel_count_group * group_idx;
                for (uint32_t k = 0; k < kernel_count_group; ++k)
                {
                    arma::frowvec kernel = (groups_ == 1) ? kernel_matrix_arr_.at(k) : kernel_matrix_arr_.at(kernel_count_group_start + k);
                    ConvGemmBias(input_matrix, output_tensor, group_idx, k, kernel_count_group, kernel, output_w, output_h);
                }
            }
        }
        return InferStatus::kInferSuccess;

    }

    virtual int GetConfig(OpKernalConfig& config) override;

private:
    void Img2Row(const std::vector<std::shared_ptr<Tensor<T>>>& in_imgvec, std::vector<ArmaRowType>& out_imgvec_row)
    {
        if (in_imgvec.empty()) return;

        const uint32_t convkernel_num = this->factor_weights_.size();
        const uint32_t kernel_h = this->factor_weights_[0]->rows();
        const uint32_t kernel_w = this->factor_weights_[0]->cols();
        const uint32_t kernel_c = this->factor_weights_[0]->channels();
        const uint32_t row_len = kernel_h * kernel_w;
      
        ZASSERT_RETURN_VOID_WITHSTR((kernel_h > 0 && kernel_w > 0 && kernel_c > 0), "Invalid kernel dimensions");

        /*** 预计算并验证所有核尺寸 ***/
        #pragma omp parallel for
        for (uint32_t r = 0; r < convkernel_num; r++)
        {
            const auto& kernel = in_imgvec[r];
            ZASSERT_RETURN_VOID_WITHSTR((kernel->rows() == kernel_h && kernel->cols() == kernel_w && kernel->channels() == kernel_c)
                   , "Kernel " << k << " has inconsistent dimensions");
        }

        const uint32_t kernelnum_eachgroup = convkernel_num / cfg_.groups;
        out_imgvec_row.reserve(convkernel_num);  /*** 预分配空间 ***/
      
        /*** 统一处理逻辑，避免代码重复 ***/
        auto ProcessKernel = [&](uint32_t k_idx, ArmaRowType& kernel_row) {
            ArmaRowType kernel_row_item(row_len * kernel_c);

            const auto& kernel = in_imgvec[k_idx];
            for (uint32_t ic = 0; ic < kernel_c; ++ic)
            {
                std::memcpy(kernel_row_item.memptr() + row_len * ic, kernel->matrix_raw_ptr(ic), row_len * sizeof(float));
            }
            kernel_row = std::move(kernel_row_item);
        };

        if (cfg_.groups == 1)
        {
            for (uint32_t r = 0; r < convkernel_num; r++)
            {
                //out_imgvec_row.emplace_back(ProcessKernel(r));
                ProcessKernel(r, out_imgvec_row[r]);
            }
        }
        else
        {
            for (uint32_t group_idx = 0; group_idx < cfg_.groups; group_idx++)
            {
                const uint32_t group_offset = group_idx * kernelnum_eachgroup;
                #pragma omp parallel for
                for (uint32_t r = 0; r < kernelnum_eachgroup; r++)
                {
                    //out_imgvec_row.emplace_back(ProcessKernel(r + group_offset));
                    ProcessKernel(r + group_offset, out_imgvec_row[r + group_offset]);
                }
            }
        }
        //this->kernel_matrix_arr_ = std::move(kernel_matrix_arr);
    }

private:
    OpKernalConfig cfg_;  /*** 卷积层的配置参数(大小...) ***/
    //Tensor<T> factor_weights_;  /*** 卷积运算的系数：权重 ***/
    //Tensor<T> factor_bias_;  /*** 卷积运算的系数：偏移 ***/
    std::vector<std::shared_ptr<Tensor<T>>> factor_weights_;
    std::vector<std::shared_ptr<Tensor<T>>> factor_bias_;
    std::vector<ArmaRowType> factor_weights_row_;
};

template <typename T>
class OpKernal_MaxPooling : public OperationKernal
{
public:
    OpKernal_MaxPooling(): type_(OperatorType::OPTYPE_POOLING), 
    virtual ~OpKernal_MaxPooling();

    virtual int Forward() override;
    virtual int GetConfig(OpKernalConfig& config) override;
};

#endif  //__COMPUTAYIONAL_GRAPH_IMPL_H__

