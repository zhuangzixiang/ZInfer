/*
 * @Description: 常规卷积
 * @Author: ZhuangZixiang
 * @Date: 2025-04-28 23:28:22
 * @LastEditTime: 2025-05-06 17:33:06
 * @LastEditors: ZhuangZixiang
 */
#ifndef __NORMAL_CONVOLUTION_H__
#define __NORMAL_CONVOLUTION_H__

//template <typename T>
//class Convalutoin_Normal : public ConvalutoinBase
//{
//public:
//
//private:
//};

#include "../../tensor.h"
#include "../../com_opt_defines.h"
#include "zcom_tick.h"

namespace zinfer
{

template <typename T>
class Convolution_Normal/* : public ConvalutoinBase<T>*/
{
public:
    using simd = SimdType<T>;

    Convolution_Normal(uint32_t padding, uint32_t stride, bool use_bias)
        : padding_(padding), stride_(stride), use_bias_(use_bias)
    {}

    // 设置偏置项（支持多输出通道）
    void SetBias(std::vector<Tensor<T>>& bias)
    {
        // 使用移动语义避免拷贝，要求调用方后续不再使用原始bias对象
        biases_ = std::move(bias);

        /* 有效性验证示例：
            * 每个偏置张量应为向量形式（1x1x1xC）
            * 需要与卷积核数量一致
            */
        if (!biases_.empty())
        {
            for (const auto& b : biases_)
            {
                if (b.channels() != 1 || b.rows() != 1 || b.cols() != 1)
                {
                    throw std::invalid_argument("Bias must be 1x1x1xC format");
                }
            }
        }
    }

    // 设置卷积核（支持多输入/输出通道）
    void SetFilters(std::vector<Tensor<T>>& filters)
    {
        filters_ = std::move(filters);

        /* 卷积核格式要求：
            * 每个卷积核尺寸：[height, width, in_channels]
            * 卷积核数量决定输出通道数
            */
        if (!filters_.empty())
        {
            const uint32_t in_channels = filters_[0].channels();
            for (const auto& f : filters_)
            {
                if (f.channels() != in_channels)
                {
                    throw std::invalid_argument("All filters must have same input channels");
                }
            }
        }
    }

    // 设置输入数据（支持批量处理）
    void SetInputs(std::vector<Tensor<T>>& inputs)
    {
        inputs_ = std::move(inputs);

        /* 输入数据格式要求：
            * 每个输入张量尺寸：[height, width, channels]
            * 通道数必须与卷积核输入通道数一致
            */
        if (!inputs_.empty() && !filters_.empty())
        {
            const uint32_t required_channels = filters_[0].channels();
            for (const auto& in : inputs_)
            {
                if (in.channels() != required_channels)
                {
                    throw std::invalid_argument("Input channels mismatch with filters");
                }
            }
        }
    }

    // 执行卷积操作
    int DoConv(std::vector<Tensor<T>>& output)
    {
        // 清空并预分配输出容器
        output.clear();
        output.reserve(inputs_.size());

        // 遍历每个输入样本
        for (const auto& input : inputs_)
        {
            // 1. 数据填充（零填充）
            const auto padded_input = PadInput(input);

            // 2. 计算输出尺寸
            const auto output_shape = CalculateOutputShape(padded_input);
            Tensor<T> output_tensor(output_shape);

            // 3. 执行多通道卷积
            auto tick_in = zmodule::zcom::tick_time::now_mills();
            ConvolutionImpl(padded_input, output_tensor);
            auto tick_out = zmodule::zcom::tick_time::now_mills();
            std::cout << "ConvolutionImpl cost " << (tick_out - tick_in) << "ms" << std::endl;

            // 4. 添加偏置项
            if (use_bias_ && !biases_.empty())
            {
                AddBias(output_tensor);
            }

            output.emplace_back(std::move(output_tensor));
        }

        return 0; // 成功返回0
    }

private:
    // 成员变量
    uint32_t padding_ = 0;
    uint32_t stride_ = 1;
    bool use_bias_ = false;
    std::vector<Tensor<T>> biases_;
    std::vector<Tensor<T>> filters_;
    std::vector<Tensor<T>> inputs_;

    // 零填充实现（返回填充后的新张量）
    Tensor<T> PadInput(const Tensor<T>& input)
    {
        std::vector<uint32_t> padding_size(4, padding_);
        return std::move(input.Paddingd(padding_size, 0));
        //return input.Paddingd(padding_size, 0);

        //const uint32_t pad = padding_;
        //const auto& data = input.data(); // 假设GetData返回arma::fcube&
        //
        //// 计算填充后尺寸
        //const uint32_t new_rows = data.n_rows + 2 * pad;
        //const uint32_t new_cols = data.n_cols + 2 * pad;
        //
        //// 创建填充张量
        //Tensor<T> padded_tensor({new_rows, new_cols, input.channels()});
        //auto& padded_data = padded_tensor.data();
        //
        //// 中心区域填充原始数据
        //padded_data.subcube(pad, pad, 0, 
        //                   pad + data.n_rows - 1,
        //                   pad + data.n_cols - 1,
        //                   data.n_slices - 1) = data;
        //
        //return padded_tensor;
    }

#if 0
    // 卷积核实现（性能关键部分）
    void ConvolutionImpl(const Tensor<T>& padded_input, Tensor<T>& output)
    {
        const auto& input_data = padded_input.data();
        auto& output_data = output.data();

        const uint32_t out_channels = filters_.size();
        const uint32_t in_channels = filters_[0].channels();

        // 并行化外层循环（OpenMP示例）
#pragma omp parallel for collapse(2)
        for (uint32_t oc = 0; oc < out_channels; ++oc)
        {
            for (uint32_t ic = 0; ic < in_channels; ++ic)
            {
                // 获取当前卷积核切片
                //const auto& filter = filters_[oc].data().slice(ic);
                const auto& filter = filters_[oc].data();

                // 滑动窗口卷积
                for (uint32_t r = 0; r < output_data.n_rows; ++r)
                {
                    const uint32_t input_r = r * stride_;
                    for (uint32_t c = 0; c < output_data.n_cols; ++c)
                    {
                        const uint32_t input_c = c * stride_;

                        const auto window = input_data.subcube(
                            arma::span(input_r, input_r + filter.n_rows -1),
                            arma::span(input_c, input_c + filter.n_cols -1),
                            arma::span::all
                        );
                        // 三维逐元素乘法 + 累加
                        const float sum = arma::accu(window % filter);
                        output_data(r, c, oc) = sum;
                    }
                }
            }
        }
    }

#else
#ifdef USE_OPT_SIMD
    //void ConvolutionImpl(const Tensor<T>& padded_input, Tensor<T>& output)
    //{
    //    using simd = SimdType<T>;
    //    constexpr int vec_size = simd::elements_per_vector;
    //
    //
    //    const auto& input_data = padded_input.data();
    //    auto& output_data = output.data();
    //
    //    const uint32_t out_channels = filters_.size();
    //    const uint32_t in_channels = filters_[0].channels();
    //    const uint32_t in_channelsize = padded_input.rows() * padded_input.cols();
    //    const uint32_t in_filterchannelsize = filters_[0].rows() * filters_[0].cols();
    //
    //    // 并行化外层循环（OpenMP示例）
    //    for (int oc = 0; oc < out_channels; ++oc)
    //    {
    //        const auto& filter = filters_[oc].data();
    //
    //        // 滑动窗口卷积
    //        for (int c = 0; c < output_data.n_cols; ++c)
    //        {
    //            const uint32_t input_c = c * stride_;
    //            const uint32_t offset_cols = input_c * input_data.n_rows;
    //            for (int r = 0; r < output_data.n_rows; ++r)
    //            {
    //                const uint32_t input_r = r * stride_;
    //
    //                T sum = 0.0f;
    //                //typename simd::vec_type acc_vec = simd::zero();  // 新增1：创建SIMD累加器
    //                //for (int ic = 0; ic < in_channels; ++ic) {
    //                //    const uint32_t offset_channel = ic * in_channelsize;
    //                //    const T* input_ptr = input_data.memptr() + offset_channel + offset_cols + input_r;
    //                //    const uint32_t offset_filterchannel = ic * in_filterchannelsize;
    //                //    const T* filter_ptr = filter.memptr() + offset_filterchannel;
    //                //
    //                //    uint32_t offset_r_filter = 0;// kc* filter.n_rows;
    //                //    uint32_t offset_r_input = 0;// kc* input_data.n_rows;
    //                //    for (int kr = 0; kr < filter.n_rows; kr++) {
    //                //        // 新增2：SIMD乘加操作（自动处理非连续内存）
    //                //        auto input_v = simd::load(input_ptr + offset_r_input + kr); // 非对齐加载
    //                //        auto filter_v = simd::load(filter_ptr + offset_r_filter + kr);
    //                //        acc_vec = simd::fmadd(input_v, filter_v, acc_vec); // 融合乘加
    //                //    }
    //                //    offset_r_filter += filter.n_rows;
    //                //    offset_r_input += input_data.n_rows;
    //                //}
    //                //sum = simd::horizontal_sum(acc_vec);  // 新增3：水平求和
    //                //output_data(r, c, oc) = sum;  // 新增4：保留原有输出
    //
    //
    //                for (int ic = 0; ic < in_channels; ++ic) {
    //                    const uint32_t offset_channel = ic * in_channelsize;
    //                    const T* input_ptr = input_data.memptr() + offset_channel + offset_cols + input_r;
    //                    const uint32_t offset_filterchannel = ic * in_filterchannelsize;
    //                    const T* filter_ptr = filter.memptr() + offset_filterchannel;
    //
    //                    // ★★★ 关键修改1：二维卷积核遍历
    //                    for (int kr = 0; kr < filter.n_cols; kr++) { // 行方向
    //                        const T* in_row = input_ptr + kr * input_data.n_rows; // 跨行访问
    //                        const T* flt_row = filter_ptr + kr * filter.n_rows;    // 二维滤波器
    //
    //                        // ★★★ 关键修改2：向量化列方向
    //                        int kc = 0;
    //                        typename simd::vec_type acc = simd::zero();
    //                        for (; kc <= filter.n_rows - vec_size; kc += vec_size) { // 列向量处理
    //                            auto v_in = simd::load(in_row + kc);   // ★★★ 正确跨行加载
    //                            auto v_flt = simd::load(flt_row + kc); // 滤波器列连续
    //                            acc = simd::fmadd(v_in, v_flt, acc);
    //                        }
    //                        sum += simd::horizontal_sum(acc);
    //
    //                        // ★★★ 关键修改3：处理剩余列元素
    //                        for (; kc < filter.n_cols; ++kc) // 解决未处理数据问题
    //                            sum += in_row[kc] * flt_row[kc];
    //                    }
    //                }
    //                output_data(r, c, oc) = sum;
    //            }
    //        }
    //    }
    //}

    void img2Chn(const Tensor<T>& padded_input, std::vector<T>& out)
    {
        out.resize(padded_input.size());

        //auto* input_ptr = const_cast<Tensor<T>&>(padded_input).rawptr();
        auto* input_ptr = padded_input.rawptr();

        int count = 0;
#pragma omp parallel for collapse(3)
        for (int chn = 0; chn < (int)padded_input.channels(); chn++)
        {
            for (int col = 0; col < (int)padded_input.cols(); col++)
            {
                for (int row = 0; row < (int)padded_input.rows(); row++)
                {
                    //out[(col * padded_input.rows() + row) * padded_input.channels() + chn] = *(input_ptr + count);
                    out[(col * padded_input.rows() + row) * padded_input.channels() + chn] = input_ptr[count];
                    count++;
                }
            }
        }


//#pragma omp parallel for collapse(3)
        //const int const_offset = padded_input.rows() * padded_input.channels();
        //int out_offset_col = 0;
        //int out_offset_row = 0;
        //for (int chn = 0; chn < (int)padded_input.channels(); chn++)
        //{
        //    out_offset_col = chn;
        //    for (int col = 0; col < (int)padded_input.cols(); col++)
        //    {
        //        out_offset_row = 0;
        //        for (int row = 0; row < (int)padded_input.rows(); row++)
        //        {
        //            out[out_offset_col + out_offset_row] = input_ptr[count++];
        //            out_offset_row += padded_input.channels();
        //        }
        //        out_offset_col += const_offset;
        //    }
        //}

//        const int channels = padded_input.channels();
//        const int cols = padded_input.cols();
//        const int rows = padded_input.rows();
//        const int total_size = channels * cols * rows;
//#pragma omp parallel for collapse(2)
//        for (int col = 0; col < cols; col++) {
//            for (int row = 0; row < rows; row++) {
//                const int spatial_offset = col * rows + row;
//                for (int chn = 0; chn < channels; chn++) {
//                    const int input_idx = (spatial_offset * channels) + chn;
//                    const int output_idx = (chn * cols * rows) + (col * rows) + row;
//                    out[output_idx] = input_ptr[input_idx];
//                }
//            }
//        }



//        const int channel_size = padded_input.cols() * padded_input.rows();
//        const int channels = padded_input.channels();
//        const int cols = padded_input.cols();
//        const int rows = padded_input.rows();
//        const int total_size = channels * cols * rows;
//#pragma omp parallel for
//        for (int r = 0; r < total_size; r++)
//        {
//            const int chn = r / channel_size;
//            const int col = (r % channel_size) / rows;
//            const int row = (r % channel_size) % rows;
//            out[r] = input_ptr[(col * rows + row) * channels + chn];
//        }

    }

//     void ConvContinueVec(const std::vector<T>& input_vec, const std::vector<T> filter_vec, simd::vec_type& result)
//     {
//         using simd = SimdType<T>;
//         constexpr int vec_size = simd::elements_per_vector;
// 
//         int last_elem_count = input_vec.size() % vec_size;
//         int simd_count = input_vec.size() - last_elem_count;
//         for (int r = 0; r < simd_count; r += vec_size)
//         {
//             auto data_in = simd::load(input_vec.data() + r);
//             auto data_flt = simd::load(filter_vec.data() + r); 
//             result = simd::fmadd(data_in, data_flt, result);
//         }
// 
//         if (last_elem_count > 0)
//         {
//             auto* data_in_ptr = input_vec.data() + simd_count;
//             auto* data_flt_ptr = filter_vec.data() + simd_count;
//             for (int r = 0 r < last_elem_count; r++)
//             {
//                 result += data_in_ptr[r] * data_flt_ptr[r];
//             }
//         }    
//     }
    void ConvContinueVec(const T* input_vec, int input_vec_len, const T* filter_vec, int filter_vec_len, typename simd::vec_type& result)
    {
        constexpr int vec_size = simd::elements_per_vector;

        int last_elem_count = input_vec.size() % vec_size;
        int simd_count = input_vec.size() - last_elem_count;
        for (int r = 0; r < simd_count; r += vec_size)
        {
            auto data_in = simd::load(input_vec.data() + r);
            auto data_flt = simd::load(filter_vec.data() + r);
            result = simd::fmadd(data_in, data_flt, result);
        }

        if (last_elem_count > 0)
        {
            auto* data_in_ptr = input_vec.data() + simd_count;
            auto* data_flt_ptr = filter_vec.data() + simd_count;
            for (int r = 0; r < last_elem_count; r++)
            {
                result += data_in_ptr[r] * data_flt_ptr[r];
            }
        }

    }

    void ConvolutionImpl(const Tensor<T>& padded_input, Tensor<T>& output)
    {
        using simd = SimdType<T>;
        constexpr int vec_size = simd::elements_per_vector;

        std::vector<T> input_vec;
        img2Chn(padded_input, input_vec);

        std::vector<T> filter_vec;
        for (int f_idx = 0; f_idx < filters_.size(); f_idx++)
        {
            const auto& filter = filters_[f_idx];
            img2Chn(filter, filter_vec);
        
            //// 因为转为chn优先，在卷积时filter的宽高内，每列的chn是连续的，但是列之间不连续
            //typename simd::vec_type acc_vec = simd::zero();
            //for (int f_clo = 0; f_clo < filter.cols(); f_clo++)
            //{
            //    //ConvContinueVec(input_vec, )
            //}
        }

    }


#else
    void ConvolutionImpl(const Tensor<T>& padded_input, Tensor<T>& output)
    {
        const auto& input_data = padded_input.data();
        auto& output_data = output.data();
    
        const uint32_t out_channels = filters_.size();
        const uint32_t in_channels = filters_[0].channels();
        const uint32_t in_channelsize = padded_input.rows() * padded_input.cols();
        const uint32_t in_filterchannelsize = filters_[0].rows() * filters_[0].cols();
    
        // 并行化外层循环（OpenMP示例）
    //#pragma omp parallel for collapse(3)
        for (int oc = 0; oc < out_channels; ++oc)
        {
            const auto& filter = filters_[oc].data();
    
            // 滑动窗口卷积
            for (int c = 0; c < output_data.n_cols; ++c)
            {
                const uint32_t input_c = c * stride_;
                const uint32_t offset_cols = input_c * input_data.n_rows;
                for (int r = 0; r < output_data.n_rows; ++r)
                {
                    const uint32_t input_r = r * stride_;
    
                    T sum = 0.0f;
                    for (int ic = 0; ic < in_channels; ++ic)
                    {
                        const uint32_t offset_channel = ic * in_channelsize;
                        const T* input_ptr = input_data.memptr() + offset_channel + offset_cols + input_r;
                        const uint32_t offset_filterchannel = ic * in_filterchannelsize;
                        const T* filter_ptr = filter.memptr() + offset_filterchannel;
                        //std::cout << input_ptr[0] << " " << padded_input.at(ic, r, c) << " " << padded_input.index(offset_channel + offset_rows + input_c) << std::endl;
                        // input_ptr[] 和 padded_input.index()一致，但是和padded_input.at()不一致，前两者应该是以列优先去偏移的
    
    #if 1               
                        uint32_t offset_r_filter = 0;// kc* filter.n_rows;
                        uint32_t offset_r_input = 0;// kc* input_data.n_rows;
                        for (int kc = 0; kc < filter.n_cols; kc++) {
                            //const uint32_t offset_r_filter = kc * filter.n_rows;
                            //const uint32_t offset_r_input = kc * input_data.n_rows;
                            for (int kr = 0; kr < filter.n_rows; kr++) {
                                sum += input_ptr[offset_r_input + kr] * filter_ptr[offset_r_filter + kr];
    
                                //sum += input_ptr[kr * input_data.n_cols + kc] * filter(kr, kc, ic);
                                //sum += input_ptr[kc * input_data.n_rows + kr] * filter_ptr[kc * filter.n_rows + kr];
                                //sum = padded_input.at(ic, r+kr, c+kc) * filter(kr, kc, ic);
                                //sum = input_data(r+kr, c+kc, ic) * filter(kr, kc, ic);
                                //sum = padded_input.index(offset_channel + offset_rows + input_c);
                                //sum = input_ptr[kr * input_data.n_cols + kc];
                                //sum = padded_input.at(ic, r + kr, c + kc);
                            }
                            offset_r_filter += filter.n_rows;
                            offset_r_input += input_data.n_rows;
                        }
    #else               
                            // only support filter:3x3xn
                        for (int kc = 0; kc < filter.n_cols; kc++) {
    
                            const uint32_t offset_r_filter = kc * filter.n_rows;
                            const uint32_t offset_r_input = kc * input_data.n_rows;
                            //sum += input_ptr[kr * input_data.n_cols + kc] * filter(kr, kc, ic);
                            sum += input_ptr[offset_r_input] * filter_ptr[offset_r_filter] +
                                input_ptr[offset_r_input + 1] * filter_ptr[offset_r_filter + 1] +
                                input_ptr[offset_r_input + 2] * filter_ptr[offset_r_filter + 2];
    
                            //sum = padded_input.at(ic, r+kr, c+kc) * filter(kr, kc, ic);
                            //sum = input_data(r+kr, c+kc, ic) * filter(kr, kc, ic);
                            //sum = padded_input.index(offset_channel + offset_rows + input_c);
                            //sum = input_ptr[kr * input_data.n_cols + kc];
                            //sum = padded_input.at(ic, r + kr, c + kc);
                        }
    #endif              
                    }
                    output_data(r, c, oc) = sum;
                }
            }
        }
    }
#endif
#endif

    // 输出尺寸计算
    TensorShape CalculateOutputShape(const Tensor<T>& padded_input)
    {
        const uint32_t out_channels = filters_.size();
        const auto& input = padded_input.data();

        const uint32_t output_rows = (input.n_rows - filters_[0].rows()) / stride_ + 1;
        const uint32_t output_cols = (input.n_cols - filters_[0].cols()) / stride_ + 1;

        return { output_rows, output_cols, out_channels };
    }

    // 偏置添加
    void AddBias(Tensor<T>& output) const
    {
        auto& output_data = output.data();

#pragma omp parallel for
        for (int oc = 0; oc < output.channels(); ++oc)
        {
            if (oc < biases_.size())
            {
                const float bias_val = biases_[oc].data()(0, 0, 0);
                output_data.slice(oc) += bias_val;
            }
        }
    }
};
}

#endif // __NORMAL_CONVOLUTION_H__