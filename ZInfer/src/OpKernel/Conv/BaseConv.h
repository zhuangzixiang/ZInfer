/*
 * @Description: 卷积基类
 * @Author: ZhuangZixiang
 * @Date: 2025-04-28 23:28:22
 * @LastEditTime: 2025-05-06 17:33:06
 * @LastEditors: ZhuangZixiang
 */
#ifndef __NORMAL_CONVOLUTION_H__
#define __NORMAL_CONVOLUTION_H__

template <typename T>
class ConvalutoinBase
{
public:
    ConvalutoinBase(uint32_t padding, uint32_t stride): padding_(padding), stride_(stride)
	{}
	
	virtual void SetFilters(std::vector<Tensor<T>>& filters);
	virtual void SetInputs(std::vector<Tensor<T>>& inputs);
	
	virtual int DoConv(std::vector<Tensor<T>>& output);

private:
    uint32_t padding_ = 0;
	uint32_t stride_ = 1;
};

#endif // __NORMAL_CONVOLUTION_H__