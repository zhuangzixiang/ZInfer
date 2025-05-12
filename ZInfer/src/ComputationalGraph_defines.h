/*
 * @Description: 计算图相关定义
 * @Author: ZhuangZixiang
 * @Date: 2025-04-28 23:28:22
 * @LastEditTime: 2025-04-28 23:54:28
 * @LastEditors: ZhuangZixiang
 */
#ifndef __COMPUTAYIONAL_GRAPH_DEFINES_H__
#define __COMPUTAYIONAL_GRAPH_DEFINES_H__

#include <string>

enum class OperatorType: uint32_t
{
    OPTYPE_UNKOWN,
    OPTYPE_CONV,
    OPTYPE_POOLING,

};

enum class PaddingMode : uint32_t
{
    PMODE_VALID,  /*** valid: 无填充 ***/
    PMODE_SAME,  /*** same: 有填充，输入输出shape一样 ***/
    PMODE_FULL,  /*** full: 有填充， ***/

};

std::string ToString(OperatorType& otype)
{
  switch (otype)
  {
  case OperatorType::OPTYPE_CONV:
    return "convolution";
  case OperatorType::OPTYPE_POOLING:
    return "pooling";
  case OperatorType::OPTYPE_UNKOWN:
  default:
    return "unkown";
  }
}

std::string ToString(PaddingMode& pmode)
{
    switch (pmode)
    {
    case PaddingMode::PMODE_VALID:
        return "valid";
    case PaddingMode::PMODE_SAME:
        return "same";
    case PaddingMode::PMODE_FULL:
        return "full";
    default:
        return "valid";
    }
}
#endif  //__COMPUTAYIONAL_GRAPH_DEFINES_H__

