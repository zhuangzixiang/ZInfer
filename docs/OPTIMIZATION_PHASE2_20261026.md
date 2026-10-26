# 第二阶段：算子归因与通用优化（2026-10-26）

本轮完成性能归因、CPU 拼接/卷积布局、GPU 全局池化、CUDA 卷积展开与 CPU 执行对象缓存。尚未完成全部路线图；CPU 仍未达到30FPS；Vulkan曾出现偶发长帧；Metal只有源码更新、未实机验收。

## 1. 可信的前后对照

保持旧 DLL 与 benchmark 在 `build/optimization_phase2/baseline/`，候选试验在 `candidate/`，最终版本在 `final/`。CPU RGB24 内存输入，输入、权重、线程数相同；文件读取/解码、权重加载和预热排除。包括完整 CPU 结果：人脸框+106点、独立人物框、原图尺寸 mask。

硬件：i5-14600KF，RTX 4070 Ti SUPER 16GB，驱动610.88；8线程，5次预热、100次测量，FP32。后端和版本串行运行。性能表禁用 Nsight、oneDNN verbose 和 ZNN profile。

| 后端 | 尺寸 | 旧版 mean ms | 新版 mean ms | 新版P95 ms | 新版P99 ms | 新版max ms | 新版FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| cpu | 640 | 273.857 | 185.792 | 195.656 | 200.005 | 200.860 | 5.38 |
| cuda | 640 | 18.828 | 11.438 | 13.496 | 14.517 | 15.217 | 87.43 |
| vulkan | 640 | 33.268 | 25.549 | 27.405 | 27.796 | 27.800 | 39.14 |
| cpu | 320 | 78.967 | 48.641 | 52.910 | 56.657 | 62.959 | 20.56 |
| cuda | 320 | 7.538 | 6.096 | 8.866 | 9.380 | 11.019 | 164.04 |
| vulkan | 320 | 15.140 | 11.323 | 12.986 | 13.197 | 13.240 | 88.31 |

| 后端 | 尺寸 | 预处理 | 上传 | 前向 | 下载 | 后处理 |
|---|---:|---:|---:|---:|---:|---:|
| cpu | 640 | 1.630 | 0.001 | 183.251 | 0.000 | 0.910 |
| cuda | 640 | 1.597 | 0.337 | 6.336 | 2.292 | 0.877 |
| vulkan | 640 | 1.618 | 0.569 | 19.018 | 3.468 | 0.876 |
| cpu | 320 | 0.446 | 0.001 | 47.370 | 0.000 | 0.823 |
| cuda | 320 | 0.444 | 0.150 | 4.136 | 0.585 | 0.782 |
| vulkan | 320 | 0.427 | 0.197 | 8.945 | 0.963 | 0.792 |

注意旧版CPU本轮640为273.857 ms，之前一次为408.479 ms。机器状态/调度波动原因未确定，不能用较慢历史值夸大加速比。本轮优先以上述保存旧DLL的同轮对照评价收益。

## 2. GPU 时间线解释 FP16 为什么慢

Nsight CUDA 捕获是5次预热+10次测量，共15次前向；下表为所有15次的kernel总时长均值，含预热，不能与禁用插桩的热态端到端耗时直接相减。

| 原实现 CUDA 项目 | FP32 ms/前向 | FP16 ms/前向 |
|---|---:|---:|
| 全局平均池化 | 4.388 | 4.284 |
| im2col | 4.268 | 4.204 |
| GEMM | 2.066 | 0.981 |
| F32→F16转换 | 0 | 1.806 |

每帧216次转换；GEMM节省约1.085 ms，转换花费约1.806 ms，转换开销已经超过GEMM收益。临时转换缓冲原本已复用，问题不是每次都cudaMalloc；权重内容仍反复转换，其他算子和输出仍主要是F32。

新CUDA FP32时间线：全局池化0.038 ms/前向，im2col0.955 ms/前向。二者是本轮主要收益来源，不是调高精度模式开关。最新FP16/TF32的额外100帧结果保留在validation/final_cuda_FP16.log、final_cuda_TF32.log；仍不能代替完整数据集精度验收。

## 3. 改动和验收

| 工作 | 实施与结果 |
|---|---|
| 通用观测 | `ZNN_PROFILE_FILE` 输出设备/算子/输入形状/次数/主机耗时；GPU记录的是主机提交耗时，不冒充GPU执行时间；嵌套调用耗时包含子调用 |
| CPU cat | 连续输入按拼接轴整块memcpy；非连续视图保留正确的逐元素路径；覆盖批次/不同轴/偏移/梯度 |
| cat正确性 | 新测试发现CUDA/Vulkan负轴处理有误；在ZNN公共入口归一化轴，所有后端共享修复 |
| 全局平均池化 | CUDA/Vulkan/Metal每通道工作组并行归约；CPU沿用oneDNN；保留普通窗口、padding和梯度语义 |
| CUDA卷积 | im2col布局改为[N,C*KH*KW,HO*WO]，相邻线程处理相邻空间位置；GEMM直接写NCHW，移除输出转置和y2临时张量 |
| CPU oneDNN布局 | 外部仍为NCHW，内部让oneDNN选择AVX2友好布局；分组/深度卷积不再被固定plain布局限制到通用GEMM |
| CPU执行计划 | 按形状/stride/pad/dilation/groups缓存描述符、primitive和外部memory对象；不缓存权重内容，每次读取最新权重并重排，训练更新及data()写入安全 |
| 内存取舍 | 保留每层大缓冲的试验峰值约407.5 MiB，收益约1%；最终取消这种保留方式，30帧复测峰值约184.8 MiB，耗时约185.0 ms。此数是CPU进程峰值工作集，不是显存 |
| 回归 | CPU/CUDA/Vulkan 477项通过；禁用新oneDNN布局的回退路径477项通过；真实模型新旧FP32输出差小于1e-4，详见JSON |
| Metal | 全局池化源码同步实现，缺Apple机器，未编译/未实测，未宣称通过 |

CPU计划最多128项，使用张量/临时区估算尺寸进行256 MiB的LRU记账；这是缓存记账上限，不是进程内存硬上限。oneDNN内部自身的缓存及临时分配不等同于此上限。当前ZNN运行时尚不承诺多线程推理会话安全。缓存为线程局部，数量/记账上限是每个线程的上限。

oneDNN verbose诊断显示：原卷积执行约114.44 ms/帧；新卷积约41.61 ms/帧，但增加约37.03 ms/帧的reorder。它说明下一步需要跨算子保留内部布局，而不是仅继续压缩卷积kernel。verbose包含开销，不能代替无插桩性能表。

## 4. Vulkan尾延迟仍需验收

前后对照的一次100帧测试中，新版640出现P99=108.343 ms、max=108.776 ms；旧版也出现长帧。随后开启flush分项记录的300帧复测得到mean25.579 ms、P99=28.256 ms、max29.367 ms。异常未稳定复现，不能删除异常样本并宣称已经稳定。

Nsight Vulkan跟踪需要注册跟踪层；当前权限环境下普通运行和提升后的尝试都无法完成注册，因此没有拿到Vulkan设备时间线。改用ZNN内置`ZNN_VK_TIMING_FILE`记录提交、vkQueueWaitIdle、清理耗时。复测中最长约77.9 ms等待发生在首轮预热；计时区没有重现此前长帧。尚不能判定此前长帧来自驱动、系统调度或框架，保持未解决状态。

## 5. 复现

以ZInfer为当前目录，在VS2019 x64开发者命令行构建：

```powershell
cmake --build build-dll --target znn ztestkit
# UniFP沿用现有CMake配置（含oneDNN部署路径）
cmake --build ../UniFP/build --target unifp_bench unifp_train unifp_eval unifp_infer
$env:OMP_NUM_THREADS='8'
$env:ZNN_STRICT_NATIVE='1'
../UniFP/build/unifp_bench.exe ../UniFP/runs/ftk/abs_12000.ufp cuda 640 100 build/perf_audit_20261026/input.rgb 1080 1273 3240 FP32
# 换cpu/vulkan及320；不要并行跑性能用例。
```

独立诊断（不将其耗时当性能验收）：

```powershell
$env:ZNN_PROFILE_FILE='build/optimization_phase2/profile/cpu_ops_new.csv'
../UniFP/build/unifp_bench.exe ../UniFP/runs/ftk/abs_12000.ufp cpu 640 3 build/perf_audit_20261026/input.rgb 1080 1273 3240 FP32
Remove-Item Env:ZNN_PROFILE_FILE
$env:ZNN_VK_TIMING_FILE='build/optimization_phase2/profile/vk_new.csv'
../UniFP/build/unifp_bench.exe ../UniFP/runs/ftk/abs_12000.ufp vulkan 640 300 build/perf_audit_20261026/input.rgb 1080 1273 3240 FP32
Remove-Item Env:ZNN_VK_TIMING_FILE
# ZNN_NO_ONEDNN=conv_layout 可诊断旧卷积路径；conv_cache 可禁用计划缓存。
```

Nsight文件、CPU算子CSV、oneDNN verbose在profile/；各版本日志、四输出原始张量、回归与manifest在validation/；修改均属于ZNN通用算子/后端，无UniFP数值绕行，数据集和56k权重未改动。

## 6. 下一阶段边界

- [x] CPU算子、oneDNN内核与CUDA GPU时间线归因。
- [x] CPU拼接与卷积布局、CUDA/Vulkan全局池化、CUDA im2col；Metal源码同步。
- [x] CPU执行对象缓存，形状变化/权重修改/梯度回归，去掉高内存低收益方案。
- [ ] ZNN统一执行计划：跨算子布局、缓冲生命周期、冻结权重与更新失效契约。
- [ ] 原生低精度常驻/预打包权重；必须先解决上一项，不能按裸地址缓存可变权重。
- [ ] LN+affine+SiLU融合、CUDA Graph、Vulkan命令重放。
- [ ] Vulkan不可映射本地内存staging、跨设备预算与尾延迟复现。
- [ ] Metal实机、更多真实RGB图像与长时稳定性；低精度完整数据集精度验收。

最终部署后已用真实56k权重完成1步CUDA训练回归（batch=1+1），loss=2.60835，梯度有限并成功保存检查点，输出在validation/train_one_step/。本轮没有重启长训练，也没有重新执行全量精度评估；前一阶段的精度指标保持历史记录，新版本数值验证范围如上。

```json
{
  "baseline_dll": "F9DD16843519177946C40AD4D0A9F3C6E4E8B3749D442B5A7D6C2118D30C234C",
  "final_dll": "294837484058B15E48FEF23A9ABE5276054E1238E0170FABD0C2C66F867B7DEF",
  "weights": "A1D7F65280175D7F4DDECBD188E36E78A45385CF881D2E22CB6AF205B64D1BB2",
  "rgb": "1E34564A41295E2B187F83293DAAF0D8A1F9D75134B5D1C895AEF70B09C49C39",
  "input": "1080x1273 RGB24 stride3240",
  "precision": "FP32",
  "warmup": 5,
  "iterations": 100,
  "threads": 8,
  "tests": 477
}
```
