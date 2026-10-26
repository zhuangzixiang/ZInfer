# UniFP / ZNN 优化与 RGB 内存推理验收（2026-10-26）

本次完成主要正确性修复和性能优化，但**不等于全部路线图完成**。CPU 仍未达到实时；Vulkan 640 的尾延迟需按表判断；Metal 未实机验收。旧 STATUS 原始记录保留作历史，当前结果以本文为准。

## 1. 测量口径

- i5-14600KF、RTX 4070 Ti SUPER 16GB、驱动 610.88；Release；OMP_NUM_THREADS=8。
- 同一 56k 权重 `runs/ftk/abs_12000.ufp`；RGB 来源为 P3M val_P 的 p_0074c250.jpg，预先转为 1080×1273 RGB24，stride=3240。基准只在循环外读取 raw 文件，循环内从 CPU 内存开始，**不计 PNG/JPEG 解码、文件读取、权重加载与预热**。
- 5 次预热，100 次串行推理；FP32；640/320 是 letterbox 网络画布，输出 mask 仍为原图尺寸。不同后端测试串行运行。
- 总耗时包括 RGB resize/pad/NCHW/归一化、上传、前向、三个检测头与 mask 下载一次、独立 face/person 解码/NMS、106 点坐标还原和 CPU mask。
- 分项边界用 GPU 同步读保证完成。它包含分项同步开销，未将异步提交时间冒充执行时间。单张固定图重复测试只反映本机热态，不等价于视频长时稳定性或多图性能分布。
- RGB 接口不是 YUV 零拷贝接口。YUV 的转换在调用端完成；本表没有计入 YUV→RGB，应用采用 YUV 时应单独测并计入自身总预算。

## 2. CPU 内存 RGB → CPU 结果

| 后端 | 网络尺寸 | mean ms | P50 ms | P95 ms | P99 ms | max ms | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| cpu | 640 | 408.479 | 411.166 | 457.036 | 478.724 | 515.636 | 2.45 |
| cuda | 640 | 18.899 | 18.817 | 19.696 | 19.968 | 20.014 | 52.91 |
| vulkan | 640 | 31.075 | 30.889 | 33.127 | 33.482 | 33.768 | 32.18 |
| cpu | 320 | 108.145 | 117.981 | 126.496 | 132.243 | 133.939 | 9.25 |
| cuda | 320 | 7.761 | 7.582 | 9.237 | 9.967 | 10.118 | 128.85 |
| vulkan | 320 | 12.878 | 12.543 | 14.078 | 14.205 | 14.297 | 77.65 |

| 后端 | 网络尺寸 | 预处理 ms | 上传 ms | 前向 ms | 下载 ms | 后处理 ms |
|---|---:|---:|---:|---:|---:|---:|
| cpu | 640 | 1.991 | 0.002 | 405.330 | 0.000 | 1.155 |
| cuda | 640 | 1.638 | 0.355 | 13.655 | 2.363 | 0.888 |
| vulkan | 640 | 1.781 | 0.589 | 23.953 | 3.756 | 0.996 |
| cpu | 320 | 0.519 | 0.001 | 106.663 | 0.000 | 0.961 |
| cuda | 320 | 0.437 | 0.151 | 5.672 | 0.669 | 0.832 |
| vulkan | 320 | 0.447 | 0.222 | 10.339 | 1.075 | 0.795 |

Metal：本机为 Windows，无 Apple GPU，未编译、未测速；不填估计值。

CUDA 640 的额外精度模式对比：

| 模式 | 全流程 mean ms | P99 ms | 对 FP32 四输出最大绝对误差 |
|---|---:|---:|---:|
| FP32 | 18.899 | 19.968 | 0 |
| TF32 | 18.663 | 19.436 | 0.00905 |
| FP16 | 21.202 | 23.041 | 0.02792 |

FP16 的临时转换成本抵消收益，未采用为默认；TF32 收益很小。精度模式的单图误差不能代替完整验证集验收，当前默认保留 FP32。CPU/CUDA/Vulkan 的 FP32 四输出均有限且最大绝对差约 2.03e-5。390 项回归通过；详细数据见 numerics/。

以30 FPS的33.33 ms预算衡量：CUDA有余量；Vulkan640的P99=33.482 ms、max=33.768 ms，只有平均达标，不能称稳定达标；Vulkan320有余量；CPU两档均不达标。

旧审计中 Vulkan LOCAL 640 前向约 37.55 ms、回读约 236.63 ms；本次回读约几毫秒，主要消除了 CPU 直接读取未缓存 BAR 映射的瓶颈。历史基线与新 RGB 全流程口径不同，不能相除后宣称整体加速倍数。CUDA 也没有获得未经验证的十倍收益。

## 3. 每项修复、验证与结果

| 项目 | 实施 | 验证及发现 |
|---|---|---|
| Vulkan 内存 | 默认优先本地可映射计算内存；CPU 回读走 HOST_CACHED staging；扩大描述符池 | 100 帧、偏移回读、真实权重验证；应用不再需要设置 ZNN_VK_LOCAL=1 |
| Vulkan 视图 | 在 ZNN 派发边界材料化非连续/非零偏移视图 | 原实现从颈部开始读错连续切片；57 个 ConvNormAct 输出对拍误差约 1.1e-5，完整四输出误差见 JSON |
| CPU | oneDNN 分组卷积/广播描述符/执行参数修正；fallback 1×1 直接 GEMM、depthwise；空间分块 LN | CPU 仍慢；实测强制直接 GEMM/depthwise 比 oneDNN 优先更慢，最终保留 oneDNN 优先，快路径用于回落 |
| CPU SiLU | oneDNN swish alpha 由错误的 0 修正为 1 | 旧 CPU 实际执行 x/2，不能把旧 CPU 速度当成正确模型基线 |
| CUDA | 1×1 stride1/pad0 避免 im2col/转置；通道 LN 合并空间访问；常量原生填充 | 分组/批次/尾块、加权 LN 梯度、真实模型输出均验证 |
| Metal | 批量提交、原生填充、1×1 GEMM；归一化 eps/momentum 位解释与线程网格修正 | 仅源码修改，待 Mac 编译与同组数值/性能测试 |
| 公共 ZNN | 优化器状态保存恢复；LayerNorm 非连续视图；SGD 零动量越界修复 | CPU/CUDA/Vulkan 回归，STRICT_NATIVE 开启；NaN 不能再被 max_diff 漏判 |
| RGB 接口 | 借用 CPU RGB24，支持 stride；融合预处理；检测头只下载一次 | 9 组尺寸/填充行测试，与旧预处理逐值一致 |
| 训练恢复 | .ufp + .optim + .state；按全局 batch 的确定性采样与增强；有序预取 | 连续 4 步 vs 2+2，最终权重和优化器文件 SHA256 完全相同 |
| 评价 | overall AP 独立匹配，分桶忽略桶外 GT；完整 2000 张 LaPa | 旧 overall AP 三次计入同一预测；更正结果见下一节 |

算子修复在 ZNN 内实现，UniFP 没有加入针对错误的数值绕行。UniFP 的改动用于内存输入、训练状态与评价口径。

## 4. 56k 权重重评

本轮完整评价使用 DLL SHA256 `66B234B5CCD917EFF3C9286EE06E3621B3B68BB668997988FB8C01F609869E25`；最终测速使用文末 F9DD… DLL。两者之间仅增加 LN 非连续视图处理、优化器文件校验等收尾修正，不改变本模型 CUDA FP32 连续输入前向。

以下使用 CUDA FP32；WIDER 是自定义 AP50、最小脸边≥10px，**不是官方 WIDER easy/medium/hard AP**。人像框指标以 mask 连通域派生 GT 评价，不代表真实多人遮挡下独立人物框标注的精度。LaPa NME 只统计检测匹配成功样本，应同时看 matched 数量。

```text
[WIDER val AP50] big(>=64px): 0.7101  mid(32-64): 0.6452  small(10-32): 0.1566
[WIDER val AP50] overall(min>=10px): 0.4168  (gt=28587)
[P3M-500 mask IoU] 0.9606 (n=500)
[Person-box AP50 @P3M] 0.9515 (gt persons=513)
[PPM-100 mask IoU] 0.9448 (n=100)
[LaPa NME] 0.0856 (ocular-normalized, matched=1999/2000)
```

性能修复不会凭空改善训练能力。小脸检测、关键点仍需继续提升；先用完整采样与真实优化器恢复建立新的训练曲线，再决定损失/分辨率/模型容量调整。历史 56k 权重没有旧优化器状态，无法补造精确续训，只能首次 warm start；新检查点之后可以精确恢复。

## 5. 构建与复现

在 VS2019 x64 Native Tools 命令行中，以 ZInfer 为当前目录：

```powershell
cmake --build build-dll --target znn ztestkit
cmake -S ../UniFP -B ../UniFP/build -DCMAKE_BUILD_TYPE=Release -DZNN_ONEDNN_RUNTIME="E:/Documents/ZShared/ZProjs/ZSpeedAI/ZInfer/build/Release/dnnl.dll"
cmake --build ../UniFP/build --target unifp_bench unifp_train unifp_eval unifp_infer
$env:OMP_NUM_THREADS='8'
$env:ZNN_STRICT_NATIVE='1'
Remove-Item Env:ZNN_VK_LOCAL -ErrorAction SilentlyContinue
../UniFP/build/unifp_bench.exe ../UniFP/runs/ftk/abs_12000.ufp cuda 640 100 build/perf_audit_20261026/input.rgb 1080 1273 3240 FP32
# backend 换为 cpu / vulkan；网络尺寸换为 320。
# raw 文件读取在计时外；应用直接传自己的 RGB 指针，不需要 raw 文件。
../UniFP/build/unifp_eval.exe ../UniFP/runs/ftk/abs_12000.ufp ../UniFP/data all
```

oneDNN 路径是本机已存在的依赖示例，其他机器传实际文件路径。CMake 把它复制到应用目录；日志 `onednn_available=1` 表示库加载成功，不承诺每个算子都由 oneDNN 执行。未部署时有 Eigen/reference 回退，但性能不是本文 CPU 配置。新增公共 API 与 Optimizer 虚接口意味着需要同时更新 DLL/import library/header 并重编译消费者。

`UniFP/scripts/bench_rgb.ps1` 可串行测试后端/尺寸并生成 manifest；参数见脚本。`scripts/train_chain.ps1` 直接引用上一段完整检查点，禁止覆盖已有输出目录，避免只复制 .ufp 丢失状态。两者不会自行启动后台任务。

应用调用示例（初始化/加载发生一次）：

```cpp
znn::Runtime::set_grad_enabled(false);
znn::Runtime::set_precision(znn::Precision::FP32);
ufp::UniFPNet net;
std::vector<znn::Tensor> parameters;
net.collect(parameters);
if (!ufp::load_weights(parameters, "model.ufp")) throw std::runtime_error("weights");
net.set_train(false);
ufp::RgbView frame{rgb_pointer, width, height, row_stride_bytes};
auto result = ufp::predict_rgb(net, frame, znn::Device::CUDA, 640);
// result.faces: 人脸框+106点；persons: 独立人物检测框；mask: 原图大小的CPU二值mask。
```

RGB 内存须在调用期间有效；每个 net 实例固定一个后端；目前接口及运行时并非并发会话安全。训练恢复要求相同数据内容、排序、batch 配置与代码；尚未保存数据内容指纹，改变数据集后不可声称精确恢复。

## 6. Vulkan 开关与待完成范围

生产使用默认自动策略即可。`ZNN_VK_LOCAL=1` 与默认同义；`=0` 仅保留作诊断对照，不需要维护两套应用路径。当前自动策略支持可映射本地内存，缺失时回退主机可见内存；**纯不可映射 DEVICE_LOCAL 的上传 staging、预算驱动策略尚未完成**，其他设备不能套用本机速度结论。

尚未完成：CUDA Graph/通用执行计划、Vulkan 命令重放、LN+affine+SiLU 融合、原生低精度常驻/预打包权重、CPU 跨算子 blocked layout、Metal 实机验证。它们是后续工程，未把路线图勾选成已完成。当前 F16/BF16 存储仍有 F32 提升路径，Precision::FP16 并不等价于完整原生半精度模型执行。

## 7. 结果文件与校验

独立输出目录：`ZInfer/build/optimization_20261026/`，分 performance / numerics / resume / evaluation；详细原始张量和逐层 trace 留在 `build/perf_audit_20261026/`，不混入生产源码或数据集。无长训练任务遗留，原始权重与数据未修改。

```json
{
  "dll_sha256": "F9DD16843519177946C40AD4D0A9F3C6E4E8B3749D442B5A7D6C2118D30C234C",
  "weight_sha256": "A1D7F65280175D7F4DDECBD188E36E78A45385CF881D2E22CB6AF205B64D1BB2",
  "rgb_sha256": "1E34564A41295E2B187F83293DAAF0D8A1F9D75134B5D1C895AEF70B09C49C39",
  "input": "CPU RGB24 1080x1273 stride=3240",
  "precision": "FP32",
  "iterations": 100,
  "warmup": 5,
  "cpu_threads": 8,
  "gpu": "RTX 4070 Ti SUPER 16GB",
  "driver": "610.88",
  "cpu": "i5-14600KF",
  "resume_weights_equal": true,
  "resume_optimizer_equal": true
}
```
