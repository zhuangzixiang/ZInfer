# UniFP 三日提交、续训与跨后端性能复核（2026-10-26）

## 结论

本次完成源码审计与隔离测量，没有修改生产算子、训练代码、权重或数据集，没有启动训练。现有 STATUS 中“精度仅受 epoch 预算限制”“Vulkan 320 已满足应用实时”“小 GEMM 必须保留系统内存默认放置”的结论均需修订。CUDA 十倍提速目前没有证据保证。

首先处理评估、采样恢复和性能测量口径，再在 ZNN 内实施通用传输、卷积及执行优化；不能通过 UniFP 特定调用技巧绕过框架问题。

## 版本与测量条件

- UniFP：ed9770c6dfe68968537771b92e27305b0928820b。
- ZInfer 源码：316794394278a461dde4f236f46cd8b25e7f7e10。
- GPU 实机：NVIDIA GeForce RTX 4070 Ti SUPER；CPU：i5-14600KF。STATUS 的 Laptop 标注不适用于本次机器。
- 实测使用 UniFP/build/znn.dll，SHA256：0D931FE1155E77760F909EC0844E1171EB37CFA1501983D323656132347222C4。
- ZInfer/build-dll/znn.dll 是另一份二进制，SHA256：61401EF8F2DDB06F88762C0D20687695A2F7C9128C166EAD03693E127FFEADFE。不能把两个 DLL 的测量结果混用；源码提交也不单独证明二进制来源。
- 权重：UniFP/runs/ftk/abs_12000.ufp，SHA256：A1D7F65280175D7F4DDECBD188E36E78A45385CF881D2E22CB6AF205B64D1BB2。
- stage3.log 最终评估实际使用 abs_12000，不是 STATUS 写的 abs_8000。
- Release CMake 构建，FP32，关闭梯度，输入驻留，3 次预热；GPU 前向 20–30 次，CPU 3–5 次；回读连续 5 次。不是长期稳定性、P95/P99 或热平衡验收。
- 通过独立 1 元素设备张量回读完成队列同步，避免把完整 mask 回读混入前向。此操作仍有少量同步成本，不是纯 GPU timestamp。
- 真实图片：UniFP/data/P3M/val_P/images/p_0074c250.jpg。只读，未修改。
- 诊断源文件、构建与原始日志：ZInfer/build/perf_audit_20261026/。生成脚本仅复制并插桩 model.cpp，不改原模型。阶段 profile 与常态前向分别测量。

## 三日提交审计

| 时间 | 提交 | 变化 | 复核结果 |
|---|---|---|---|
| 10-23 | a493b44 | 原生 ZNN 网络与数据/验证重构 | 建立可训练基线，但需继续检查评估及续训状态 |
| 10-24 | c2cc665 | 最终评估器、16k 训练 | WIDER 总体 AP 合并算法有错误，LaPa 只评估 500 张 |
| 10-25 | 4bd2205 | 56k 分段续训、关键点精调 | 权重持续更新，但不是优化器和采样器完整恢复 |
| 10-25 | 3238dc4、998b46b | 前向基准、原生通道 LN | 减少布局转换合理；基准解码循环缺少自增 |
| 10-25 | c6db14a、a077ef9 | 临时工具修整与清理 | build 仍有陈旧工具，不能据可执行文件名称推断当前源码 |
| 10-26 | db707c1、58c63b8 | VK 探针、原生上采样 | 探针并非真实网络阶段，不能据此排定内核优先级 |
| 10-26 | ed9770c | STATUS 汇总 | 环境、最终权重、AP、实时口径均需补充修正 |

## 精度问题：先纠正诊断依据

1. **重复采样前缀**：stage3_chain.sh 每 2000 步启动一次 train；train 固定 Loader seed=1234；Mixer 每次重新 shuffle 并从 di=si=0 开始。检测池 12880+18168=31048，分割池 9421+2667=12088。每段 batch=2+1，主要再次训练相同前约 4000 个检测样本和 2000 个分割样本（线程完成顺序与预取边界可能有少量差异）。新增 56k 并没有按累计计数完整遍历全部训练集；此前 16k 阶段的覆盖另算。不能用累计样本数除数据集大小直接称为有效 5.5 epoch。
2. **优化器重置**：每段重新创建 AdamW，动量与时间步归零，代码也明确打印 not an exact resume。这是重复 warm-start，不是完整续训。需要保存模型、优化器、全局步数、采样位置及随机状态，并验证恢复一致性。
3. **总体 AP 错误**：final_eval.cpp 将每个预测分别加入三个尺寸 bucket，再把三个预测数组拼起来计算 overall；对某个尺寸的正确预测，在其他 bucket 会变成假阳性。总体必须对全部有效 GT 独立匹配一遍。尺寸分层还要明确其他尺寸 GT 的 ignore 规则。当前自定义尺寸分组也不是 WIDER 官方 Easy/Medium/Hard，不能直接对比官方目标。修正前不公布替代 AP 数值。
4. **关键点评估范围**：代码传 max_images=500，而 LaPa val 有 2000 张；记录 NME=0.0867 是 matched=500/500 的子集结果。后续同时报告全验证集 NME、检测成功率和失败率。
5. **人像框指标边界**：P3M 的框 GT 来自 mask 连通域，是弱标注评价，不能证明拥挤多人独立人物检测已达标。

建议保留全部现有权重，用统一修正后的评估器重评 16k、72k、精调末期；修好采样和续训后再增加训练预算。不能把所有差距归咎于训练步数，更不能据此立即增加 P2 或扩大模型。

## 真实耗时分解

### 前向与全部原始输出回读分开

全部原始输出包含三个 384 通道检测头和 mask：640 下约 14.54 MB，320 下约 3.64 MB；每份只下载一次。

| 后端/模式 | 输入 | 前向平均 ms | 全输出回读平均 ms | 两项相加 ms |
|---|---:|---:|---:|---:|
| CUDA，真实图 | 640 | 16.77 | 2.68 | 19.45 |
| CUDA，真实图 | 320 | 6.85 | 0.64 | 7.49 |
| Vulkan 默认，随机图 | 640 | 318.94 | 42.48 | 361.42 |
| Vulkan LOCAL，真实图 | 640 | 37.55 | 236.63 | 274.18 |
| Vulkan LOCAL，真实图 | 320 | 13.06 | 59.00 | 72.06 |
| CPU，无 oneDNN，8 线程配置 | 640 | 2432.94 | 不涉及 | 2432.94 |
| CPU，oneDNN，8 线程 | 640 | 1255.23 | 不涉及 | 1255.23 |

两项相加不是完整端到端实测：还未包括图片读取、预处理、上传、输出解码、mask 映射及业务渲染。单张图片 CPU 预处理约 5.5ms@640 / 1.4ms@320，上传约 0.17–0.60ms；未作多样本统计。真实图双类别 CPU 解码约 0.01–0.12ms，候选数依赖图片与阈值，不代表拥挤场景。

原 bench 只回读 seg，因而不能代表实际需要三个检测头的应用。原 infer 两次调用 decode，各自 to_device(CPU)，还会重复下载三个检测头；本表已采用只下载一次的更小口径，VK 仍不满足 30fps。320 分辨率也需要独立精度验收，不能用 640 精度配 320 速度。

### 真实模型阶段定位（640，预热后，ms）

| 阶段 | CUDA | VK LOCAL | CPU 无 oneDNN |
|---|---:|---:|---:|
| Stem + S1 | 3.97 | 6.75 | 652.35 |
| S2 | 2.41 | 5.23 | 360.50 |
| S3 | 1.59 | 5.18 | 254.17 |
| S4 + attention | 0.82 | 3.94 | 62.18 |
| Neck | 1.91 | 8.92 | 241.24 |
| 检测头 | 1.38 | 3.47 | 195.86 |
| 分割头 | 4.39 | 4.92 | 632.14 |

阶段之间人为同步会改变 GPU 调度与内存池行为，不能精确相加或视为设备 kernel 时间。该表用于找优先级。CPU 基线首轮与 GPU 测量有短暂重叠，后续 oneDNN 线程试验串行运行。

ConvNormAct 子操作插桩（不是全模型所有算子）：CUDA conv 9.22ms / LN 3.38ms / SiLU 1.49ms；CPU conv 1711ms / LN 110ms / SiLU 81ms。因此 CPU 只改 LN 不可能解决秒级问题。CPU oneDNN 1/4/8 线程无 verbose 结果为 2034/1276/1255ms；oneDNN verbose 已确认 OpenMP nthr=8，不能再把线程数作为纯猜测。

## Vulkan：两套放置的意义与去留

当前只有一套计算实现，ZNN_VK_LOCAL 是分配策略开关：默认小于 64MB 使用主机可见系统内存；开启后优先 DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT。它不是两个模型，也不是两套算法。

**受控 512³ GEMM 对照**（操作数提前上传，预热，计算与输出读取分开）：

| 放置 | 计算+标量同步 ms | 完整输出回读 ms |
|---|---:|---:|
| 默认 | 0.462 | 1.629 |
| LOCAL | 0.137 | 8.658 |

LOCAL 计算更快，回读更慢。“LOCAL 使 GEMM 计算回退 22 倍”不成立于本次测量。历史绝对比例尚未原样复现，但其将计算和传输混在一起的结论不能继续支撑产品设计。

源码 copy_device_to_host/off 在 flush 后对 mapped 指针直接 memcpy，没有专用 HOST_CACHED 回读池或 GPU staging copy。对本机 LOCAL 映射输出执行 CPU 读取形成严重瓶颈；不能据此把所有中间张量重新放回系统内存。

目标应是一套自动内存策略：常驻权重/中间结果优先设备本地内存；上传池与回读池按用途独立选型；回读选择适合 CPU 读取的内存，显式 GPU copy 与正确同步；集显/统一内存设备按能力选择共享路径。支持没有可映射本地堆的设备，并按 memoryTypeBits、预算和分配失败正确回退。开关短期仅作为 A/B 排错项，修复并跨设备验证后，不再要求应用设置 ZNN_VK_LOCAL。

另外 getenv 判断的是“存在”，当前 ZNN_VK_LOCAL=0 也会开启；诊断关闭必须取消该环境变量。

参考：[Vulkan 内存分配指南](https://docs.vulkan.org/guide/latest/memory_allocation.html)、[同步示例](https://docs.vulkan.org/guide/latest/synchronization_examples.html)。

## 下一步通用优化顺序

| 优先级 | 工作 | 通用性与验收 |
|---|---|---|
| P0 | 修复 AP、benchmark 死循环，建立独立前向/传输/后处理统计 | 记录硬件、DLL hash、权重、真实输入、预热、P50/P95/P99；端到端目标独立于纯前向 |
| P0 | 修复分段训练的采样和优化器恢复 | 数据覆盖日志；同一检查点恢复与不中断训练对照 |
| P1 | ZNN 统一计算/上传/回读内存职责，先修 Vulkan 回读 | 不改模型算术；测试 D2H/H2D/D2D、偏移视图、空张量、生命周期、连续推理及无 BAR 回退 |
| P1 | CPU 依赖部署可观测化，补分组/深度卷积快速路径 | 现 oneDNN conv2d 显式拒绝 groups!=1；回落 Eigen 每组 im2col，1x1 也材料化。先处理实际热点，不只改 LN |
| P2 | 各后端通用 1x1、depthwise、常见卷积布局/缓存优化 | 按算子形状和设备能力选择，不匹配 UniFP 名称；CPU/CUDA/VK/Metal 同语义测试 |
| P2 | 通道 LN 连续空间分块、向量化及归约优化 | 保留任意 dim/视图/数值正确性；可优先 LN+affine+SiLU 融合 |
| P2 | 统一执行计划、内存复用与常量驻留 | 降低每次分配、主机上传常量、拷贝及无必要同步；检查所有后端主机读写边界 |
| P3 | 后端执行复用与安全融合 | CUDA Graph；VK 命令复用与资源依赖屏障；Metal 批提交。Metal 目前仍有逐次 commit+waitUntilCompleted，需要在 Mac 验证 |
| P3 | 原生低精度存储和预打包权重 | 避免每次 GEMM 临时转换；每后端能力检测、误差及完整精度验证，不强制所有后端同一内核 |

“全后端”是共同 API、相同数值语义、通用策略及后端各自实现，不代表 CPU 要使用 Vulkan 的内存机制。当前未运行 Metal，不能标记全后端性能验收通过。

注意 ChannelNorm 是依赖当前输入的 LayerNorm，不能像推理 BatchNorm 一样静态折叠进卷积权重。所谓 Conv+LN+SiLU 融合需实际运行时融合与数值验证。CUDA Graph 降低的是启动开销，不会消除卷积计算；参见 [CUDA Graph 官方指南](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)。目前 CUDA 16.8ms 到 1.7ms 的十倍目标、CPU 640 达 30fps 均不能保证。

## 复现入口

```powershell
python build/perf_audit_20261026/generate.py
cmake -S build/perf_audit_20261026 -B build/perf_audit_20261026/out -G 'Visual Studio 16 2019' -A x64
cmake --build build/perf_audit_20261026/out --config Release
$env:OMP_NUM_THREADS='8'
$env:ZNN_STRICT_NATIVE='1'
$env:ZNN_VK_LOCAL='1'
./build/perf_audit_20261026/out/Release/perf_audit.exe vk 640 ../UniFP/runs/ftk/abs_12000.ufp 30 brief ../UniFP/data/P3M/val_P/images/p_0074c250.jpg
# 默认放置对照：Remove-Item Env:ZNN_VK_LOCAL
# CPU oneDNN：将已验证的 build/Release/dnnl.dll 所在目录加入当前进程 PATH。
# 不传 brief 与图片参数时，执行随机输入及真实模型阶段/ConvNormAct 插桩。
```

生产代码和交付 DLL 没有被本次诊断替换。后续应先把可重复的观测能力整合进 ZNN 通用诊断工具，再按以上顺序逐项修改与回归，而不是一次引入多种优化后无法归因。
