# 性能基准记录 (BENCH)

环境: Windows x64, MSVC 19.29 (VS2019), Release 构建。基准工具:
`tests/bench.cpp`（`znn_bench`）。对照方式: 同一二进制, `ZNN_FORCE_REF=1`
强制朴素参考内核 vs 默认(快速内核槽位), 排除编译优化差异的公平对比。

## 2026-10-15 · Eigen 引入前后的内核级对比

| 基准 | 朴素 ref | Eigen 内核(kernels[CPU]) | 加速比 |
|---|---:|---:|---:|
| matmul 128×128×128 | 0.74 ms (5.7 GF/s) | 0.11 ms (39.7 GF/s) | **7.0×** |
| matmul 512×512×512 | 57.2 ms (4.7 GF/s) | 6.1 ms (44.3 GF/s) | **9.4×** |
| matmul 1024×1024×1024 | 1185.6 ms (1.8 GF/s) | 45.7 ms (47.0 GF/s) | **25.9×** |
| conv2d 8×16×32×32 × 32×3×3(pad 1) | 288.9 ms (0.26 GF/s) | 5.2 ms (14.5 GF/s) | **55.4×** |
| MLP 784-256-10 训练步(B=64, fwd+bwd+step) | 38.4 ms (36.0 s/epoch) | 15.1 ms (**14.2 s/epoch**) | 2.5× |
| MLP 784-256-10 推理步(B=64) | 10.1 ms | 5.0 ms | 2.0× |

## MNIST 端到端(5 epochs 全流程含加载与逐 epoch 评估)

| 版本 | 5 epoch 总耗时 | 最终测试集准确率 |
|---|---:|---:|
| 朴素内核(Debug) | ~12 min | 97.19% |
| Eigen 内核(Release) | **152 s** | 97.09% |

准确率差异 0.1% 来自 GEMM 浮点求和顺序不同, 属正常数值噪声;
两条路径均满足 >97% 验收标准。

## 数值一致性验证

- matmul Eigen vs ref: 17×23 随机矩阵, 最大绝对差 < 1e-3 ✓
- conv2d Eigen vs ref: 3×4×9×9 随机输入, 最大绝对差 < 1e-3 ✓
- 全部 40+ 单元/梯度/有限差分/收敛测试两路径均通过 ✓

## 2026-10-16 · OpenMP 多线程 GEMM(在 Eigen 基础上)

| 基准 | 单线程 Eigen | OpenMP 多线程 | 提升 |
|---|---:|---:|---:|
| matmul 512³ | 6.1 ms (44.3 GF/s) | 1.45 ms (**184.7 GF/s**) | 4.2× |
| matmul 1024³ | 45.7 ms (47.0 GF/s) | 11.0 ms (**195.4 GF/s**) | 4.2× |
| conv2d 8×16×32×32 | 5.2 ms | 4.0 ms | 1.3× |
| MNIST 5 epoch | 152 s | 249 s* | — |

\* MNIST 全程反而变慢: MLP 的 GEMM 较小(64×784×256 等), OpenMP 线程
同步开销超过收益, 且逐元素/规约仍是单线程。教训: 小矩阵场景线程池
需要任务粒度控制(v1.1 记录; 大矩阵基准 4.2× 是真实收益)。
准确率 97.29%(数值噪声范围内, >97% ✓)。

## 2026-10-16 · GPU cuBLAS GEMM: 四算子统一 + TF32 真实收益(勘误)

环境: RTX 4070 Ti SUPER(Ada, sm_89) / CUDA 13.4。
内核级数据: 独立探针(纯 cuBLAS 循环, 预分配, 10 次迭代)——
app 级基准含每次调用的输出张量 cudaMalloc/cudaFree(~64MB @4096³
≈ +5.9ms/次), 会稀释 GEMM 真实吞吐, 故以探针数据为准。

**matmul 4096³ 内核级(勘误前条目撤销)**:

| 路径 | TF/s |
|---|---:|
| cublasSgemm 纯 FP32 | 32.2(卡 FP32 峰值的 ~73%) |
| sgemm + SetMathMode(TF32) | 41.1 |
| **cublasGemmEx FAST_TF32(现行)** | **42.4** |

**TF32 真实收益 = 1.32×(32.2 → 42.4)**, 非此前记录的 1.93×。
此前数字是双重假象: (1) 提交 2562ff6 的 typedef 多传了 C/D 三参数,
gemmEx 从未真正执行, 当时测到的是陈旧二进制里 sgemm+残留
SetMathMode 的效果; (2) app 级基准含分配开销。两处均已修正。

**GEMM 类四算子(app 级, 数据常驻, 含分配开销)**:

| 算子 | 朴素内核 | cuBLAS 后 | 加速比 |
|---|---:|---:|---:|
| linear 4096×2048×2048+bias | 0.34 TF/s | 16.13 TF/s | **48×** |
| conv2d 8×16×32×32→32×3×3 | 0.83 TF/s | 0.17 TF/s* | — |
| attention 4×8×256×64 | 0.70 TF/s | 0.70 TF/s* | — |

\* conv2d/attention 小负载下计算仅 ~50µs, 每次调用的 scratch
im2col/P 张量分配(cudaMalloc+cudaFree 含隐式同步)占主导——
**GPU 缓存分配器是下一个必须项**(机制变更, 待用户审核)。

数值正确性: linear/conv2d/attention 新增 CUDA↔CPU 对拍测试
(L=48/D=40 非方阵锁死回归); 顺带发现并修复 CPU attention_ref
存量 bug(gemm 辅助 at 标志只转置 A, L≠D 时 K 按 [D,L] 误读,
L==D 时两种布局巧合相同故从未暴露)。

## 2026-10-16 · GPU 异步分配器: cudaMallocAsync 消除分配开销

改动: cuda_device.cu 中 alloc/free 改为 cudaMallocAsync/cudaFreeAsync
(默认流, 流有序免隐式同步), available() 一次设置默认池释放阈值上限
(空闲块留池复用)。约 15 行, 零新增全局状态。

**端到端 gpu_bench 对比(改前 → 改后, 同一基准)**:

| 基准 | 改前 | 改后 | 提升 |
|---|---:|---:|---:|
| matmul 1024³ TF32 | 5.88 TF/s | 21.90 TF/s | 3.7× |
| matmul 2048³ TF32 | 13.11 TF/s | 24.97 TF/s | 1.9× |
| matmul 4096³ TF32 | 18.80 TF/s | 24.74 TF/s | 1.3× |
| matmul 4096³ FP32 | 15.99 TF/s | 20.13 TF/s | 1.3× |
| linear 4096×2048×2048+bias | 16.13 TF/s | **31.58 TF/s** | 2.0× |
| conv2d 8×16×32×32→32×3×3 | 0.447 ms | **0.051 ms** | 8.8× |
| attention 4×8×256×64 | 0.770 ms | **0.270 ms** | 2.9× |

- 端到端 TF32/FP32 比恢复到 ~1.3×, 与内核级探针(42.4/32.2)一致,
  证实此前差距确为分配稀释;
- 4096³ 端到端(24.7)仍低于纯内核(42.4): 余量为基准间时钟状态
  与 d2h 同步拷贝, 非分配问题;
- conv2d 首次超过旧朴素内核(0.091ms); attention 端到端 2.9×。

## 2026-10-16 · GPU FP16 混合精度(计算转半精度, 存储/累加仍 FP32)

`Runtime::set_precision(FP16)`: gemm 原语入口把 A/B 转 FP16 scratch
(带步长转换内核, 列切片/权重布局均覆盖), cublasGemmEx(16F, 16F, 32F,
COMPUTE_32F); 输出仍写 FP32。训练配 GradScaler(动态损失缩放)防下溢。

| matmul(端到端) | FP32 | TF32 | FP16 |
|---|---:|---:|---:|
| 1024³ | 16.5 | 20.5 | 23.8 TF/s |
| 2048³ | 18.7 | 24.7 | 28.6 TF/s |
| 4096³ | 19.8 | 24.8 | **30.9 TF/s** |

FP16/TF32 = 1.25×(端到端含 2 个转换内核的读写与 scratch 分配;
纯 GEMM 内核收益更高)。FP16 数值: matmul/linear 对拍 2% 相对容差通过;
GPU FP16+GradScaler 训练 60 步: 损耗下降, 有限性达标。

## 2026-10-16 · oneDNN 探针(委托决策数据, 未实施)

环境: oneDNN v3.13.4 源码构建(CPU-only, OpenMP/VCOMP140),
i5-14600KF; 探针含数值抽查(c[0]/y[0] 与手算一致)。
GitHub 现行版本不再发 Windows 预编译包(仅 v1.7/2021 太旧),
pip 的 onednn 轮子是空壳(仅文档)——部署需源码构建或随包分发 DLL。

| 负载 | znn 现状(Eigen+OMP) | oneDNN | 加速比 |
|---|---:|---:|---:|
| matmul 512³ | 184.7 GF/s | 494.3 GF/s | **2.7×** |
| matmul 1024³ | 195.4 GF/s | 753.5 GF/s | **3.9×** |
| matmul 2048³ | — | 902.3 GF/s | — |
| conv2d 8×16×32×32→32×3×3 | 17.7 GF/s | 357.0 GF/s | **20×** |

实施路径(已批准方向): dynlib 加载 dnnl.dll(缺失回落 Eigen) +
matmul/linear 走 dnnl_sgemm(一期) + conv2d 走 primitive API(二期)。

## 2026-10-18 · oneDNN 全算子委托实施(dnnl.dll 动态加载, 缺失回落 Eigen/ref)

覆盖: 13 逐元素(add/sub/mul/div/neg/exp/log/sqrt/relu/gelu/sigmoid/
tanh/silu)+ softmax(任意维)+ sum/mean(≥1M 元素, 小规约 ref 更快)+
matmul/linear(dnnl_sgemm)+ conv2d + max/avg_pool + layer_norm +
batch_norm(推理)。ZNN_NO_ONEDNN="family,..." 可按族禁用(A/B 排查)。
引擎/流一次性缓存(小算子高频调用的开销大头)。

| 基准 | Eigen(禁用 oneDNN) | oneDNN | 加速比 |
|---|---:|---:|---:|
| matmul 128³ | 96 GF/s(OMP) | 72-97 GF/s | ~1×(小矩阵无益) |
| matmul 512³ | 185 GF/s | **403 GF/s** | 2.2× |
| matmul 1024³ | 195 GF/s | **686 GF/s** | 3.5× |
| conv2d 8×16×32×32 | 5.2 ms | **0.97 ms** | 5.4× |
| MLP 训练步(B=64) | 13.2-14.8 ms | **8.5 ms** | 1.55× |
| MLP 推理步 | 5.0-6.1 ms | **0.11 ms** | 45× |
| MNIST 5 epoch | 83 s | **42 s** | 2.0×(97.33%) |

排查记录(性能诊断方法论): 首版 MLP 训练反而慢 3.6× —— 逐层分段计时
(fwd/bwd/step)+ 逐算子微基准(µs 级)+ 按族禁用二分定位:
1. linear_eigen.cpp 存在两个同名实现, oneDNN 钩子接在未注册的死代码上
   (带 bias 的 Eigen 复合表达式路径 25ms/次 —— 合并后 83µs, 300×)
2. dnnl_sgemm 是行主序约定(文档明示, 异于 cuBLAS 列主序), 转置标志
   位置曾放错; NT 场景 trans_b='T'
3. 卷积原语第二参数是 DNNL_ARG_WEIGHTS(非 SRC_1), 传错则 query_md
   返回空 → 静默回落(表面"慢", 实为没走 oneDNN)
4. sum_into 通用路径每元素做步长算术(慢 20×) —— 补连续首/末维快速路径
   (受益于所有后端的 CPU 回落)
5. 小规约(<1M 元素)oneDNN 线程划分不合算 → 回落 ref

## 2026-10-19 · Vulkan 计算后端(RTX 4070 Ti SUPER 实测)

环境: Vulkan 1.4.341 驱动 / 本机 NVIDIA GPU / 手写 GLSL 着色器。
双内核按规模分发(同 cuBLAS): ≤~50 GFLOP 用 32×32 分块(占用优先),
大矩阵用 64×64 分块+双缓冲预取(B 装载摊销翻倍)。双内存类型:
≥64MB 分配走 BAR 显存(GPU 直读 VRAM), 小分配走系统内存
(实测更快: CPU 写缓存 + L2 吸收读; 池按类型分桶, 256MB 上限)。

| matmul | 峰值 | 稳态(多轮冷却均值) | CPU oneDNN |
|---|---:|---:|---:|
| 512³ | 348 GF/s | ~300(受 ~0.6ms/次固定开销限制) | ~200 |
| 1024³ | 612 GF/s | ~575 | 686 |
| 2048³ | **1365 GF/s** | ~1320(**CPU 的 1.9×**) | 686 |
| 4096³ | **1451 GF/s**(冷机) | ~600(热节流; 旧 32×32 内核仅 487 崩塌值) | — |

- 逐元素 16K 元素对拍 0 误差(13 算子共用一着色器);
  sum/mean/softmax 对拍通过; 18 算子原生, 其余透明回落 CPU
- 512³ 固定开销构成: 命令缓冲分配+submit+waitIdle(全同步往返,
  同步 Tensor API 的架构代价, 与 CUDA 路径的同步拷贝同性质)
- 4096³ 热节流说明: 桌面卡与 Windows 合成器共享, 持续负载降频;
  冷机首跑 1451 证明内核本身达到 1.45 TF/s

## 已知瓶颈(后续优化目标)

1. FP16 已接入; **GPU 反向路径存量缺口(部分)**: pools/embedding/
   permute/mean-sum 的梯度规则仍为宿主循环, GPU 训练触及即崩
   (FP16 块已修 full/ones/mse/unary/matmul视图/优化器clear/conv2d 梯度
   接线; 剩余待原生内核化——下块工作);
2. h2d/d2h 仍用同步 cudaMemcpy(小张量无所谓, 流水线场景需
   异步拷贝 + pinned memory);
3. conv2d im2col 有一次显式拷贝, 可用 stride-map 直接 GEMM 消除;
4. autograd 节点为裸 new, 未引入节点池(DESIGN §13.3 规划项)。

---

## 三后端全算子吞吐表(2026-10-21, Vulkan 构建; CUDA/Metal 各自构建跑同表)

| 算子 | 形状 | CPU | Vulkan | CUDA | Metal |
|---|---|---|---|---|---|
| matmul | 128³ | 32 GF/s | **60 GF/s** | — | — |
| matmul | 512³ | 151 GF/s | **372 GF/s** | — | — |
| matmul | 1024³ | 122 GF/s | **824 GF/s** | — | — |
| linear | 256×512×256 | 95 GF/s | **352 GF/s** | — | — |
| conv2d | 4×3×32×32→16 | 0.44 ms | **0.08 ms (5.6×)** | — | — |
| attention | 1×4×128×64 | 2.14 ms | 1.95 ms | — | — |
| add(bw) | 4M elems | 8.7 GB/s | **14.1 GB/s** | — | — |
| softmax | 256×1024 | 0.97 ms | 1.38 ms | — | — |
| embedding | 8×128 | ~0 ms | 0.05 ms | — | — |
| layer_norm | 64×512 | **0.03 ms** | 0.10 ms | — | — |

**观察**: Vulkan matmul 512³+ 达 CPU 2.5-6.7×(oneDNN CPU vs 直接着色器 GPU);
conv2d 5.6×(直接内核 vs im2col+GEMM CPU); 逐元素受同步往返限制(3.6ms
固定开销); softmax/LN 小形状受每派发 0.6ms 提交开销主导; embedding/
LN 小于 0.1ms 属可接受. Metal 端由用户 Mac 构建后跑同 zbench 补全.

### Metal 补全(Mac mini AppleClang 21 / arm64, 用户实跑 2026-10-21)

| 算子 | 形状 | CPU(M1) | Vulkan(4070TiS) | Metal(M1) | Metal/CPU |
|---|---|---|---|---|---|
| matmul | 128³ | 38 GF/s | 60 GF/s | 19 GF/s | 0.5× |
| matmul | 512³ | 74 GF/s | 372 GF/s | 158 GF/s | **2.1×** |
| matmul | 1024³ | 97 GF/s | 824 GF/s | 276 GF/s | **2.8×** |
| linear | 256×512×256 | 95 GF/s | 352 GF/s | 139 GF/s | 1.5× |
| conv2d | 4×3×32×32→16 | 0.25 ms | 0.08 ms | 0.26 ms | ~1× |
| attention | 1×4×128×64 | 2.6 ms | 2.0 ms | 3.5 ms | 0.7× |
| add(bw) | 4M elems | 17.6 GB/s | 14.1 GB/s | **43.0 GB/s** | **2.4×** |
| softmax | 256×1024 | 0.43 ms | 1.38 ms | 0.52 ms | ~1× |
| embedding | 8×128 | ~0 ms | 0.05 ms | 0.18 ms | — |
| layer_norm | 64×512 | 0.04 ms | 0.10 ms | 0.37 ms | 慢 |

**Metal 观察**:
- 大 GEMM(512³+)充分发挥 M1 GPU: 276 GF/s = CPU 2.8×(统一内存带宽优势)
- 逐元素带宽 43 GB/s 遥遥领先(共享内存零拷贝, vs Vulkan 14 GB/s 跨 PCIe)
- 小形状(LN 0.37ms / embedding 0.18ms)受每内核 dispatchThreads+waitUntil
  同步往返主导(M1 ~0.1ms/次, 积小成多)
- conv2d/attention 与 CPU 持平(直接法内核, 未做分块/寄存器优化)
- CPU 侧 matmul 74-97 GF/s 为 oneDNN 无 OpenMP 构建下的单线程性能

---

## 三后端全量对比表(2026-10-21 最终版, 含用户 perf 更新后的代码)

| 算子 | 形状 | CPU | Vulkan(4070TiS) | CUDA(4070TiS) | Metal(M1) |
|---|---|---|---|---|---|
| matmul | 128³ | 40 GF/s | 59 GF/s | **428 GF/s** | 19 GF/s |
| matmul | 512³ | 196 GF/s | 369 GF/s | **8441 GF/s** | 158 GF/s |
| matmul | 1024³ | 178 GF/s | 849 GF/s | **21158 GF/s** | 276 GF/s |
| linear | 256×512×256 | 161 GF/s | 349 GF/s | **2969 GF/s** | 139 GF/s |
| conv2d | 4×3×32×32→16 | 0.49 ms | 0.06 ms | **0.03 ms** | 0.26 ms |
| attention | 1×4×128×64 | 0.80 ms | 1.93 ms | **0.12 ms** | 3.51 ms |
| add(bw) | 4M elems | 9.5 GB/s | 14.1 GB/s | **1281 GB/s** | 43.0 GB/s |
| softmax | 256×1024 | 0.94 ms | 1.29 ms | **0.015 ms** | 0.52 ms |
| embedding | 8×128 | ~0 ms | 0.05 ms | **0.015 ms** | 0.18 ms |
| layer_norm | 64×512 | 0.03 ms | 0.056 ms | **0.016 ms** | 0.37 ms |

### 后端特征总结

**CUDA(4070 Ti SUPER 16GB)**: 全算子碾压级优势——matmul 1024³ 21.2 TF/s
= Vulkan 25× = CPU 119×; 逐元素带宽 1281 GB/s(显存直访); 小算子
0.015ms 级(kernel launch 开销下限). **训练首选**.

**Vulkan(同卡)**: 逐元素带宽 14 GB/s —— 跨 PCIe 瓶颈(HOST_VISIBLE
系统内存映射, 非显存直址); matmul 849 GF/s = CUDA 的 1/25(朴素
32×32 分块 vs cuBLAS 高度调优). **跨平台推理可用**.

**Metal(M1)**: 逐元素 43 GB/s 统一内存带宽最强(零拷贝); 大 GEMM
276 GF/s = M1 CPU 2.8×(GPU 均衡架构); 小算子受 ~0.1ms/次同步往返.
**Mac 端推理首选**.

**CPU(oneDNN)**: matmul 178-196 GF/s; 逐元素 9.5 GB/s(单线程 Eigen).
**兜底基线**.

## Vulkan 内存放置开关(ZNN_VK_LOCAL, 2026-10-26)

| 负载 | 默认(sysRAM<64MB) | ZNN_VK_LOCAL=1(全 BAR) |
|---|---|---|
| UniFP CNN @320(多小算子) | 71.7 ms | **19.5 ms(3.7×, 51fps)** |
| UniFP CNN @640 | 336.9 ms | **64.2 ms(5.2×)** |
| matmul 512³(流式 GEMM) | **1.50 ms** | 33.4 ms(回退 22×) |
| matmul 1024³ | **11.5 ms** | 13.9 ms |

结论:放置最优解按访问模式分化(CNN 逐算子往返 vs GEMM 流式),默认
维持原策略;CNN/推理类负载设 ZNN_VK_LOCAL=1。ztestkit 339 两种放置均绿。
