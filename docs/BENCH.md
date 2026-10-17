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

## 已知瓶颈(后续优化目标)

1. FP16/BF16 未接入(Ada Tensor Core 峰值再翻倍);
2. h2d/d2h 仍用同步 cudaMemcpy(小张量无所谓, 流水线场景需
   异步拷贝 + pinned memory);
3. conv2d im2col 有一次显式拷贝, 可用 stride-map 直接 GEMM 消除;
4. autograd 节点为裸 new, 未引入节点池(DESIGN §13.3 规划项)。
