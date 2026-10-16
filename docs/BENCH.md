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

## 2026-10-16 · GPU cuBLAS: FP32 vs TF32 Tensor Core(matmul)

环境: RTX 4070 Ti SUPER(Ada, sm_89) / CUDA 13.4 / 同一二进制。
公平性: 数据常驻 GPU(GPU 建张量, 零 PCIe 往返),
`Runtime::set_precision(FP32/TF32)` 切档, to_device(CPU) 同步计时。

| 矩阵 | FP32(sgemm) | TF32(gemmEx FAST_TF32) | 加速比 |
|---|---:|---:|---:|
| 512³  | 5.00 TF/s  | 14.39 TF/s  | 2.9× |
| 1024³ | 5.07 TF/s  | 7.96  TF/s  | 1.6× |
| 2048³ | 11.02 TF/s | 18.87 TF/s  | 1.7× |
| 4096³ | 14.89 TF/s | 28.68 TF/s  | **1.93×** |

- FP32 峰值 ~15 TF/s ≈ 该卡 FP32 标称算力, cuBLAS sgemm 已打满;
  TF32 经 cublasGemmEx(CUBLAS_COMPUTE_32F_FAST_TF32)启用 Tensor Core。
- 512³ 加速比最高但绝对吞吐低(占用不足); 1024³ 在过渡区。
- 对拍注意: TF32 尾数 19 位, 与 FP32 结果差 ~1e-3,
  数值对拍须 `set_precision(FP32)` 后再比对(测试已如此处理)。

## 已知瓶颈(后续优化目标)

1. TF32 目前仅 matmul 内核; linear/conv2d/attention 仍走 sgemm,
   待统一到同一 gemmEx 入口(收益同类);
2. conv2d im2col 有一次显式拷贝(col_buf), 可用 stride-map 直接 GEMM 消除;
3. autograd 节点为裸 new, 未引入节点池(DESIGN §13.3 规划项);
4. FP16/BF16 混合精度未接入(Ada 卡 Tensor Core FP16 峰值再翻倍)。
