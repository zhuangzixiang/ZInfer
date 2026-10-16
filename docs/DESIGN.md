# ZInfer (znn) 设计文档 — v0.1 讨论稿

> 状态: 架构已商定，接口待多轮讨论定稿（见 §12 与文末决策点）
> 日期: 2026-10-15

## ⛔ 铁律

1. **不允许使用过渡方案**。任何"暂时先这样、以后再改"的设计必须先经用户审核同意才能实施。
2. **每种后端必须走该后端的原生内核**。CUDA 张量的计算走 CUDA 内核，后续 Metal/Vulkan 同理。**不允许 CPU 桥接/暂存**——即不允许"拷到 CPU 算完再拷回 GPU"的路径存在于任何正式代码中。

---

## 1. 定位与目标

一个**小型、可扩展、训练+推理同核**的 C++ ML/DL 库，未来以"共用内核 + 平行扩展包"方式支持 LLM。

**目标：**

- ffmpeg 式扩展机制：新增算子只需"算子文件 + operator_list 一行 + 构建脚本"，不碰内核。
- 多后端: CPU（参考实现 + Eigen 快速版）→ CUDA → Metal → Vulkan，单注册表 + 后端槽位。
- 训练与推理共用一个内核（autograd 引擎算子无关，推理走 no_grad 跳过 tape）。
- 性能策略 = 委托现成库（Eigen/oneDNN/cuBLASLt/MPS），不手搓内核堆量。
- 正确性优先：每个算子带朴素参考实现，作为所有后端的对拍基准。

**非目标（v1）：**

- 不追平 SOTA 性能（优化路线见 §6，靠换委托库而非手写内核）。
- 不做分布式训练、图编译大框架、Python 绑定（远期可选）。
- 不做 NPU/DSP 后端（架构预留第五后端扩展位）。

---

## 2. 总体架构

```
┌───────────────────────────────────────────────────┐
│ zllm (远期平行扩展包):                              │
│   KV cache 管理 / paged attention 算子 / sampler    │
│   GGUF·safetensors 加载 / 自回归执行循环             │
│   —— 只 import znn, 零改动内核                      │
├───────────────────────────────────────────────────┤
│ znn/nn: Module / Sequential / Linear / Conv2d ...  │
│   —— 用户按论文逐层搭网络的 API 层                   │
├───────────────────────────────────────────────────┤
│ znn/core (内核, 稳定不动, ~2,300 行):                │
│   Tensor │ Device/Allocator │ Operator+注册表          │
│   AutogradEngine(算子无关) │ Optimizer │ NoGrad     │
├───────────────────────────────────────────────────┤
│ 可插拔算子层: ops/*.hpp + *.cpp (+ *.cu/.mm/.comp)  │
│ 注册汇总: reg/operator_list.cpp (≈ ffmpeg codec_list)│
└───────────────────────────────────────────────────┘
```

核心性质：**加算子/加内核/加融合规则，永远不碰 core**。dispatch、autograd、优化器全部算子无关。

### 单注册表模型（已商定）

每个算子一条 `Operator` 注册项，内含各后端内核槽位：

```cpp
struct Operator {
    const char* name;
    ShapeInferenceFunction infer;                        // 形状推导
    GradientFunction     grad;                         // 反向规则; nullptr = 仅推理
    KernelFunction    kernels[Device::COUNT];       // 后端槽位, 可空 → 兜底链
    KernelFunction    ref;                          // 朴素参考实现(CPU), 兜底+对拍基准
};
```

- 算子语义（形状/求导/默认保存什么）与内核实现集中在一个注册项里，**新增算子只改一处**。
- 槽位为空 → 兜底链：`该设备内核 → 透明拷贝到 CPU 跑 kernels[CPU] → ref`（可配置为直接报错）。
- "CPU 性能强化版"（oneDNN）= 替换 `kernels[CPU]`，不是新槽位。
- 新增第五种后端 = Device 枚举加一项 + 写后端骨架 + 给关心的算子填槽；旧算子空槽走兜底，**无 N×M 爆炸**。

物理现实：内核代码因编译器原因分属不同翻译单元（`.cu`/`.mm`/`.comp`），但注册项统一引用它们——逻辑上仍是一处。

---

## 3. 目录结构

**对外只有 2 个头文件**（结构定义 + 接口定义；内部实现头一律不暴露，因此内部可自由重构、Eigen 永不进用户编译单元）：

```
ZInfer/
├── include/                # ★ 对外全部: 2 个文件, 固定不变
│   ├── znn_defines.hpp     # ① 结构定义: 枚举/Shape/Options/Param
│   └── znn.hpp             # ② 接口定义: Tensor/Activator/Loss/Layer/Optimizer
├── src/                    # 内部, 对用户不可见
│   ├── operator.hpp          # 算子契约(算子作者唯一要读): Operator/内核签名/注册宏
│   ├── znn_internal.hpp/.cpp  # znn.hpp 的实现支撑: Storage/Tensor::Impl/Shape/Runtime
│   ├── autograd.hpp/.cpp   # 自动求导引擎: GraphNode/建图(record)/反向(backward)
│   ├── op_dispatcher.hpp/.cpp # 算子派发器: 执行一次算子调用的完整链路(注册数据在 operator_list)
│   ├── operators/          # 一算子一文件, 文件名=算子名; 异构后端=同名异后缀
│   │   ├── operator_list.cpp   # ← 注册数据(extern + 列表, ≈ ffmpeg codec_list.c)
│   │   └── add.cpp  conv2d.cpp  (conv2d.cu/.mm/.comp 里程碑 2+)
│   ├── layers/             # 一层一文件(层无后端维度, 设备选择在算子派发)
│   │   ├── layers.hpp  layer_factory.cpp
│   │   └── linear.cpp conv2d.cpp batch_norm2d.cpp ... activation.cpp pool2d.cpp
│   ├── optim/              # 一个优化器一文件(同样无后端维度)
│   │   ├── optim.hpp  optimizer_factory.cpp
│   │   └── sgd.cpp adam.cpp adamw.cpp
│   └── backends/           # cuda/ metal/ vulkan/ (里程碑 2+)
├── tests/                  # 对拍测试: fast vs ref; GPU vs CPU golden
├── examples/mnist/         # 里程碑 1 验收
└── docs/
```

**一次算子调用的完整链路**（各文件在链路中的位置）：

```
Tensor::add(a, b)                          operators/add.cpp(前端)
  └─ OpDispatcher::run(&g_add_operator, ...)     op_dispatcher.cpp(派发)
       ├─ g_add_operator.infer(...)             → 输出形状
       ├─ 分配输出 Tensor                   znn_internal(Tensor::Impl/Storage)
       ├─ g_add_operator.kernels[设备](...)      operators/add.cpp(内核; 空槽→兜底链)
       └─ AutogradEngine::record(...)       autograd.cpp(训练时建 tape 节点)

loss.backward()
  └─ AutogradEngine::backward(loss)         autograd.cpp: 逆拓扑执行各 g_*_def.grad
       └─ 广播梯度规约 BackwardContext::reduce_to
```

**文件划分判定规则：按编译器分文件、按逻辑聚目录。** 同编译器的实现（CPU ref 版与 Eigen 版）留在同一文件的不同内核槽位；异构后端（.cu/.mm/.comp）为同目录同名异后缀文件，注册项 extern 引用。层与优化器没有后端维度——它们是"参数 + 调用算子"的组合，设备选择发生在算子派发。职责边界：注册**数据**在 `operators/operator_list.cpp`，派发**机制**在 `op_dispatcher`；算子契约在 `operator`；求导引擎在 `autograd`；`znn_internal` 只是公共 API 的实现支撑，不含任何引擎。include 一律不带相对路径（CMake 以 `src/` 为私有包含目录）。用户接入方式：`#include "znn.hpp"` + 链接 `libznn`。

CMake 按平台/开关条件启用后端：`ZNN_CUDA / ZNN_METAL / ZNN_VULKAN / ZNN_ONEDNN`，`ZNN_MINIMAL` 可裁掉 Eigen 只留参考实现。可选项：CMake 扫描 `src/ops/` 自动生成 `operator_list.cpp`（对应 ffmpeg configure 生成 codec_list.c 的做法），实现"丢文件即扩展"。

---

## 4. 核心模块设计

### 4.1 Tensor

- 引用计数存储，shape/stride 支持零拷贝视图（view/slice/transpose/permute/reshape）。
- dtype: `F32, F16, BF16, I32, I64, U8`（v1 实际启用 F32 + I64 索引；半精度枚举预留）。
- autograd 元数据：`requires_grad`、`grad`、`grad_fn` 节点（推理模式下不创建）。
- 数据读取原语 data<T>()(当前设备指针, 同 PyTorch data_ptr 约定)/item<T>()(单值)/strides()/is_contiguous()；跨设备搬运入口 to_device()。

### 4.2 Device / Allocator

- `enum class Device : uint8_t { CPU, CUDA, Metal, Vulkan }`（+COUNT）。
- 每后端一个分配器；CPU 默认 malloc + 可选 arena（推理内存池在 §6 的 v1.5 引入，接口预留）。
- 后端骨架职责：设备初始化、host↔device 拷贝、GEMM 委托对象、（预留）stream/event。

### 4.3 Dispatch 与兜底链

- 启动期由 `operator_list.cpp` 构建 `name → Operator*` 表与 id 映射；运行时 `id → 函数指针` 直接调用，零字符串查找。
- 调用路径：`znn::relu(x)`（自由函数，ops/relu.hpp 声明）→ 取 Operator → 按 `x.device()` 选槽 → 空槽走兜底链 → autograd 包装（grad 规则存在且 requires_grad 时建 tape 节点）。
- 默认保存前向输入+输出供 backward（v1 简单起步；后续可按算子声明精简保存列表）。

### 4.4 Autograd 引擎（~500 行，算子无关）

- 反向模式：标量 `loss.backward()` 触发，按 tape 拓扑逆序执行各 Operator 的 `GradientFunction`。
- `GradientFunction(BackwardContext&)`：context 提供 saved inputs/outputs、grad_output，写入 grad_input。广播的梯度自动规约（学 PyTorch unbroadcast）。
- `Runtime::set_grad_enabled(false)` 跳过 tape 创建（旧 NoGrad 的函数形式）。
- 自定义算子不写 grad → 自动标记仅推理，训练时报清晰错误。

### 4.5 Optimizer

- `Optimizer` 基类（持有 ParamList，`step()/zero_grad()`），v1 提供 SGD(momentum)/Adam/AdamW。

### 4.6 nn 层

- `Module` 基类：`forward` 虚函数 + 递归参数收集 + `train/eval` 模式。
- 内置层：Linear/Conv2d/Pool/BatchNorm/LayerNorm/Embedding/Dropout/激活/Sequential——全部只是"参数 Tensor + 调用 ops 自由函数"的组合，无内核。

---

## 5. 扩展路径

### 5.1 新增算子（四步，零改 core）

1. 声明调用函数：内置算子的调用点是 `include/znn.hpp` 的静态方法（`Tensor::matmul` / `Activator::relu` / `Loss::cross_entropy`）；扩展算子按同一风格在 `znn.hpp` 增加静态声明。
2. `src/operators/foo.cpp`：定义 `g_foo_operator`（extern 前缀保证外部链接）+ 前端函数；实现 `infer/grad/ref/kernels`。
3. （可选，要性能时再做）`src/operators/foo.cu|.mm|.comp` 填对应槽位。
4. `operators/operator_list.cpp` 的 extern 块与 `g_operator_list` 数组各加一行；CMake 的 `ZNN_OPERATORS` 清单加一行。

### 5.2 新增第五种后端（如 NPU/oneDNN-direct）

1. `Device` 枚举加一项，`kernels[Device::COUNT]` 自动扩容（重编译即可，无逐算子修改）。
2. `src/backends/npu/`：骨架（初始化/分配/拷贝/stream 预留）+ 选定 GEMM 委托对象。
3. 给关心的算子填 `kernels[NPU]` 槽，其余空槽自动兜底。
4. 对拍测试：NPU vs CPU golden。

### 5.3 新增图优化 pass（融合规则）

- 远期。eager 执行不依赖任何 pass；`pass_list` 注册制，每条 PatternRewrite 规则 ~20 行（conv+bn 折叠、add+act 融合起步）。zllm 的 RMSNorm+RoPE 融合走同一机制。

### 5.4 zllm 扩展包（共用+扩展，不在旧框架长新运行时）

内核已预留的钩子：dtype 含 F16/BF16；分配器预留 arena/可增长块；stream 字段预留；Operator 零改动。

zllm 作为平行包新增：

| 组件 | 形态 |
|---|---|
| KV cache 管理 | 普通 Tensor + 应用层环形/分页管理器 |
| paged attention / fused RMSNorm+RoPE | 新算子注册（走 §5.1 同一流程） |
| sampler (top-k/top-p/温度) | 纯 CPU 应用层代码 |
| GGUF / safetensors 解析 | 自写 ~300 行，"格式插件" |
| 自回归执行循环 / continuous batching | 应用层调度器，类比 ffmpeg.c 之于 libav* |

---

## 6. 后端策略与性能路线（不靠手搓堆量）

| 阶段 | 手段 | 代码量 | 预期 |
|---|---|---|---|
| v1 | Eigen GEMM + im2col conv | 0（已含） | PyTorch CPU 50–80% |
| v1.5 | 动态加载 oneDNN 替换 GEMM/conv + 推理 arena 内存池 | +200–400 胶水 | 90–100% |
| v2 | Winograd 模板 + NEON/AVX2 GEMV + INT8（int8 委托 cuBLASLt/oneDNN） | +800–1,500 | conv 类追平 ncnn 大半 |
| CUDA | cuBLASLt（epilogue 融合免费拿） | +400 胶水 | ≈ 原生 |
| Metal | MPS 系统库 | +300 | ≈ 原生 |
| Vulkan | 手写 GLSL compute GEMM（唯一绕不开的手写点） | +500–1,000 | cuBLAS 50–70% |

所有提速 = 往 `kernels[]` 槽位换实现，dispatch/autograd/用户代码零改动。

---

## 7. 三方库与职责

| 库 | 职责 | 引入阶段 | 形态 |
|---|---|---|---|
| Eigen | CPU GEMM 与向量化基础算子 | v1 | vendored header，仅内部使用，可裁剪（ZNN_MINIMAL） |
| cuBLASLt | CUDA GEMM + epilogue 融合 | 里程碑 2 | CUDA toolkit 自带 |
| Metal + MPS | Metal GEMM | 里程碑 3 | macOS 系统框架，零依赖 |
| （手写 GLSL） | Vulkan compute GEMM | 里程碑 4 | 内嵌字符串编译 |
| oneDNN | CPU 性能强化 + int8 | 里程碑 5 | **动态加载（可选）**，缺失则回 Eigen |
| doctest | 单元测试框架 | v1 | vendored 单头，仅测试目标 |
| IDX/MNIST 解析 | 里程碑 1 验收 | 里程碑 1 | examples 内自写 ~150 行 |
| GGUF/safetensors | LLM 权重加载 | 里程碑 6 (zllm) | 自写 ~300 行 |

唯一默认依赖 = Eigen。其余全部按 CMake 开关可裁剪。

---

## 8. 代码量预算（不含三方库）

| 部分 | 行数 |
|---|---|
| core（Tensor/注册表/autograd/优化器/分配器） | ~2,300 |
| 基础算子 ~30 个（含 conv/pool/norm/attention/loss，各 30–60 行） | ~1,900 |
| **v1 CPU 合计** | **~4,200** |
| CUDA 后端（骨架 + cuBLASLt 胶水 + 拷贝） | +800–1,200 |
| Metal 后端（MPS，单 ObjC++ 文件） | +600–900 |
| Vulkan 后端（运行时 + GLSL GEMM） | +1,200–1,800 |
| CPU 性能强化（oneDNN/Winograd/GEMV/int8） | +800–1,500 |
| zllm 扩展包 | +2,000–3,500 |

---

## 9. 里程碑与执行顺序（已商定）

1. **core + 基础算子（CPU 参考实现 + Eigen）+ MNIST 训练跑通** ← 验收节点，~2,500 行
2. CUDA 后端（正确性对拍 CPU 参考实现）
3. Metal 后端（MPS；仅 macOS 可构建，CMake 按平台启用）
4. Vulkan 后端
5. CPU 性能强化版（oneDNN 动态加载 / Winograd / GEMV / int8）
6. zllm 扩展包

第 1 步产出的 CPU 参考实现从此是所有后端的对拍基准（golden）。

---

## 10. 测试策略

- 每算子：`kernels[CPU]`（Eigen）vs `ref` 对拍；GPU 后端逐算子 vs CPU golden。
- 数值容差：fp32 相对误差 1e-5（GPU 半精度另定）。
- 端到端：MNIST 训练收敛阈值（如 5 epoch 内测试集 >97%）作为回归测试。
- 兜底链本身可测：`ZNN_FORCE_REF=1` 环境变量强制全库走参考实现。

---

## 11. 风险与现实约束

- **Vulkan 无 BLAS 可委托**，是唯一必须手写内核的位置（~1,000 行）。
- **Metal 仅 macOS 可构建**；Windows 开发机上 CUDA/Vulkan/CPU 可验，Metal 需 CI 或手动 mac 验证。
- Metal shader 需运行时编译或预编译 .metallib（v1 取运行时编译，简单）。
- 异步执行 v1 不暴露（全同步），但 KernelContext/接口预留 stream 位，避免日后破坏 API。
- 编译时间：转编译库后 Eigen 模板只在 znn 内部 TU 展开，用户工程秒级。

---

## 12. 接口草案 v0.4（讨论稿）

**权威版本已落为真实头文件，以文件为准**：`include/znn_defines.hpp`（结构定义）+ `include/znn.hpp`（接口定义）。v0.4 接口语言："一切操作挂在类名下"（Tensor/Activator/Loss/Runtime 静态方法 + Layer/Optimizer 虚接口与 create/destroy 工厂，无全局自由函数）；公共头零实现体（data/item 模板由库内显式实例化）。算子契约在 `src/operator.hpp`（库内开发接口，不随库发布）。

典型用法（MNIST 训练循环）：

```cpp
using namespace znn;

std::vector<Layer*> layers = {
    Layer::create(LayerType::Linear, LayerParam{/*.in=*/784, /*.out=*/256}),
    Layer::create(LayerType::ReLU,   LayerParam{}),
    Layer::create(LayerType::Linear, LayerParam{/*.in=*/256, /*.out=*/10}),
};
Optimizer* opt = Optimizer::create(OptimType::Adam, layers, OptimParam{/*.lr=*/1e-3});

for (auto& batch : mnist_train) {
    Tensor x = Tensor::from_host(batch.imgs.data(), {B, 784});
    Tensor y = Tensor::from_host(batch.labels.data(), {B}, DType::I64);

    Tensor h = x;
    for (Layer* l : layers) h = l->forward(h);     // 按论文逐层
    Tensor loss = Loss::cross_entropy(h, y);

    opt->clear();
    loss.backward();
    opt->step();
    printf("loss = %f\n", loss.item<float>());
}

Runtime::set_grad_enabled(false);                  // 验证/推理段暂停建图
/* ... evaluate ... */
Runtime::set_grad_enabled(true);

for (Layer* l : layers) Layer::destroy(l);
Optimizer::destroy(opt);
```

## 13. 内部机制设计（性能相关）

### 13.1 Tensor 与内存管理

- Tensor 句柄 = 单个裸指针（8 字节），传参/返回零成本；拷贝 = 原子引用计数 +1（约 1~2ns）。无 Python 绑定层与多级派发器，单算子调度开销目标 ≤ PyTorch eager（约 1~5µs）的 1/5。
- Impl 内联 shape/stride（≤8 维不额外堆分配）；Impl 与求导图节点从对象池分配复用，稳定循环中无 malloc。
- v1.5 引入 arena：前向激活从内存池线性分配、整池回收，推理循环零 malloc/free。
- 视图零拷贝：slice/transpose 只新建池内 Impl，共享 Storage。
- 峰值内存：反向按依赖计数逐步释放保存的中间量，峰值 ≈ 前向激活总量（与主流框架相当）；v1.5 按算子精简保存列表，重算式检查点远期预留。

### 13.2 线程模型（三层）

1. **用户层（v1 全同步语义）**：同步指数据一致性在读取点（`item`/`data`/`to_device(CPU)`/`step`）保证；GPU 上相邻算子在同一 stream 流水执行，并非每算子同步等待。
2. **库内（全局线程池 fork-join）**：复用 Eigen ThreadPool（零额外代码），逐元素/规约按外维并行；委托库（oneDNN 等）共享同一池，避免线程超额。
3. **线程安全边界**：注册表只读无锁；不同线程各自建模型/推理完全并行；同一 Tensor/Layer/Optimizer 不做内部同步，跨线程共享由调用者串行化；grad 开关为 thread_local。

GPU：v1 单默认 stream；多 stream 远期，KernelContext 已预留字段。

### 13.3 自动求导引擎

- 前向建 tape：每个建图算子从节点池分配 GraphNode（Operator 指针、保存张量、依赖计数）；**节点创建序即拓扑序，反向无需排序**。
- backward：从损失逆序遍历，分支汇合用"输出消费计数归零触发"（等价 PyTorch InputBuffer，约百行），梯度在 grad buffer 累加。
- 单线程遍历 + 内核内部并行（反向耗时由内核主导，图级并行收益小，远期可选）。
- 节点与中间量用完即时归还池，见 13.1 峰值内存策略。

### 13.4 数据通路

- CPU 内部：视图直传（内核支持 strided 输入或先紧凑化）。
- host→device：`from_host` 单次 DMA；v1.5 加 pinned 暂存 + 双缓冲预取。
- device 内部：同 stream 流水，无 host 往返。

## 14. 内部机制问答（评审问题沉淀）

### 14.1 AutogradEngine::record 的命名与作用

"record" 是动态图领域标准动词（PyTorch RecordFunction / TF GradientTape.record）：把一次运算记录到计算图。步骤：节点池分配 GraphNode → 记 op 入口与 saved 张量(共享存储不拷贝) → 链接上下游(输入的 consumers += 本节点, pending++) → 写各输出 Impl::grad_fn。触发条件：grad_enabled && op->grad && 任一输入 requires_grad；纯推理零开销。

### 14.2 operator.hpp 逐项职责与数据流位置

| 定义 | 作用 | 数据流位置 |
|---|---|---|
| Span | 指针+长度视图, 零拷贝传参 | 所有内核签名 |
| TensorMeta | 张量名片(shape/dtype/device, 无数据指针) | run 调 infer 前构造 |
| KernelContext | 内核上下文(设备 + 预留 stream) | run 构造并传入内核 |
| BackwardContext | 反向环境: saved 张量/grad_out 入/grad_in 出/reduce_to 广播规约 | backward 逐节点装配 |
| KernelFunction | 后端无关算体签名(数据几何经 Tensor 公共访问器获取) | run 经函数指针调用 |
| ShapeInferenceFunction | 形状推导(分配输出之前) | run 第 2 步 |
| GradientFunction | 反向规则(nullptr=仅推理) | backward 逐节点调用 |
| Operator | 注册项: name/infer/grad/ref(对拍基准)/kernels[后端槽位] | 前端引用, run 消费 |
| g_operator_list | 注册数据数组(直接普通数组, 无宏) | 首次 instance() 建表后只读 |

### 14.3 op_dispatcher 在数据流中的位置(注册发生在编译期)

- instance(): Meyers 单例, 首次调用遍历 g_operator_list 建 name→Operator 哈希表, 此后只读无锁
- find(): 调试/测试; 正式数据流零字符串查找(前端直接持 &g_*_def)
- run(): 29 个算子前端的唯一入口, 完整链路八步:
  构造名片 → infer 形状 → 分配输出(Impl::create→Storage) → 按输入设备选内核(空槽兜底: CPU→ref, GPU→拷回 CPU+warning) → 执行内核 → 满足条件则 record 建图 → 返回输出

### 14.4 mut 与 Storage

- mut(): const 成员函数返回可写参数指针的 const_cast 封装。备选方案(待定): params() 改返回 std::vector<Tensor>(句柄共享 Impl), const_cast 可整体消除。
- Storage = 引用计数内存块(数据+生命周期); Tensor::Impl = 数据视角(offset+strides)。视图(slice/transpose)共享同一 Storage 实现零拷贝, 最后一个引用释放内存。与 PyTorch Tensor/Storage、numpy ndarray/base 同构。
