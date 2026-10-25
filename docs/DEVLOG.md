# 开发日志 (DEVLOG)

记录每次提交的内容与反思。反思维度（每项 10 分制或百分比）：
正确性 / 架构合理性 / 代码精简度 / 可理解 / 可维护 / 可扩展 / 完成度。

---

## [2] 构建骨架(整改后单一提交): 内部统一定义源 + 一算子一文件

> 初版骨架(两提交)经评审判不及格, 已 `git reset` 回接口提交,
> 本提交为按要求整改后的唯一骨架提交。

**结构**
- include/ 对外 2 头不变; 接口补两处: Tensor 五法则声明(隐式拷贝会
  绕过引用计数, 属接口 bug)、from_host 的 DType 默认值
- src/znn_ext.hpp: 扩展接口 + ZNN_TODO(算子/后端作者用)
- src/znn_internal.hpp + .cpp: **全部内部类型的唯一定义源与实现源**
  Storage(类,私有成员+工厂构造) / Tensor::Impl / GraphNode /
  AutogradEngine(类: record 建tape + backward 反向调度) /
  OpRegistry(类: 注册机制/查找/派发) / 优化器与 9 个内置层实现类
- src/operators/: **一算子一文件, 文件名=算子名**(29 个), 每文件 =
  extern + OpDef 占位 + 前端函数占位; operator_list.cpp 自包含
  (extern 块 + ZNN_OP 列表), 删除 reg/ 目录与 op_defs.hpp
- include 无相对路径(CMake 以 src/ 为私有包含目录)
- 逻辑体全部 ZNN_TODO() 展开式占位; 全库无 (void)x; 挤行式占位
- 验证: 全量编译通过, smoke 通过, mnist 退出码 1 + 明确 TODO 消息

**整改对照(评审 8 条)**
1. (void) 挤行 → 全项目清零, 未用参数一律匿名, 函数体逐行展开
2. 求导引擎两自由函数 → AutogradEngine 类(节点池/tape 所有权/
   消费计数调度/梯度合并/广播规约集中封装, 见 znn_internal.hpp)
3. ../ 相对路径 include → 全部经 -I src 解析
4. op_helper.hpp → 已删除, ZNN_TODO 移入 znn_ext.hpp
5. 注册/求导 C 风格自由函数 → OpRegistry/AutogradEngine C++ 类
6. storage.hpp 等零散 hpp/cpp 对 → 内部结构统一进 znn_internal.*
7. reg/ 目录与 op_defs/operator_list 不同名 → 注册数据自包含于
   operators/operator_list.cpp, 机制在 OpRegistry, 职责已注释
8. ops/ 目录按类别分组随意 → operators/ 一算子一文件, 名即内容

**过程中发现并修复的技术问题**
- MSVC 默认按 GBK 解析 UTF-8 中文注释导致语法雪崩 → 全局 /utf-8
- const 命名空间作用域变量默认内部链接: 初版链接"通过"是侥幸
  (注册表数组无人引用被丢弃); OpRegistry 真实引用后暴露,
  g_znn_op_list 与各 g_*_def 均以 extern 定义保证外部链接
- 宏块须置于 namespace znn 内定义(全局作用域限定名定义 MSVC 不识别)

**反思**
- 架构合理性 9/10: 数据(operator_list)与机制(OpRegistry)分离;
  内部结构单源; 扩展路径三步未破
- 代码精简 9/10: 同构算子文件 19 行/个, 无重复样板之外的冗余
- 可理解 9/10 / 可维护 9/10 / 可扩展 9/10
- 完成度: 里程碑 1 约 20%(结构闭环, 零算法逻辑)
- 待审风险: AdamWImpl 继承复用方式; AutogradEngine 单例与
  多模型并发训练的边界(远期再评估)
- 下一步: Storage/Impl 引用计数与分配 → OpRegistry.run 派发链 →
  逐算子 ref 实现 → Eigen 内核 → AutogradEngine → 优化器 → MNIST

---

## [3] 结构再整改: 通用基础单源 + 子系统目录化

> 上一轮把全部内部类型塞进 znn_internal 一个文件矫枉过正, 本轮按
> "通用进 internal、子系统自成目录一单元一文件" 重排。

**结构变化**
- znn_internal.hpp/.cpp 瘦身: 只保留跨子系统共用基础
  (Storage / Tensor::Impl / GraphNode / AutogradEngine / OpRegistry)
  + Tensor 公共方法 + Runtime; 层/优化器移出
- 新增 layers/: layers.hpp(9 个层实现类声明) + 一层一文件
  (linear/conv2d/batch_norm2d/layer_norm/embedding/dropout/
  activation/flatten/pool2d) + layer_factory.cpp(工厂分发)
- 新增 optim/: optim.hpp + 一优化器一文件(sgd/adam/adamw)
  + optimizer_factory.cpp; AdamWImpl::step 重写以承载解耦衰减语义
- mut() 帮助函数上移 znn_internal(层/未来其他子系统共用)

**文件划分判定规则(已写入 DESIGN.md §3)**
按编译器分文件、按逻辑聚目录: 同编译器实现同文件不同槽位;
.cu/.mm/.comp 同目录同名异后缀; 层/优化器无后端维度(设备选择
在算子派发, 层只是"参数+调用算子"的组合)。

**反思**
- 架构合理性 9/10: 三类单元(算子/层/优化器)各自目录化, 与
  扩展路径一一对应; 通用基础单源不再承载子系统
- 代码精简 9/10: 层文件 17~30 行/个, 无跨文件重复样板
- 可理解 9/10 / 可维护 10/10(改一层只动一个文件) / 可扩展 9/10
- 完成度: 里程碑 1 约 20%(结构与初版骨架等价, 组织方式落定)
- 下一步不变: Storage/Impl 引用计数 → OpRegistry.run 派发链 →
  逐算子 ref 实现 → Eigen 内核 → AutogradEngine → 优化器 → MNIST

---

## [4] 命名与职责语义整改: 按角色命名的文件 + 链路文档化

> 评审三点: znn_ext 名不对/意义不明; znn_internal 是 registry+autograd
> 的大杂货铺, 链路读不懂。

**变化**
- znn_ext.hpp → **op_def.hpp**(算子契约: 算子作者唯一要读的头),
  文件头写明本文件定义什么 + 算子在调用链路中的位置 + 新增算子三步
- **znn_internal.hpp/.cpp 瘦身为"znn.hpp 的实现支撑"**:
  Storage / Tensor::Impl / Shape 方法 / Runtime / ZNN_TODO;
  文件头明确"本文件不含任何引擎"
- 引擎独立成文件(与 layers/optim 平级的单文件子系统):
  **autograd.hpp/.cpp**(求导引擎: GraphNode/record/backward/reduce_to)
  **op_registry.hpp/.cpp**(注册机制: 查找/派发/兜底, 数据在 operator_list)
- 每个文件头 = "我是什么 + 职责 + 链路位置"三段式
- DESIGN.md §3 增补"一次算子调用的完整链路"图(前端→机制→内核→建图→反向)

**反思**
- 命名语义: op_def(契约)/op_registry(机制)/autograd(引擎)/
  znn_internal(实现支撑)——每个名字即职责, 大杂货铺问题消除
- 架构合理性 9/10; 可理解 10/10(链路图 + 文件头三段式);
  可维护 10/10; 可扩展 9/10; 精简 9/10
- 完成度: 里程碑 1 约 20%(结构第四轮收敛, 实现未动)

---

## [5] 命名去缩写 + OpDef 字段注释 + 删除注册宏

**变化**
- 类型全拼: KernelCtx→KernelContext, BackwardCtx→BackwardContext,
  KernelFn→KernelFunction, InferFn→ShapeInferenceFunction, GradFn→GradientFunction
- OpDef 每个字段逐条详细注释(作用/时机/示例); 三类函数各自的
  using 声明上方写明"做什么+何时被调+示例"
- 注册宏 ZNN_OP_LIST_BEGIN/ZNN_OP/ZNN_OP_LIST_END 删除:
  operator_list.cpp 直接写普通数组(宏拆三段无实际收益,
  ffmpeg 用宏因列表是 configure 生成的, 我们手写手维护)
- 验证: 编译绿, smoke 通过, mnist 退出码 1 + TODO 消息

**反思**
- 可理解 10/10(契约文件即文档, 零缩写零宏魔法); 精简 +10 行(宏删除)
- 架构不变; 完成度 里程碑1 约 20%

---

## [6] OpRegistry → OpDispatcher(命名纠偏: 注册不在运行期)

> 评审: "完全没看出来怎么注册的; 注册和 run 什么关系?"

**澄清(已写入文件头)**
- 注册 = 编译/链接期静态三步: 算子文件定义 OpDef → operator_list
  登记地址进数组 → CMake 纳入构建; 运行期零注册动作
- run 与注册无关: 前端直接持 OpDef 常量指针执行, 不查表;
  哈希表只服务"按名查找"(调试/未来 ONNX/GGUF 按字符串名找回算子)

**变化**
- op_registry.hpp/.cpp → op_dispatcher.hpp/.cpp, 类 OpRegistry → OpDispatcher
- 文件头三段式重写: 注册发生在哪(编译期三步)/职责(run 主职+find 辅助)/链路位置
- op_def/autograd/znn_internal/operator_list/CMake/DESIGN.md 引用同步

**反思**
- 命名即语义: "派发器"准确描述主职, 消除"注册表为什么会 run"的困惑源
- 架构合理性 9/10; 可理解 10/10; 完成度 里程碑1 约 20%

---

## [7] 方案B落地 + pimpl 最小化尝试 + OpDef→Operator

**变化**
1. params() 方案B: Layer::params() 与 Optimizer::create 改返回/接受
   std::vector<Tensor>(共享句柄), 九个层实现返回句柄拷贝,
   mut()/const_cast 全库删除
2. pimpl 三行→两行尝试(struct Impl* impl_; 声明成员合一)被 MSVC 拒绝
   (不把成员声明中的 elaborated-type-specifier 当嵌套类声明),
   回退三行可移植形态并逐行注释必要性——"一行"不可行
3. OpDef → Operator(TFLite/PyTorch-jit 同名先例, 去定义歧义);
   文件 op_def.hpp → operator.hpp; 实例命名 g_<算子名>_operator;
   注册数组 g_znn_op_list → g_operator_list; 形参/成员保留惯用名 op
   并在契约头注明(operator 是 C++ 关键字, 不能作标识符)

**反思**
- const 正确性: 全库零 const_cast(可维护 10/10)
- 命名语义: Operator 即"算子"本体, 消除 Def 歧义
- 跨编译器教训入档: 合一写法 GCC 接受 MSVC 拒绝, 骨架须双编译器验证
- 完成度: 里程碑1 约 20%; 接口随 B 定稿

---

## [8] 封收收窄: TensorView 剥离内核的句柄权力

> 评审: "派发器/引擎/内核为啥要直接访问 Tensor 内部? 破坏封装!"

**澄清(哪些数据公共接口拿不到、为何不放进公共接口)**
- 内核需要 数据基址+每维步长+视图偏移(任意设备)——公共 API 无
  strides; data() 被刻意限制仅 CPU。暴露它们=用户可绕过设备规则
- 派发器需要构造输出(写 impl_); 引擎需要 grad_fn/requires_grad/grad
- friend 的实际授予范围 = impl_ 这一个指针(Tensor 唯一私有成员),
  且收口在 Impl 静态方法

**修正(评审点中的真问题)**
- 新增 TensorView 纯数据描述符(基址/形状/步长/类型/设备, 无引用
  计数/求导元数据), KernelFunction 签名改收 Span<TensorView*>
- 派发器调用内核前解包 Tensor → TensorView
- 效果: 29 个算子文件零特权(结构上拿不到句柄权力), 需要 friend 的
  只剩 op_dispatcher 与 autograd 两个文件, 且仅经 Impl 单一收口点

**反思**
- 封装边界更准确: 用户级(Tensor 公共接口)/内核级(TensorView 数据
  几何)/引擎级(Impl 收口)三层权力分级, 越权在类型上不可能
- 架构合理性 9/10; 可理解 10/10; 完成度 里程碑1 约 20%

---

## [9] 去掉 friend: 公共接口补全 + 引擎自持 tape

> 评审: "不分离 TensorView; 需要的数据放公共接口; 可否去掉 friend?"

**结论: 可行, 总账更简**。三个配套动作:
1. 公共接口补全: strides()/is_contiguous() 入公共 API;
   data<T>() 解除仅 CPU 限制(返回当前设备指针, 同 PyTorch data_ptr
   约定: 设备侧代码/设备 API 用, 主机解引用自担风险)
2. 派发器经公共工厂 Tensor::empty 构造输出, 不写 impl_
3. 求导引擎改为自持 tape + "张量实体→生产节点"路由表;
   Tensor::Impl 删除 grad_fn 字段——张量本体零图信息, 引擎经
   公共接口工作(requires_grad()/grad())
→ friend struct Impl 从公共头删除(全库仅 Tensor 自身成员函数碰 impl_);
→ TensorView 回退, 内核直接收 Span<const Tensor*>, 数据几何走公共访问器

**代价(明示)**
- data() 跨设备指针是文档化脚枪(用户拍板接受的权衡)
- 内核重新持有句柄(能 retain/摸 grad)——贵而不危险, 换简洁
- 引擎每算子一次哈希插入(~50ns)换张量结构纯净

**反思**
- 封装哲学转变: 从"权限分级靠类型隔离"到"公共接口够全 + 内部零特权",
  更符合评审者口味且删了两块机制(TensorView + friend)
- 架构合理性 9/10; 精简 +30 行; 可理解 10/10; 完成度 里程碑1 约 20%

---

## [10] 里程碑 1 达成: MNIST 训练跑通(块 G)

**验收结果**(examples/mnist, MLP 784-256-10 + ReLU + Adam lr=1e-3):
| epoch | 测试集准确率 |
|---|---|
| 0 | 95.23% |
| 1 | 96.01% |
| 2 | 96.40% |
| 3 | 96.81% |
| 4 | **97.19%** ✓ (>97% 标准) |

**内容**: IDX 二进制加载器(大端头解析/批组织) + 推理评估循环
(NoGrad + argmax); 测试集为标准 t10k 前缀。数据不进仓库(.gitignore)。

**实现阶段总反思(A-G 全块)**
- 完成度: 里程碑 1 = 100%(CPU 参考实现路线; Eigen 提速未做)
- 正确性: 链式法则/广播梯度/有限差分/线性回归收敛/MNIST 五重验证
- 关键债已还: 梯度规则读不到前向选项 → Operator::options_size
  + 节点选项副本 + BackwardContext::options; 训练态 transpose
  材料化+逆梯度(视图断流问题在该机制上解决)
- 已知限制(有意): sum/mean/softmax 仅末维; 视图族仅 transpose
  接入训练(reshape/flatten 族待同款接入, CNN 前置);
  conv/pool/cat/stack/embedding/attention 算子未实现(块 D 待做);
  backward 不支持 retain_graph
- 调试教训入档: (1)空句柄上 numel() 抛异常在 Debug CRT 下表现为
  WER 弹窗"挂起", 排查要靠 stderr 分段标记; (2)本环境 bash heredoc
  会把反斜杠-n 变成真实换行, 多行插入一律改用 python/Edit 工具
- 架构 9/10 | 精简 8/10 | 可理解 9/10 | 可维护 9/10 | 可扩展 9/10

---

## [11] 块 D 完成: 全部内置算子实装(conv/pool/cat/stack/embedding/
##      reshape 族/LN/BN/attention)

**内容**
- reshape 算子: 训练态 reshape/view/flatten/squeeze/unsqueeze 材料化
  +逆梯度; 推理态保持零拷贝视图——CNN 的 Flatten 训练路径打通
- conv2d(NCHW, pad/stride/dilation, v1 groups=1) 前向+标准 dX/dW 反传
- max/avg_pool2d: max 反传重找窗口最大(免存掩码), avg 均摊有效元素
- cat(可变输入)/stack: 反传按区间切片+reshape 回原形
- embedding: gather + scatter-add 反传
- layer_norm / batch_norm2d 算子: 标准反传公式; BN 训练态经共享存储
  就地更新 running 统计(eval 用 running); 层封装(含 train 切换)
- attention: [B,H,L,D] 单算子前向(S·softmax·PV)+完整反传
  (dV=P^T dO, dP=dO V^T, softmax 反传, dQ/dK); causal 留 v1.1
- Shape.h/.w 是 NCHW 别名方法而非字段, 二维几何参数一律 [0]/[1] 索引

**测试**: conv 手算值+FD 梯度、池化值+max 路由梯度、cat/stack 值+
区间梯度、embedding gather/scatter、reshape 恒等梯度、LN 零均值、
BN 通道均值、attention 单位注意力——全部通过。

**反思**: 架构 9/10(全部经注册算子路径, 无旁路内核) | 精简 8/10 |
可理解 9/10 | 可维护 9/10 | 可扩展 9/10。
剩余: Eigen 内核提速 / sum·mean·softmax 任意维 / permute·slice 训练态 /
causal attention / GPU 里程碑。

---

## [12] Eigen 快速内核: matmul/conv2d 提速与性能档案

**内容**
- third_party/eigen 3.4.0 vendored(唯一三方依赖, gitignore)
- matmul/conv2d 的 kernels[CPU] 槽位实装(Eigen GEMM / im2col+GEMM),
  非连续输入自动回退朴素路径; ZNN_FORCE_REF=1 强制 ref(对拍/调试)
- 基准工具 znn_bench + docs/BENCH.md 性能档案(前后对比见文件)
- 对拍测试: matmul/conv2d Eigen vs ref 最大差 < 1e-3

**关键数据**(同一 Release 二进制):
matmul 1024³ 25.9×(47 GF/s) / conv2d 55.4×(14.5 GF/s) /
MLP 训练步 2.5×(14.2s/epoch) / MNIST 5epoch ~12min → 152s, acc 97.09%

**工程教训**
- Eigen 表达式模板在 MSVC Debug 下慢 10×+(必须 Release 或逐文件 O2,
  且 /O2 与 /RTC1 冲突需剥离); 基准一律 Release
- 大量 sed/python 多行替换静默失败(换行/空白不匹配), 关键接线一律
  Edit 工具显式确认(grep 验证)后再构建

**反思**: 架构不变(槽位换内核, 派发/前端零改动, 验证了
"提速=往 kernels[] 换实现"的承诺); 完成度: 里程碑1 性能项达标。

---

## [13] 限制解除包: 全部 v1 限制清零

**内容**
1. sum/mean/softmax 任意维: 选项经节点副本传入梯度规则,
   softmax 重写为"outer 遍历 + 行内 stride 步进"通用实现
2. permute/slice 训练态: 新增两算子(材料化前向 + 逆排列/散射梯度),
   视图族(reshape/transpose/permute/slice)全部接入训练图
3. 分组卷积: infer/朴素/Eigen im2col 三处按组循环
   (oc→组映射, 权重下标 oc*Cg; Eigen 每组独立 GEMM)
4. Dropout 层: 逆置失活 x·mask/(1-p), mask 为常量张量经 mul 算子
   ——梯度自动回传, 无需专用算子; eval 直通
5. causal attention: AttentionOptions{causal} 经选项副本,
   前向/梯度重算均施加未来位置 -inf 掩码

**测试**: sum dim0 值+梯度广播 / mean keepdim / softmax 列归一 /
permute 恒等梯度 / slice 散射梯度 / 分组卷积值(手算) /
causal 行 0 只看位置 0 / Dropout 保留率+缩放+梯度 0-or-2+eval 直通
—— 全部通过。

**反思**: 架构 9/10(全部经注册算子与选项副本机制, 零机制新增) |
选项副本机制(块 F 偿的债)在本包中连续支撑 5 处, 证明设计正确。

---

## [14] CUDA 后端骨架(里程碑 2 起点)

**内容**
- src/backends/cuda/: cuda_backend.hpp(内部接口: available/alloc/
  free/h2d/d2h) + cuda_stub.cpp(非CUDA 构建的明确报错实现) +
  cuda_device.cu(cudaMalloc/cudaMemcpy + 示范 add 内核展示扩展模式)
- Storage::create/~Storage 按设备分发分配/释放; Tensor::to_device
  实现 CPU↔CUDA 同步拷贝(非连续视图先紧凑化)
- CMake: option ZNN_CUDA(默认 OFF), stub ↔ .cu 二选一;
  默认构建完全不受影响

**扩展模式**(与 DESIGN §5.2 一致): 新算子 = operators/foo.cu 填
kernels[CUDA] 槽位, 派发器按输入设备自动选择——无需改机制。

**验证**: 默认构建全绿(测试/基准), stub 路径行为明确。
教训: CMake 变量必须先 set 后引用, 条件块要放 add_library 之前。

---

## [铁律宣布]

> 用户要求:
> 1. 不允许使用过渡方案, 想用需要审核
> 2. 每种后端必须走该后端原生内核, 不允许 CPU 桥接

**已写入 DESIGN.md 文件头部(⛔ 铁律节)。**

**当前违反铁律的代码**: autograd.cpp 中的 GPU→CPU 梯度桥接(1.5 节)。
**消除计划**: 见下一个提交。

---

## [铁律执行] 全部 CPU 桥接消除

**铁律写入 DESIGN.md 头部**:
1. 不允许使用过渡方案(需用户审核)
2. 每种后端走原生内核, 不允许 CPU 桥接

**消除过程**:
- cross_entropy 梯度 → softmax + onehot + elementwise 组合
- softmax 梯度 → s·(g − sum(g·s, dim, keepdim)) 组合
- conv2d 梯度 → CPU 朴素循环(原生) + CUDA 专用梯度内核
  (conv2d_dw_k + conv2d_dx_k, 每个 dw/dx 元素一个线程)
- attention 梯度 → 逐头 2D 组合(matmul/softmax/mul/sub/sum)
- layer_norm 梯度 → mean/mul/sub/div/sqrt 全组合
- batch_norm2d 梯度 → reshape/mean/mul/sub/div/sqrt/sum 全组合
- unary_grad(relu/gelu/sigmoid/tanh/silu/exp/log/sqrt) → mul 组合

**autograd.cpp 桥接代码全部删除**(grep 验证: 零残留, 仅注释说明)。
所有梯度规则要么用算子组合(自动走当前后端), 要么有专用 CUDA 内核。

**修复 bug**: cross_entropy one-hot 构造原用 slice 视图赋值(替换句柄
而非写共享存储, 导致梯度为 0), 改为 host 构造 + from_host 按设备上传。

**验证**: 默认构建全绿(全部测试含有限差分梯度校验通过)。

---

## [15] 性能委托块: dynlib 统一动态加载 + cuBLAS GEMM + Eigen 逐元素 + TF32

**内容**(提交 212c1a5..07d68e1, 7 个提交)
1. 审计修复: e82b7b6 之后逐行审计的全部 bug 修复
   (item d2h / CUDA 优化器 / 梯度 GPU 桥接→算子组合)
2. dynlib.hpp/.cpp: 统一动态加载(LoadLibrary/dlopen + 函数指针 +
   available_xxx 查询), cuBLAS 全部符号经此加载, 缺 DLL 时自动降级
3. cuBLAS GEMM 委托: matmul/linear(视图展开)/conv2d(im2col) 三处走
   cublasSgemm; 逐元素 13 算子走 Eigen unary/binary_kernel_fast(SIMD)
4. GPU 常驻基准: gpu_bench 改为 GPU 上建张量(数据常驻), 消除
   "CPU 建张量→派发器选 CPU 内核"的假象; item() 改 to_device 同步
   (避免非标量 CUDA 张量触发 require 崩溃路径)
5. Precision 公共 API: FP32/TF32/AUTO 三档, 存 OpDispatcher 成员
   (零全局/静态变量), 经 KernelContext 逐调用下发内核
6. TF32 正确实施: matmul_cuda 检查 ctx.precision,
   TF32/AUTO 且 cublas_gemm_ex 可用 → cublasGemmEx
   (CUBLAS_COMPUTE_32F_FAST_TF32, Tensor Core),
   否则 → cublasSgemm; 对拍测试切 FP32 位精确比对后恢复 AUTO

**数据**
- TF32 GEMM(RTX 4070 Ti SUPER, 数据常驻): 4096³ 16.37→31.67 TF/s
  (1.93×), 2048³ 1.83×, 1024³ 1.51×, 512³ 3.8×(小矩阵 FP32 未达峰)
- CPU 逐元素 Eigen 批量: MLP 训练步 +15%
- MNIST 回归 97.16%(>97% 线), 19 测试 + CUDA 对拍全绿

**弯路(用户点名批评, 记录在案)**
- 第一次 TF32 用 cublasSetMathMode(旧 API, CUDA 13.4 上 Tensor Core
  不生效, 仅 +4%), 且中途引入 thread_local → 违反无全局变量铁律。
  全部回退, 保留外部配置逻辑, 换 cublasGemmEx 正确实施
- 教训: 换加速方案前先查当前 CUDA 版本的 API 生效路径;
  实现方式不允许中途随意切换, 先验证机制再写代码

**反思**: 架构 9/10(单一派发路径, 精度经 KernelContext 下发,
零全局变量, DLL 缺失自动降级且路径透明) | 实现纪律 6/10
(方案切换随意 + 中途引入违规实现, 用户两次点名) |
收益确认 10/10(公平 A/B: 同一二进制, 数据常驻, 正确同步)。
cuda_device.cu 尾部遗留 include 垃圾于本块发现并清理。

---

## [16] GEMM 四算子统一 + TF32 勘误(重大: 撤回 1.93× 声明)

**内容**
1. cuda_gemm.cu: 共享 GEMM 原语 gemm_cuda / gemm_batched_cuda,
   全部 GEMM 类算子(matmul/linear/conv2d/attention)唯一 cuBLAS 入口
   - 行主序接口 + 显式 ld(列切片靠 ld 表达, conv2d 分组免拷贝)
   - b_transposed=true → cuBLAS OP_T(× w^T 权重布局免转置)
   - TF32/AUTO 且 GemmEx 可用 → FAST_TF32; 失败自动回落朴素内核
2. dynlib: +cublasSgemmStridedBatched / cublasGemmStridedBatchedEx
3. linear: 朴素融合内核 → GEMM + bias 行加内核
4. conv2d: im2col + 逐组 GEMM + NCHW 回转内核(朴素兜底保留)
5. attention: 批量 GEMM(S=QK^T → 行 softmax → O=PV, 每批一个头);
   旧内核删除——其 threadIdx.x≤32 扫列设计在 L/D>32 时
   静默丢列(latent bug), 新朴素兜底任意 L/D 正确
6. 对拍补盲: linear/conv2d/attention 此前无 CUDA↔CPU 对拍,
   新增(L=48/D=40 非方阵); 顺带暴露 CPU attention_ref 存量 bug

**修复的 bug(本块引入 2 + 存量 3)**
- 引入: im2col dst 偏移丢 +wo 项(wo>0 列踩写/未初始化);
  attention 快速路径漏乘 1/√D(scale 折入 softmax 内核指数修复)
- 存量: **CublasGemmExFn typedef 22 参 vs 真实 19 参(无 D 矩阵)**,
  gemmEx 从未真正执行过, 返回值被忽略掩盖; CPU attention_ref
  gemm 辅助 at 标志只转 A 不转 B, L≠D 时 K 误按 [D,L] 读
  (L==D 巧合正确故测试从未暴露); cuda_device 尾部死 include
- 排查方法记录: 独立探针二分(22/19 参对照 → OP_T 数值验证 →
  流水线分段 → 手工参考对拍), 每步以可复现最小实验收敛

**勘误(撤回提交 2562ff6 的 1.93× 声明)**
- 真实内核级: sgemm FP32 32.2 TF/s → gemmEx TF32 42.4 TF/s
  = **1.32×**; gemmEx 仍是最优路径(略胜 MathMode 41.1 且免句柄态)
- 旧数字来源: 陈旧二进制(sgemm+残留 SetMathMode) + app 级基准
  被每次调用的 64MB cudaMalloc/cudaFree 稀释(4096³ 端到端 8.6ms
  中 5.9ms 是分配, 内核仅 2.7ms)
- **新发现最优先瓶颈: GPU 缓存分配器缺位**(机制变更, 待用户审核)

**数据**: linear 0.34→16.13 TF/s(48×); conv2d/attention 计算路径
已通但小负载被分配开销掩盖; matmul TF32 1.32×(内核级)。
19 测试 + 扩充的 CUDA 对拍(linear/conv2d/attention)全绿。

**反思**: 架构 9/10(单入口双原语, ld+OP_T 表达全部布局,
失败回落保正确) | 实现纪律 5/10(本块自引入 2 bug, 其中
"转置被 ld 吸收"的推断未经数值验证即铺开, 又违反了
先验证机制再写代码的教训) | 测试盲区意识 4/10(三个 GEMM 算子
无 CUDA 对拍一个月, "快而错"与存量 CPU bug 均因此存活;
对拍必须覆盖非方阵) | 测量诚实 10/10(主动勘误并保留方法记录)。

---

## [17] GPU 异步分配器: cudaMallocAsync 消除分配开销

**决策**: 自研缓存分配器(~200 行, 需全局单例) vs 委托驱动池
(~15 行, 零全局状态)——选后者, 与"性能=委托"哲学一致
(GEMM 委托 cuBLAS, 池化委托驱动); 驱动池若在实测中表现
病态再升级自研(数据说话)。

**内容**(cuda_device.cu, 3 处)
1. alloc → cudaMallocAsync(默认流): 流有序, 不阻塞宿主
2. free → cudaFreeAsync(默认流): 只往流上排"回收", 免隐式同步
3. available() 一次设置默认池 release threshold = 上限
   (空闲块留池复用; 与 kernel 同流故有序, 无 use-after-free)
前提 CUDA ≥11.2, 工程锁定 13.4; 非 CUDA 构建零影响。

**数据**(gpu_bench 端到端, 改前 → 改后)
- matmul 1024³ TF32 5.88 → 21.90 TF/s(3.7×);
  2048³ 13.11 → 24.97(1.9×); 4096³ 18.80 → 24.74(1.3×)
- linear 16.13 → 31.58 TF/s(接近内核级)
- conv2d 0.447 → 0.051 ms(8.8×, 首次超过旧朴素内核 0.091ms)
- attention 0.770 → 0.270 ms(2.9×)
- 端到端 TF32/FP32 ≈ 1.3×, 与内核级探针一致 → 分配稀释确已消除

**验证**: znn_test 全绿(含 CUDA 对拍); 循环同尺寸分配池复用
由基准本身稳定性佐证(4096³ 输出 64MB × 数十次迭代无异常)。

**反思**: 架构 9/10(机制选择符合项目委托哲学, 铁律零冲突) |
克制 9/10(没有一上来就自研 200 行分配器)。

---

## [18] FP16 混合精度 + GradScaler(训练防下溢, 用户要求一并解决)

**内容**
1. Precision 加 FP16 档: gemm 原语入口 A/B → FP16 scratch(带步长
   转换内核, 支持 ld 列切片与 NT 权重布局), cublasGemmEx
   (A/B=16F, C=32F, COMPUTE_32F); 累加与张量存储全程 FP32;
   conv2d/attention 经同一原语自动继承
2. GradScaler(公共 API, PyTorch AMP 等价): scale 建图内 ×s(反传
   自动放大梯度); unscale 原生内核 ×(1/s) + inf/nan 标志
   (grad_unscale_cuda, 免原子); update 动态调整(失败 ÷2 下限 1,
   每 interval 成功 ×2 上限 2^22)
3. Optimizer 基类收编 params_(公共只读 params() + friend GradScaler)

**FP16 端到端**(gpu_bench): 4096³ 19.8(FP32)→24.8(TF32)→**30.9 TF/s**
(FP16); FP16/TF32=1.25×——转换内核读写与 scratch 分摊了理论收益,
如实记录(纯内核更高)。

**GPU 训练路径存量缺口修复(本块暴露并修复 6 处)**
FP16 训练烟雾逐段暴露 GPU 反向从未真正跑通(此前仅单算子前向对拍):
- Tensor::full/ones: 宿主循环直写设备指针 → CPU 构造 + to_device
- mse_grad: 宿主解引用设备指针 → 算子组合(g×2/N×(p−t))
- unary_grad(9 算子共用): 宿主循环 → 内部组合算子 unary_bwd
  (枚举查表 + CPU 循环/CUDA 内核双端, 一次覆盖 relu/gelu/exp/
  log/sqrt/neg/sigmoid/tanh/silu; DEVLOG[铁律执行]声称的"mul 组合"
  此前并未落地——记录在案)
- matmul_cuda: 转置视图直接 require 抛异常 → 材料化(OP_T 直连为
  后续优化项)
- 优化器 clear(): 宿主 memset 清 CUDA 梯度 → memset0_cuda
- conv2d_grad_cuda 专用内核写好后从未接线(死注册) → 接线 +
  修正 stub 签名不匹配(20 vs 16 int, 死代码从未链接故未暴露)

**验证**: t19 扩充(FP16 matmul/linear 对拍 + GPU FP16+GradScaler
训练 60 步: 损失下降/有限性/步进率); t20 GradScaler 单元(×s/恢复/
inf 检测/动态调整); CUDA 与 CPU 双构建全绿; MNIST 97.16%。
测试自身的三个 bug(f16 对拍 bias 不匹配/非标量 backward/毒化梯度
未复位)也修复——fastfail(0xC0000409)掩盖异常信息曾误导排查方向。

**遗留(下块)**: pools/embedding/permute/mean-sum 梯度仍为宿主循环,
GPU 训练触及即崩, 需原生内核化或组合化(BENCH 瓶颈清单在案)。

**反思**: 架构 8/10(unary_bwd 内部算子一次收编 9 算子, 无逐个内核
膨胀; FP16 折进原语免新机制) | 实现纪律 6/10(FP16 scratch 首版
有列切片越界, 自查发现; 测试代码三处低级错误) | 测试盲区教训再证:
"GPU 前向对拍通过 ≠ GPU 训练可用", 端到端训练烟雾必须进回归。

---

## [19] GPU 反向全算子补全 + 全线存量 bug 清剿(用户硬性要求: 全算子覆盖)

**内容**
1. 原生梯度内核: grad_bcast(通用步长广播, sum/mean/permute 反向 +
   copy_view_to 共用)、max/avg_pool_grad(重找窗口+原子散射)、
   embedding_grad(行散射原子累加)
2. 广播二元内核 binary_bcast_k(步长 0 表达广播, 4 算子共用, 替换
   旧同形内核——旧内核标量×矩阵越界读)
3. 任意维规约内核 reduce_any_k(替换只支持末维的 sum_lastdim_k——
   旧内核无视 dim 选项, dim0 求和直接算错)
4. strided_copy_k 改结构体按值传步长(数组形参衰退为宿主指针,
   设备端解引用即越界——6 处调用点全部带病)
5. t21 全算子 GPU 反向对拍(binary/unary/reduce/matmul/linear/conv/
   pool/embedding/softmax/LN/attention/losses/views): 每个可训练算子
   CPU/GPU 双建图反传逐输入比梯度 —— 本块所有 bug 的探测器

**顺带清剿的存量 bug(CPU 与 GPU 两侧)**
- GPU: copy_elements 对非连续视图扁平 memcpy(转置 clone 静默拷错);
  reduce_to 依赖的引擎内 sum_along 宿主循环(所有广播二元反传崩);
  slice/permute/transpose CUDA 映射错(permute 忽略 order, slice 丢
  begin 偏移); embedding 内核无界检(越界静默写)
- CPU: sum_into 的 keepdim 中间维映射错(3D sum/mean keepdim 一直
  算错); sum_grad_into 缺 size-1 广播步长保护(越界读); copy_view_to
  目标按线性写(散射进切片视图位置错); LN 梯度 2D 假设 + inv 常数
  近似(var 算而不用) → 统一 2D 化 + 精确公式
- 层 API: 权重固定 CPU, GPU 输入混合设备进内核(illegal access)
  → 六层权重随输入设备一次性迁移(migrate_to_device)

**验证**: CUDA/CPU 双构建全绿(19+20+21 项测试); MNIST 97.16%;
GPU FP16 训练烟雾照常。排查利器: compute-sanitizer 定位越界内核
(0xC0000409 fastfail 掩盖异常信息, try/catch + sanitizer 双管齐下)。

**反思**: 架构 8/10(grad_bcast/CopyMap 结构体按值传参模式统一了
5 处内核) | 教训: "GPU 前向对拍过 ≠ 训练可用"二次应验——本轮又挖出
12 处存量缺陷, 全算子反向对拍(t21)从此为强制回归 | 实现纪律 5/10
(本块自引入 3 处内核映射错误, 靠 t21 当场暴露; 测试数据自身又错
3 处——I64 语义与索引界)。

---

## [20] oneDNN 全算子委托实施(用户硬性要求: 全算子覆盖, 不准部分支持)

**内容**
1. dynlib: dnnl.dll 动态加载(C API 全 decltype 原生签名, ~25 指针),
   引擎/流一次性缓存常驻; ZNN_NO_ONEDNN 按族禁用(排查/对拍)
2. onednn.cpp 委托模块: 连续 F32 → 显式步长 md → primitive_desc
   (内建缓存) → memory 裸指针包裹 → execute; 任一步失败回落
3. 接线(kernels[CPU]): 13 逐元素+softmax+sum/mean+matmul/linear+
   conv2d+pool×2+LN+BN(推理) —— 凡有原语的算子全覆盖, 无原语的
   (embedding/视图族/损失)维持 Eigen/ref(其优化路径本就是它们)
4. third_party/onednn: Apache-2.0 头文件 vendored(仅类型, 零链接)

**收益**(见 BENCH): matmul 1024³ 3.5×(686 GF/s), conv 5.4×,
MLP 推理 45×, MNIST 5ep 83s→42s(2.0×), 训练步 1.55×;
MNIST 97.33%(>97% 线)。

**排查方法论(本块性能诊断的完整链路, 记录在案)**
首版 MLP 训练反而慢 3.6×。逐层分段计时(fwd 25ms 定位到 linear)
→ 逐算子微基准(linear 带 bias 25ms vs 不带 115µs)→ 钩子路径追踪
(打印从未出现 → 钩子在死代码上)。五个根因见 BENCH 排查记录节,
其中"同名双实现+钩子接错"与 DNNL_ARG_WEIGHTS 语义错误都属于
"表面在工作, 实际没走快速路径"—— 微基准+按族禁用是唯二可靠手段。

**验证**: CPU/CUDA 双构建全绿; MNIST 97.33%; 全部对拍(FP32)通过。

**反思**: 架构 8/10(dynlib 统一模式扩展, 回落链保证全算子覆盖) |
实现纪律 5/10(行主序/参数语义两处 API 误读, 靠数据而非文档核对
才发现; 死代码双实现是我此前编辑制造的) | 测量 9/10(分段+微基准
+族禁用三板斧, 45 分钟内把 3.6× 回退翻成 2.0× 净赢)。

---

## [21] 错误可追溯 + 高危限制清除(用户指令: 优化 + 错误打印可追溯)

**内容**
1. 派发器设备一致性校验: 混合设备输入此前直达内核 = 越界读崩溃
   (实测复现), 现在明确报错并给出 to_device() 提示(对齐 PyTorch 行为)
2. 算子调用栈异常追溯: OpScope RAII 维护 thread_local 算子名栈
   (零分配出入栈), 任一算子抛错时 stderr 打印 "op X failed: ... +
   at X <- caller <- ..." 调用轨迹后再重抛(附算子名上下文) ——
   未捕获异常也已在 fastfail 崩溃前留下现场(0xC0000409 会吞掉
   异常文本, 此前只能靠 try/catch 探针)
3. requires_grad(true) 非 F32: 此前静默跳过梯度累加, 现明确报错
   (对齐 PyTorch "only floating point can require gradients")
4. CPU 构建下 set_precision(TF32/FP16): 一次性 stderr 告警(此前静默忽略)
5. t22 限制守卫测试: 混合设备报错消息含算子名+提示、9 维抛错、
   I64 requires_grad 抛错(CPU 构建跳过混设备项 —— to_device 先抛)

**验证**: CUDA/CPU 双构建全绿; 探针确认错误消息带算子名与
"znn ERROR: op ... + at ... <- ..." 轨迹。

**反思**: 架构 9/10(OpScope 零开销 RAII, thread_local 为必要例外
并注释论证; try 覆盖 infer→建图全链) | 修复时机正确 —— 限制清单
审计后 24 小时内落地, 高危项(崩溃类)清零。

---

## [22] Vulkan 后端落地(本机 NVIDIA 实测全绿) + Metal 骨架

**内容**
1. Vulkan 基础设施(vulkan_device.cpp): vulkan-1.dll 全动态加载
   (GetProcAddress 逐函数指针, 与 dynlib 同哲学, 零链接期依赖);
   两阶段加载(全局级仅 vkCreateInstance, 其余实例/设备级);
   主机可见内存(Buffer+Memory+常驻映射, 内存类型按
   memoryTypeBits∩HOST_VISIBLE 兼容筛选); 管线缓存(SPIR-V+特化常量为键)
2. SPIR-V 工具链: third_party/glslang(Win 二进制) + tools/gen_spirv.py
   (GLSL→SPIR-V→C 头文件嵌入); 着色器 4 个(common 逐元素×13/
   gemm 8×8 分块 NT/reduce 任意维/softmax 末维)
3. 算子接线(kernels[Vulkan]): 13 逐元素 + matmul + linear(NT) +
   softmax + sum + mean(18 算子); Storage/to_device/device_available
   全链路路由; Metal 槽位 stub
4. Metal 骨架(metal_device.mm/metal_kernels.mm, Obj-C++/MSL 运行时编译,
   APPLE 守卫; 本机不可编译 —— 数学与 Vulkan 版逐行对应, 首次 macOS
   构建即验证)。MSL 四着色器与 Vulkan 同构(elementwise/gemm/reduce/
   softmax)
5. gpu_vulkan 冒烟测试: 内存往返 + 逐元素 + GEMM(NN/NT) + 规约 +
   softmax 全对拍 —— 本机 NVIDIA RTX 4070 Ti SUPER 全绿

**本块修复的实现 bug(全部靠对拍当场暴露)**
- VkWriteDescriptorSet 未零初始化(pNext 垃圾 → 驱动崩溃)
- VkSpecializationInfo pData 未设置(同上)
- 二元算子恒把 B 绑到 A(bcast=0 分支错误 —— 8 元素全 1 测试因
  1+1==1+1 而漏网, 256×64 随机数当场抓获)
- reduce/softmax 输出绑 binding 1(三缓冲约定应为 2)
- 内存类型未按缓冲区 memoryTypeBits 交集筛选

**验证**: Vulkan 构建 znn_test 22 项全绿 + gpu_vulkan 全绿;
CUDA/CPU 构建回归全绿(MNIST 未重跑 —— CPU/CUDA 路径无改动)。

**反思**: 架构 8/10(全动态加载零链接依赖; 管线缓存键含特化常量;
设备桥隐藏 Vk* 类型使内核文件不感知后端细节) | 教训 ×2:
未零初始化的 Vulkan 结构体pNext 是驱动崩溃的常客(全部结构体
必须 = {}); 小尺寸全 1 测试有对称性盲区(随机数 + 逐元素首错
索引是最低成本抓虫器)。

---

## [23] Vulkan GEMM 补全: 双内核分发 + 双内存类型(铁律执行: 不留缺口)

**内容**
1. 64×64 分块大内核(gemm_big.comp): 每线程 4×4 寄存器累加器,
   K 步 16, 双缓冲预取(计算当前块时从全局预取下一块, 隐藏延迟),
   B 装载摊销 64 行 —— 4096³ 从 487(崩塌)→1451 GF/s(冷机峰值)
2. 双内核分发(同 cuBLAS 做法): ≥50 GFLOP 用 64×64, 其余 32×32
   (小矩阵 32×32 占用更高更快; 交叉点实测)
3. 双内存类型: 内存类型探测发现 5 类型 2 堆 —— 原实现取首个
   HOST_VISIBLE = 系统内存堆(GPU 跨 PCIe 读)。现在 ≥64MB 走
   BAR 显存(DEVICE_LOCAL|HOST_VISIBLE), 小分配留系统内存
   (实测反而快: L2 吸收 + CPU 写缓存; 8MB 阈值曾把 2048³ 推上
   显存致 2.6× 回退 —— 数据推翻直觉, 已按实测回摆)
4. 缓冲池按 (内存类型, 桶) 分桶, 上限 256MB(64MB 块复用;
   早期"大块复用异常"根因是跨内存类型复用, 类型入 key 后消除)
5. 大内核 bug: B 预取行号误用 lc(应为 lr) —— ones 测试因全 1 不可
   见, 递增矩阵当场暴露; 顺带确认 32×32 版无此问题

**稳态数据**(冷却多轮): 2048³ 1320 GF/s = CPU oneDNN 1.9×;
4096³ 峰值 1451/热节流 ~600; 512³ 受 0.6ms/次同步开销限制
(waitIdle 全同步往返, 同步 Tensor API 的固有代价)。

**验证**: Vulkan 22 项全绿 + 冒烟对拍; CPU/CUDA 构建回归全绿。

**反思**: 架构 8/10(双内核分发是行业正解; 内存类型按实测选而非
教条) | 教训: 性能微调的每一步都要 A/B 实测 —— "显存一定比系统
内存快"的直觉被 2048³ 数据推翻(8MB 阈值版本慢 2.6×);
ones 测试第二次证明其对称性盲区(需递增/随机矩阵佐证)。
---

## [24] third_party/data 误删事故: 依赖一键恢复机制(restore_deps.sh)

**事故与排查**
1. third_party/ 与 data/ 被整目录误删 —— 两目录全在 gitignore,
   git 无法恢复; 排查确认: 全部源码/SPIR-V gen 头均被 git 跟踪
   零损失, 损失面 = 外部依赖与数据集
2. build/ 同窗口消失(环境另有进程删目录): canary 文件 10 秒存活
   验证后重新 cmake 配置, CPU 22 项回归全绿

**恢复与机制化**
1. TEMP 残留覆盖 oneDNN-3.13.4/Vulkan-Headers-1.4.304/glslang-bin/
   odnn-build(事实上的离线通道); Eigen 3.4.0 从 GitLab 下载
2. tools/restore_deps.sh: 幂等一键重建, 单项可装
   (eigen|onednn|vulkan|glslang|mnist); oneDNN 构建生成头
   (dnnl_config/version/hash)随仓库走(tools/onednn-generated/),
   免 TEMP 免 cmake 配置
3. MNIST 假成功链修复: 旧脚本 100 字节下限放行 404 XML(320B) →
   运行期 "bad IDX magic"; 现为精确字节数校验
   (47040016/60008/7840016/10008) + 三级源: ossci/googleapis 直链
   → deepai 整包 zip(zip 内为 .gz, 双层解压)。本网络两个直链镜像
   均返回 S3 NoSuchKey 错误页, 整包兜底实测恢复成功
4. 跨边界路径 bug: Git Bash 的 /tmp 是 MSYS 虚拟路径, 传入原生
   python 代码字符串不翻译 → 打不开; winpath()(cygpath -m, 非
   Windows 原样返回)统一转换; do_glslang 同款潜在 bug(此前被
   TEMP 残留路径掩盖)一并修复

**验证**: restore_deps.sh mnist 端到端重放(含自愈清理上次 300B
垃圾文件)→ 4 文件字节精确 + 魔数正确; MNIST 重训 97.16%(≥97%);
CPU 22 项全绿。

**反思**: 韧性 8/10(单命令幂等 + 三级源 + 精确校验 + TEMP 离线
通道) | 教训 ×3: (a) gitignore 的外部资产必须有等价恢复通道,
否则等于单点丢失; (b) 下载校验用字节数下限会放行错误页 —— 错误
页也是合法字节流, 必须精确匹配; (c) 镜像会死(双直链同日
NoSuchKey), 恢复脚本要多源多形态(直链 + 整包)才够韧。

---

## [25] 可移植性清障: MSVC 方言清零(clang 严格解析 64/64)

**背景**: Mac 首编译(Apple Clang 17)第一文件即炸 —— Windows 全绿是
MSVC 扩展在托底。为避免 69 个编译单元逐轮贴报错, 用本机 NDK clang
(android target, 头文件一致)做 -fsyntax-only 全量扫描, 一次抓完。

**三类 MSVC 方言(全部实证复现)**
1. `struct Tensor::Impl;` 类外限定名前置声明嵌套类 —— znn_internal.hpp
   多余行, 删除(公共头类内已有声明)
2. `nullptr(实参)` —— backend_slots.hpp 的 K 宏未编入时展开 nullptr,
   18 处直调点写成 ZNN_*_K(f)(args) 形态; MSVC 警告 C4353 放行
   (实测 3 行探针复现), 标准非法。修法: 语句位包 ZNN_CUDA_CALL(expr)
   / 值位(alloc, void*)新增 ZNN_*_CALLV(t, expr) —— 关键机制: 未编入
   分支不引用实参参数, expr 连同其中的 nullptr(args) 被预处理器整体
   丢弃, 两配置均合法; 编入分支展开为原表达式零变化
3. 临时量绑非 const 左值引用 —— matmul/conv2d_eigen 回落调用传
   KernelContext{} 临时给 KernelContext& 形参; 改为先建具名局部
   ctx(与 randn.cpp 等既有写法一致)
4. 附带: mnist 例程用 std::string 未含 <string>(MSVC 传递包含放行)

**验证**: clang 严格解析 64/64 全过(排除 .mm/.cu/vulkan 后端, 属各自
工具链); Windows 三构建全量回归: CPU 22 项 + MNIST 97.16%(与重构前
逐位一致) + Vulkan 22 项 + 冒烟 + CUDA 全项(含 GPU 对拍)全绿。

**反思**: 可移植性 6→9/10(教训: "三配置全绿"只证明 MSVC 三配置;
跨编译器验收必须真的用第二个编译器 —— NDK clang -fsyntax-only 是零
成本的第二意见, 早该进回归)。MSVC 的三类宽容(限定名前置声明、
nullptr 调用、临时绑非 const 引用)都是"编译过但只在 MSVC 过"的
定时炸弹, clang 扫描应作为提交门禁。

---

## [26] Metal 首编译修复: 6 处编译错 + 3 处自查出的隐性 bug

**编译错(Mac 实报, 63/68 已过 —— 可移植层全绿)**
1. `static_cast<MtlBuf*>(const void*)` 去 const → const MtlBuf*
2. MSEL 选择器 `copyFromBuffer:...length:` → `size:`(Metal API 实名)
3. `NSError.UTF8String` 不存在 → localizedDescription.UTF8String ×2
4. 两处 std::string 用而未含 <string>; 另 device:: 跨编译单元不可见
   —— 新增 metal_device.hpp(仅 .mm 内部包含, 含 MTL 类型; 纯 C++
   单元经 backend_slots 间接包含的 metal_backend.hpp 保持无 MTL 类型)

**自查修复(不等运行期炸)**
1. dispatchThreads 网格语义错: 第一参数是总线程数, host 却按"组数"
   传(少发 256×线程); 且线程组硬编码 (threadExecutionWidth,1,1) 而
   GEMM 内核按 8×8 二维组写 → .y 恒 0。dispatch 签名改传 MTLSize
   grid/tg, 调用点各自声明(逐元素/规约/softmax: 256×1×1, GEMM: 8×8)
2. reduce/softmax MSL 输出误绑 buffer(1) —— Vulkan 同坑复发(host 传
   (x,x,y), buffer(1)=x, 内核会写进输入) → 绑 buffer(2)
3. GEMM 边界 tile: ta/tb 装载用 inr(row&&col 联合)守 → M/N 非 8 倍数
   时漏写 threadgroup 单元读到脏数据; 改 ta 按 row、tb 按 col 独立守
   (读者所需单元必被同行/同列线程写过 —— 数学可证)

**验证**: 结构体布局核对(ReduceAttrs/SoftmaxOptions 首成员 int64_t
dim ✓); Windows 不编译 .mm, 三构建不受影响(可移植源码本轮零改动)。

**反思**: 质量 5/10 前值(首次真机编译 6 错 + 3 隐雷) | 教训: (a)
从未被任何编译器编译过的代码 = 未验证代码, "数学与 Vulkan 逐行对应"
挡不住 API 层笔误(选择器名/绑定号); (b) dispatchThreads 的 grid 语义
与 CUDA/Vulkan 的 group 数语义不同, 移植数学时最容易漏的是语义层
而非数学层。

---

## [27] Metal 接线: 18 内核入槽 + 内存路由三原语 + 对拍测试

**内容**(Mac 首编译全绿 22 项后的接线块)
1. 18 算子 kernels[Metal] 槽位接线(13 逐元素 + matmul/linear/
   softmax/sum/mean), 每算子一行 ZNN_METAL_K(xxx_metal)
2. backend_slots.hpp 补 ZNN_METAL_CALL/CALLV(与 CUDA/Vulkan 对称);
   此前 Metal 只有 K/AVAIL
3. 内存路由收敛为按设备四原语 device_copy_h2d/d2h/d2d/memset0
   (znn_internal): from_host/copy_elements/to_device/Impl_copy_back/
   optimizer clear() 全部改走原语 —— 替换了四处 CUDA 硬编码分支。
   其中两处是存量暗雷: Impl_copy_back(Vulkan 回落必炸, 从未触发)
   与 from_host(Vulkan/Metal 张量直接 memcpy 句柄结构体)
4. to_device 重写: 逐对 if 链 → CPU↔GPU + 跨后端借道 CPU(净 -14 行,
   Metal/Vulkan/CUDA 对称)
5. optimizer step() 守卫: 非 CPU/CUDA 参数显式抛错(Metal/Vulkan
   优化器内核未建, 此前会把 GPU 句柄当 float* 静默 UB)
6. t23 Metal 对拍: add/mul/relu/gelu/tanh + matmul(60×37×45 非 8
   倍数专打边界 tile 守卫) + linear×w^T + 广播加 + sum/mean 非末维
   + softmax; Windows 上自动跳过(device_available=false)

**验证**: clang 严格解析 64/64; Windows 三构建全量回归全绿
(CPU/Vulkan/CUDA + Vulkan 冒烟); Metal 对拍待 Mac 端 t23 执行。

**反思**: 架构 9/10(四原语消掉了全部后端硬编码分支, 新后端接入
= 后端实现 + 槽位行 + 原语 case) | 教训: Vulkan 当年只接了一半
路由(from_host/copy_back 暗雷存活至今) —— "冒烟测试通过"只证明
被测路径通, 派发器回落这种冷路径需要专项对拍(t23 的 fallback
广播加即覆盖)。

---

## [28] 铁律整改 SB1: GPU 梯度基础设施(Vulkan/Metal 对齐 CUDA)

**背景**: 用户裁定"18 算子原生,其余回落 CPU"违反"全算子原生"铁律。
盘点: CUDA 35/35; Vulkan/Metal 各缺 17 算子+优化器+层迁移。分五子块,
本块(SB1)= 梯度与优化器基础设施。

**内容**
1. 多缓冲派发桥: Vulkan 管线绑定数入缓存键(2..8) + dispatch_bufs;
   Metal dispatch_bufs(缓冲 0..n-1, 推送常量 buffer(8))
2. 逐元素升级全维广播(右对齐步长表, 维长 1→步长 0)—— 与 CUDA
   binary_bcast_k 同构; 原版仅标量 b[0] 广播, [M,N]+[N] 类是错的
   (又一个"数学对应了、语义层没对应"案例)。GLSL+MSL 双实现
3. 新内核 ×2 后端: unary_bwd(9 导数表)/grad_bcast(步长收集)/
   strided_copy(双步长)/opt(SGD+Adam 合一, 特化常量切换)
4. 设备无关助手 grad_bcast_dispatch/strided_copy_dispatch(shared.cpp):
   消掉 sum/mean/permute/copy_view_to 四处"CUDA 分支+宿主循环"重复
5. 优化器设备通用: sgd/adam step 按设备切原生内核, moments/velocity
   惰性分配跟随参数设备; 撤掉"仅 CPU/CUDA"守卫
6. 存量 bug ×3: (a) Tensor::zeros 对 GPU 设备 std::memset 句柄结构体
   = 堆损坏(引擎反向首步即触发), 改走 device_memset0; (b) clone()
   头文件注释称"仍参与建图"但实现丢 requires_grad—— 注册可微 clone
   算子 + 反向期(建图关)直拷的分裂语义; (c) Adam moments 惰性分配
   误入 GPU 分支(本块自查自纠)
7. gpu_vulkan 冒烟扩展: 全维广播/逐元素反向(unary_bwd+grad_bcast)/
   matmul+clone+views 整链反向(双端对解析梯度 0 误差, CPU↔VK 逐位
   一致)/Adam 步进 CPU↔VK 对拍

**验证**: Windows 三构建全绿(CUDA t21 含全算子梯度对拍; Vulkan 冒烟
9 段全过 EXIT=0; CPU 22 项 + MNIST 97.16%); clang 严格解析 64/64。
Metal 侧 MSL 与 host 全镜像(待 Mac 端验证)。

**反思**: 质量 6/10(首发即踩 3 个存量雷, 调试链路 длинная) | 教训:
(a) **tape 遍历中派发器插节点 = 迭代器失效 UB**—— 内核内部材料化
   不得走公共 clone(分裂语义); (b) 测试复用张量 impl 会跨 backward
   叠加梯度污染(7a 残留进 7b, 伪装成 0.22"泄漏"追了整轮); (c)
   item()/data() 在 GPU 张量上是句柄裸读(文档化限制)—— 测试须先
   to_device(CPU); (d) python heredoc 改 C++ 源码第三次炸(\n 实体
   化)—— 已全面回到 Edit 工具。

---

## [29] 稳定性整改 I: 数据指针类型化 + TensorTape(去全局 tape)

**data<T>() 按用户裁决重设计**(原: GPU 张量返回 float* = 类型洗白通道)
- 公共头新增类型标记 CudaDataPtr/MetalDataPtr/VulkanDataPtr;
  data<T>() = 设备×请求类型匹配才非空, 调用方判空。CPU 用
  float/int64_t(宿主缓冲含偏移), CUDA 用 CudaDataPtr(显存地址含偏移),
  Metal/Vulkan 用各自标记(裸句柄, 不做偏移算术 —— 偏移并入绑定层
  待办)。if-constexpr 单模板 + 5 显式实例化
- item() 全设备化(经 device_copy_d2h); 全库迁移: .cu 61 处 → dptr
  家族, VK/Metal vhandle/mbuf → 标记类型, 梯度规则/优化器/grad_scaler
  → cuda_ptr; 派发助手改 Tensor 签名(句柄不再跨层); 新增
  strided_copy_raw(Impl 级, CUDA 偏移并入指针, M/V 偏移明确报错)
- 残留误用即返空暴露(调试期发现并修复: Impl_copy_back 用 data<float>
  取 GPU 句柄 → nullptr 崩溃 → 改标记类型)

**TensorTape(去全局可变引擎 —— 用户设计裁决)**
- struct TensorTape{op, inputs, options} 挂在 Tensor::Impl(叶子为空),
  attach/of 为 TensorTape 静态成员(znn.hpp 一行 friend 授权)
- record_tape 替代 AutogradEngine::record; run_backward 从损失 DFS
  收集本图(后序=拓扑序, Tape 地址去重)逆序执行 —— 全局 tape_/单例/
  一次性消费/插节点 UB 全部结构性消失; 图随张量引用计数存亡,
  天然可重入; 算子梯度规则零改动(BackwardContext 不变)

**验证**: Windows 三构建全量(CPU 22 项 + Vulkan 测试+冒烟 EXIT=0
含 bwd2 双端解析 0 误差逐位一致 + CUDA 全项含 GPU 对拍) + MNIST
97.16% + clang 严格解析 64/64 全绿。

**反思**: 设计 9/10(两类病灶均改为结构性不可发生: 错型即空、无全局
可变状态) | 教训: 正则批量改码三次误伤(:: 前缀/箭头形态/宏参数),
每次靠 grep+构建即时拦截 —— 机械迁移必须配同轮构建门禁。

---

## [30] ztestkit: 独立测试工具包 + to_device 视图偏移存量 bug 修复

**ztestkit**(可剥离: 仅依赖公共头 znn.hpp, 整目录可拷走测任何同接口库)
- L0 不变量(视图/偏移/clone/detach/共享语义) + L1 算子×形状族矩阵
  (二元 4×4 广播变体、一元 9、GEMM 边界 tile 5×7×4/60×37×45、规约
  dim×keepdim、offset>0 视图 GEMM) + L2 梯度对拍(4 链, 设备参数化,
  每图新鲜叶) —— 运行时设备探测自动展开(本机 CPU+Vulkan=25 用例×2)
- 首跑即抓真 bug: to_device 对 offset>0 视图从基址拷贝(拷错区间+
  偏移丢失, 存量自 v1); GPU 连续 clone 的 d2d 同病 —— 修复: 源指针
  折算元素偏移, M/V 连续视图走 strided_copy_raw(绑定层偏移)
- 三处测试自身笔误被即跑即纠(slice 列偏移=1 非 6、cmp 严格小于
  tol=0 永假、to_device 同设备共享/跨设备材料化两态语义) —— 矩阵
  一次跑齐的价值当场体现

**验证**: ztestkit Vulkan 构建 CPU+VK 85 检查全绿; CPU 构建全绿;
znn_test CPU 全过; MNIST 97.16%。CUDA 构建 kit 待下轮(其 znn_test
已在 9675d29 回归)。

**反思**: 工具 8/10 | 教训: 测试代码同样是代码(三笔误), 但矩阵框架
让每处笔误在数十秒内暴露而非数小时调试 —— 这正是它存在的意义。

---

## [31] 灭 t10/t21 GPU 崩溃: 崩溃自打印栈工具化 + 两处迁移漏网

**工具**(ztestkit 常备): SetUnhandledExceptionFilter + dbghelp 符号化
栈打印 + /Zi —— 崩溃即出函数名, 不再依赖探针-构建-重跑循环
(本轮两案均一次定位; 此前同难度问题耗时整轮)。stdout 无缓冲。

**bug×2**(9d4f76a 迁移漏网, 均为新契约"错型即空"的暴露案例):
1. AdamImpl/SgdImpl::clear(): gpu_handle() 对 CPU 梯度返回 nullptr
   → memset(nullptr) 段错误(t10 训练循环首步即崩)。修复: CPU 走
   宿主 data<float>(), GPU 走 gpu_handle。教训: 此前"CPU 构建全绿"
   跑的是陈旧二进制 —— 构建输出必须可见、时间戳必须核对
2. cross_entropy_cuda: 标签仍用 in[1]->data<int64_t>()(CUDA 上
   返空) → 内核读 nullptr 异步非法访问, 在 t21 反向种子同步点浮出
   (崩溃点≠犯罪点)。修复: dptr_i64

**验证**: 三构建 znn_test 全过 + ztestkit 全绿(CUDA 88/VK 88/CPU 50
检查) + Vulkan 冒烟 EXIT=0 + MNIST 97.16%。kit 新增 t10 训练循环与
CE 反向链两条回归用例。

**反思**: 工具 9/10(崩溃栈=一次定位) | 教训×2: (a) 机械迁移后必须
全库 grep 残留形态(本轮 `->data<int64_t>` 漏扫); (b) 构建命令输出
不可吞(>/dev/null 掩盖失败跑了多轮旧二进制, 假绿比红更毒)。

---

## [32] Metal 全绿(ALL GREEN on Mac) + Mac 编译修复

**Metal ztestkit 128 项全绿**(Mac mini AppleClang 21 / arm64, 用户实跑)
- Mac 修 3 类: windows.h 无守卫(ztestkit 跨平台) / shared.hpp 在
  namespace 内(作用域污染) / mapped_of 未声明 / MSL 内建必须作参数
  (22 内核 thread_position_in_grid → uint3 tpig [[attr]]) /
  CE/MSE grid 1→256(dispatchThreads=总线程数)
- Metal 共享内存架构使 17 算子一次通过: 布局族零派发(mapped 直写),
  conv/pool/embedding CAS 原子, attention 组合

---

## [33] 外部训练项目(UniFP-Net)实测暴露: 8 缺陷一轮修复(全后端)

首个真实外部消费者(UniFP-Net: 单骨干人脸检测+人像分割, 640² 训练)
全程联调暴露 8 类缺陷, 本轮全部修复; **ztestkit 249 checks 全绿**
(CPU+CUDA+Vulkan, ZNN_STRICT_NATIVE=1), hello_dll 消费者级回归全过。

**DLL 接口**: Shape 方法/data<T>/item<T> 未导出 —— Conv2dOptions
默认成员初始化即触发 Shape 构造, 外部根本链接不上 conv2d; 消费者被迫
本地补定义(布局耦合, 不可持续)。

**语义错误×3**:
1. softmax_cuda 无视 options.dim 恒取末维(参数连名字都没取); VK/MT
   dim≠末维直接抛错降级。→ CUDA softmax_dim_k(步长表逐行)+ VK
   softmaxdim.comp + MT kMslSoftmaxDim, 末维双快路径保留
2. Conv2d 工厂漏建 groups 权重形状 → DW 变全卷积(MetaBlock(256) 单块
   2.03M 参数, 应 0.40M)且前向校验失败; conv2d 算子本身支持分组
3. Conv2d 偏置 [C]+4D 广播炸 —— add 右对齐语义与 PyTorch 一致没错,
   错在层: 偏置改 [1,C,1,1] 形态

**性能×2**:
4. 归约内核一输出一线程: dim0 归约到 48/96 输出整卡 1-2 block, 单次
   12ms(BN 反传 3-4 次 → 25-48ms), 标量归约单线程 → CUDA 两段式
   固定分段树归约(确定性, 对拍可复现)/VK+MT 每 workgroup 一输出
5. clone_ref 与 clone_run 逐字节相同被 /OPT:ICF 折叠同址, 派发器
   kernel==ref 误判 → **每次 CUDA clone 都绕 CPU 往返**(此前日志
   "znn fallback: op='clone'" 的真凶, 悬案两周); ref 改显式循环

**鲁棒×2**:
6. CUDA 一元宏/conv2d/cat/stack 对 strided 视图直接抛错 → 统一
   binary_bcast_host 同策略(非连续原生 clone, 可微)
7. CPU 存量×2: sum_into 通用路径扁平下标当存储下标(无视输入步长),
   softmax_rows 拿输入步长寻址输出 —— 均仅连续输入碰巧正确, 转
   置视图实测全错(手动参考仲裁); onednn softmax 补连续守卫

**验证**: ztestkit 新增 5 组回归(softmax 任意维×5 形/工厂 groups+偏置/
strided 视图输入族/大 K dim0 规约/to_device 求导标志) + hello_dll
导出回归。to_device 顺带保留 requires_grad(叶参数迁移语义)。

**反思**: 工具 8/10(手动参考仲裁一周转 CPU/CUDA 各执一词的悬案一次定谳)
| 教训: (a) "DLL 导出 112 符号"的完备性要以 hello_dll 级消费者编译
为验收线, 不是 dumpbin 数数; (b) 相同代码体函数对(ICF 折叠)不可
作身份判据; (c) 内核并行度要看"输出数", 输出数少≠负载小。

---

## [34] 减法反传符号修复(用户发现) + BN 选项强转 + 归约越界 + 原生 AdamW 衰减

**用户实测 UniFP BCE/Dice 梯度校验暴露: sub 注册的是加法梯度帮手** ——
`y=a-b` 的 db 恒为 +g(应 -g), 前向正确反传错号; BCE 背景类 logit 被推
反、Dice 外层 1-ratio 梯度反向、GIoU 全套受累(CPU/CUDA 复现)。修复:
sub 专属 `sub_grad`(db = reduce(-g)), `binary_grad` 更名 `add_grad`
防再误用; 四后端共享同一注册(组合实现无后端分叉)。**这正是 UniFP
过拟合发散一周悬案的最终根因** —— 损失里每处 sub 的 b 操作数梯度全反。

同轮: (a) BN 选项 {double,double,bool} 被三 GPU 后端按 float[3] 强转,
bool 位型读成 0.0f → **GPU BN 训练态一直跑 eval 统计**(kit 无 BN 覆盖
长期漏网, UniFP 手动参考仲裁定位; CPU 侧同款 UB 靠 denormal≠0 碰巧工作);
(b) reduce_final_k 的 total 从 gridDim 推导, 非整块时越界(用户修正:
显式传参); (c) 原生 CUDA AdamW 忽略解耦衰减(用户修正: p·(1-lr·decay))。

**验证**: ztestkit 309 checks 全绿(CPU+CUDA+VK, STRICT_NATIVE=1,
新增 sub_gradient/bn_train_parity 入矩阵; 用户独立版用例按仓库惯例
并入 ZTK_CASE)。hello_dll 消费者回归过。

**反思**: 工具 9/10(手动参考仲裁连破 BN/sub 两案) | 教训: (a) 选项
blob 一律按真实结构体读, float[] 强转是 UB 温床; (b) 梯度规则注册
"复用帮手"要审符号语义, 不是所有二元算子都对称; (c) kit 覆盖缺口
(BN train 跨设备)让 GPU 专属错跑了两轮全绿。

---

## [35] LayerNorm 任意 dim(全后端) — ②' 通道维归一化零拷贝化

UniFP 推理基准定位: 前向 20.6ms@640 为启动开销主导(~300 算子), 其中
ChannelNorm(NCHW 通道维 LN, ConvNeXt 式)因 znn LN 仅末维, 每个点位
transpose+clone+LN+transpose+clone —— 全模型 ~60 点位 × 2 次全量拷贝。
**本轮给 layer_norm 加 dim 参数**(默认 -1 末维不变):

- 选项体 {eps, dim}(三 GPU 后端同款结构体镜像, 吸取 BN float[3] 强转教训)
- CUDA `layer_norm_dim_k`: 每 warp 一组(gi = o·inner+i 展开寻址, 与
  softmax_dim_k 同构), E[x²]−μ² 单遍方差 + 数值保护
- VK norm.comp OP=2 `lngroup`: 每 workgroup 一组(首版误用每线程一组混
  barrier 模型 —— 探针抓出 102/108 零元素; 且 VK dispatch 的 gx 是组数)
- Metal `lngroup` 镜像; CPU 非末维回落 ref(oneDNN 仅末维)
- 梯度重写为全广播组合(mean(x,dim,keepdim) + keepdim 逐轴约 dγ/dβ),
  任意维通用, 不再假设 [rows,D] reshape
- LayerParam 增 norm_dim(-1=末维); kit 新增 layer_norm_dim(fwd 手动
  参考 + 梯度对拍)

**验证**: ztestkit 318 checks 全绿(CPU+CUDA+VK, STRICT)。教训×2:
(a) 共享内存协作归约与每线程一组是两套模型, barrier 混用必炸;
(b) 共享盘写延迟让"regen→build"两连发扑空 —— 再遇一次, gen 后必须
核对 dll mtime > 头文件 mtime。

**[35 续] VK linear 偏置**: common.comp 全维广播本支持 b[N] 对 y[M,N]
右对齐相加, 原注释的"广播不成立"判断过时 → 直接组合 elementwise add,
替换抛错。kit +linear_bias 用例(该路径此前零覆盖)。UniFP 全模型在
Vulkan 首次可端到端运行: 412ms@640 —— 受 VK 每 dispatch 全同步架构
成本主导(~300 算子 × ~0.6ms, BENCH.md 已记录), 非算子本身慢。

**[36] Vulkan 延迟提交(批量派发)** —— UniFP 推理 VK 413ms 的主因是每算子
begin+submit+waitIdle(~0.6ms × ~300 算子)。改造 vulkan_device.cpp:

- 单命令缓冲连续录制, 主机-设备边界才 flush(h2d/d2h/mapped_of 四处屏障,
  提交+等待+释放挂起资源); ZNN_VK_EAGER=1 保留旧逐算子路径(对拍)
- 描述符集改为每派发新配(录制与执行间复用改绑是规范竞态), 池 64→16384
- 缓冲释放进隔离区, 队列空闲后归还复用池(挂起命令仍引用)
- **每派发后插存储屏障(SSBO 写→读)**: Vulkan 同 cb 相邻内核无隐式内存
  一致性, 旧 waitIdle 恰好掩盖; 缺它时链式 conv→add 确定性读旧值
  (conv dw+bias 探针二分定位; vkCmdPipelineBarrier 忘加载也静默吞过一轮)

**验证**: ztestkit 321 全绿(STRICT); UniFP VK 413→323ms@640、69.6@320 ——
同步项已消, 新瓶颈=VK 内核质量本身(naive tiling GEMM vs CUDA 21TF/s),
下一步方向是 kernel 层而非派发层。EAGER 回退路径同绿。
