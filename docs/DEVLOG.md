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
