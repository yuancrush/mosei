# 输出目录使用说明

`outputs/`保存原始实验产物。是否保留由复现价值决定，不等于都可以直接写进论文。

## 写作优先使用

| 目录 | 用途 |
|---|---|
| `question1_submission/` | 问题1全量汇总、典型图和复现包 |
| `question1_quality/` | 问题1中文质量图与逐项校验 |
| `exp005_multiseed/` | Text-BiGRU多随机种子稳定性 |
| `exp006_error_analysis/` | 分类与回归错误分析 |
| `exp009_missing_stress/` | 局部缺失压力测试基线 |
| `exp011_uncertainty_aug_missing_stress/` | 三模态鲁棒机制压力测试 |
| `exp020c_multiseed_experts/` | 问题2正式三模态分类集成结果 |
| `exp021_ensemble_missing_stress/` | 问题2正式集成缺失压力测试 |
| `exp020c_attachment3_predictions/` | 问题2冻结后的附件3无标签推理 |
| `question3_stage_a_timeline/` | 问题3附件4词、token及未对齐特征的物理时间索引 |

## 只作实验追溯

- 名称含 `_smoke` 的目录只证明代码能够运行，训练轮数和样本范围不构成论文结果。
- `exp001`至`exp019`的普通目录保存结构筛选、损失、门控和消融证据。论文只引用对应实验记录明确保留的结果。
- `best.pt`、TensorBoard日志和逐样本预测用于复现与排错，不直接作为论文表格。
- 问题2已冻结，以Q2增量交接列出的正式目录为准。问题3尚未冻结；不得自行从多个解释候选中挑选案例或结论。
