# 输出目录使用说明

`outputs/`保存原始实验产物。是否保留由复现价值决定，不等于都可以直接写进论文。

## 写作优先使用

| 目录 | 用途 |
|---|---|
| `question1_submission/` | 问题1全量汇总和典型图；唯一压缩附件位于`../submission/` |
| `question1_quality/` | 问题1中文质量图与逐项校验 |
| `exp005_multiseed/` | Text-BiGRU多随机种子稳定性 |
| `exp006_error_analysis/` | 分类与回归错误分析 |
| `exp009_missing_stress/` | 局部缺失压力测试基线 |
| `exp011_uncertainty_aug_missing_stress/` | 三模态鲁棒机制压力测试 |
| `exp020c_multiseed_experts/` | 问题2正式三模态分类集成结果 |
| `exp021_ensemble_missing_stress/` | 问题2正式集成缺失压力测试 |
| `exp020c_attachment3_predictions/` | 问题2冻结后的附件3无标签推理 |
| `question3_stage_a_timeline/` | 问题3附件4词、token及未对齐特征的物理时间索引 |
| `exp023_explanation_validation*/` | 问题3验证集多尺度忠实性与稳定性实验 |
| `exp024_attachment4_explanations/` | 问题3附件4正式预测、局部证据与关键帧 |

## 只作实验追溯

- 历史名称含`_smoke`的目录已全部清理，不得重新放入提交附件。
- `exp001`至`exp019`的普通目录保存结构筛选、损失、门控和消融证据。论文只引用对应实验记录明确保留的结果。
- `best.pt`、TensorBoard日志和逐样本预测用于复现与排错，不直接作为论文表格。
- 问题2、问题3均已冻结。问题3只使用EXP-023验证集选定的分类5位置/回归10位置协议；不得根据附件4案例观感另挑解释方案。
- `exp016_attachment3_predictions`旧目录已删除；附件3正式结果唯一入口为`exp020c_attachment3_predictions/`。
- EXP-019淘汰的BERT微调权重已删除，只保留其指标和训练日志用于失败分支追溯。
