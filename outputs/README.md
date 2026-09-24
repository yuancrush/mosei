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
| `exp015_multiseed_experts/` | 当前正式任务专家集成结果 |
| `exp016_attachment3_predictions/` | 附件3无标签推理候选，尚未冻结 |

## 只作实验追溯

- 名称含 `_smoke` 的目录只证明代码能够运行，训练轮数和样本范围不构成论文结果。
- `exp001`至`exp019`的普通目录保存结构筛选、损失、门控和消融证据。论文只引用对应实验记录明确保留的结果。
- `best.pt`、TensorBoard日志和逐样本预测用于复现与排错，不直接作为论文表格。
- 问题2、问题3的最终目录将在模型冻结后写入最新增量交接；在此之前不得自行从多个输出中挑选最优数字。

