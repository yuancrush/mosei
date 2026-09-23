# EXP-001：Text-BiGRU 时序编码

## 实验目的

验证均值池化是否丢失了关键上下文和词序信息。该问题直接由 EXP-000 中文本主导、模型过早饱和的现象引出。

## 对照设计

- 对照组：Text 掩码均值池化基线。
- 实验组：线性投影 + 双向 GRU + 掩码注意力池化。
- 唯一主要变量：文本序列编码方式。
- 固定条件：官方划分、hidden=128、dropout=0.3、AdamW、双任务损失、seed=2026。
- 本实验只查看验证集。

## 运行命令

```powershell
python train_baselines.py --models text_bigru --run-name exp001_text_bigru --epochs 50 --patience 8 --batch-size 64 --validation-only
```

## 验证集结果

| 模型 | Macro-F1 ↑ | MAE ↓ | Pearson ↑ | Selection score ↑ | 最佳轮次 |
|---|---:|---:|---:|---:|---:|
| Text-Mean | 0.6035 | 0.6044 | 0.6227 | 1.0247 | 3 |
| Text-BiGRU | **0.6351** | **0.5753** | **0.6629** | **1.1062** | 5 |

原始产物：`outputs/exp001_text_bigru/`

## 结果分析

- Macro-F1 提升 0.0316，说明时序表示改善了三类情感的均衡识别。
- MAE 降低 0.0291，Pearson 提升 0.0402，分类与回归同时改善。
- 最优点出现在第 5 轮，此后训练损失继续下降而验证损失快速上升，仍存在明显过拟合。
- 三项核心指标方向一致，因此可以支持“时序建模有效”，不属于综合指标掩盖单项退化。

## 结论与引导意义

- 保留 BiGRU + 注意力池化作为后续时序基座。
- 下一步分别加入 Audio 和 Vision，判断互补模态；暂不直接假设三模态一定更好。
