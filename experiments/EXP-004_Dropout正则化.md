# EXP-004：Text-BiGRU Dropout 正则化

## 实验目的

EXP-001～003 均出现训练损失持续下降而验证损失在第 3～5 轮后上升。本实验检验提高 dropout 能否缓解过拟合并提高验证集泛化性能。

## 对照设计

- 对照组：Text-BiGRU，dropout=0.3。
- 实验组：Text-BiGRU，dropout=0.5。
- 唯一变量：dropout。
- 其余数据、结构、优化器、损失、seed 和早停规则完全不变。
- 本实验只查看验证集。

## 运行命令

```powershell
python train_baselines.py --models text_bigru --run-name exp004_text_bigru_drop05 --dropout 0.5 --epochs 50 --patience 8 --batch-size 64 --validation-only
```

## 验证集结果

| 模型 | Macro-F1 ↑ | MAE ↓ | Pearson ↑ | Selection score ↑ | 最佳轮次 |
|---|---:|---:|---:|---:|---:|
| Text-BiGRU，dropout=0.3 | **0.6351** | **0.5753** | **0.6629** | **1.1063** | 5 |
| Text-BiGRU，dropout=0.5 | 0.6095 | 0.5896 | 0.6546 | 1.0676 | 5 |

原始产物：`outputs/exp004_text_bigru_drop05/`

## 结果分析与结论

- 提高 dropout 后三个核心指标均退化，最佳轮次仍为 5，没有推迟过拟合拐点。
- 当前过拟合不能通过简单增大 dropout 解决；继续围绕单个验证集调正则强度的收益较低，且会增加验证集过拟合风险。
- 保留 dropout=0.3。下一步不再微调，而是用额外随机种子同时复现 Text-Mean 与 Text-BiGRU，检验时序增益是否稳健。
