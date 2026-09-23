# EXP-003：Text+Audio 动态门控

## 实验目的

EXP-002 表明 Audio 能改善回归，却损害分类。本实验检验样本级向量门控能否抑制不可靠音频，兼顾 Text-BiGRU 的分类能力与音频的回归增益。

## 对照设计

- 对照组一：Text-BiGRU。
- 对照组二：Text+Audio-BiGRU 直接拼接。
- 实验组：以文本为锚点，通过 `sigmoid([text; audio])` 生成逐维音频门控，再做残差融合。
- 固定条件：编码器、损失、训练参数、划分和 seed=2026 不变。
- 本实验只查看验证集。

## 运行命令

```powershell
python train_baselines.py --models text_audio_gated_bigru --run-name exp003_text_audio_gate --epochs 50 --patience 8 --batch-size 64 --validation-only
```

## 验证集结果

| 模型 | Macro-F1 ↑ | MAE ↓ | Pearson ↑ | Selection score ↑ | 最佳轮次 |
|---|---:|---:|---:|---:|---:|
| Text-BiGRU | **0.6351** | **0.5753** | **0.6629** | **1.1063** | 5 |
| Text+Audio 直接拼接 | 0.6177 | 0.5697 | 0.6693 | 1.0972 | 3 |
| Text+Audio 动态门控 | 0.6213 | 0.5782 | 0.6609 | 1.0895 | 3 |

原始产物：`outputs/exp003_text_audio_gate/`

## 结果分析

- 门控相对直接拼接将 Macro-F1 提高 0.0036，但仍比 Text-BiGRU 低 0.0138。
- 门控未保留直接拼接的回归增益，MAE 和 Pearson 都略差于 Text-BiGRU。
- 最佳轮次仍为 3，随后验证损失上升，说明门控没有解决小样本下的过拟合。

## 结论与引导意义

- 当前音频互补增益不够稳定，停止继续增加融合复杂度，视觉和三模态路线继续暂缓。
- 保留 Text-BiGRU 为当前候选主模型；下一步只调整 dropout，直接处理反复出现的过拟合。
