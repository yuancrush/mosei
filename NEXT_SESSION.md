# 跨会话实验交接

## 新会话开始时必读

1. `README.md`：实验协议、评价指标和目录结构。
2. `experiments/README.md`：实验索引与当前状态。
3. 最新一个 `experiments/EXP-*.md`：上一轮结果及下一步依据。
4. 对应的 `outputs/<run_name>/summary.csv` 和 `history.json`：原始指标。

## 当前研究状态

- 数据工程与四个最小基线已完成，官方划分无视频重叠。
- 文本是主模态，简单三模态拼接没有稳定超过文本。
- EXP-001 已完成：Text-BiGRU 相对 Text-Mean 在验证集的 Macro-F1、MAE、Pearson 均改善。
- EXP-002 已完成：Audio 改善回归但损害分类，Vision 三项核心指标均退化。
- EXP-003 已完成：动态门控仍未超过 Text-BiGRU，视觉、音频融合和三模态方案暂缓。
- EXP-004 已完成：dropout=0.5 三项核心指标均退化，保留 dropout=0.3。
- EXP-005 已完成：Text-BiGRU 的 Macro-F1、Pearson 在 3/3 种子上改善，已冻结为候选主模型。
- EXP-006 已完成：Neutral 明显过预测，正负情感回归预测向 0 收缩。
- EXP-007 已完成：Neutral precision 略升但 recall 大幅下降，恢复完全逆频率权重。
- EXP-008 已完成：MSE 的三项核心指标均退化，保留 Smooth L1。
- EXP-009 已完成：附件3的30个样本文本完整，音频/视觉均局部缺失，29个样本二者缺失位置相同；主压力测试应采用联合音视频局部块缺失。
- EXP-010 已完成：轻量来源/判别不确定性融合显著优于普通融合，但未超过 Text-BiGRU。
- EXP-011 已完成：seed 2026 下 Macro-F1/MAE/Pearson 为0.6270/0.5821/0.6523，联合音视频中段缺失30%时 Macro-F1为0.6211；三种子均值为0.6139±0.0151/0.5899±0.0132/0.6524±0.0044，分类波动较大。
- 当前总体候选仍为 Text-BiGRU；问题2的机制候选为 EXP-011 不确定性融合。完整 HyperDiff/扩散暂缓。
- EXP-012 已完成：分类使用 Text-BiGRU、回归使用 Text+Audio-BiGRU 时，验证集为 Macro-F1 0.6351、MAE 0.5697、Pearson 0.6693、综合分数1.1146，优于单一 Text-BiGRU。
- 完整阶段总结见 `reports/第1天实验阶段总结.md`。
- 日常迭代使用 `--validation-only`，不要根据测试集结果选择结构或超参数。

## 固定实验协议

- 数据：官方 train/valid/test。
- 默认 seed：2026；筛选阶段单种子，候选模型再运行 3 个种子。
- 默认训练：batch size 64、最多 50 epoch、patience 8。
- 核心指标：Macro-F1、MAE、Pearson；综合分数仅用于早停。
- 所有正式实验必须更新实验 Markdown、实验索引和本交接文件。

## 下一决策

1. 补齐 Text+Audio-BiGRU 回归专家的 seed 2027/2028，验证 EXP-012 的回归增益是否稳定。
2. 建立附件3推理脚本：由 `text_bert` 生成768维 BERT特征，复用训练集标准化参数，并输出分类/回归结果及模态覆盖率；先做结构和可复现性检查，不窥视测试标签。
3. 若判别不确定性继续研究，应增加校准指标/损失；当前 vacuity 对缺失不敏感，不能宣称已经实现完整 DCEF。
4. 只有轻量重建具有可量化增益时才考虑扩散；附件3文本完整且仅30例，完整超图扩散优先级低。
