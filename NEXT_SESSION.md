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
- 第1天的模型调参已停止，当前冻结候选为 Text-BiGRU、dropout=0.3、完全逆频率类别权重、Smooth L1。
- 完整阶段总结见 `reports/第1天实验阶段总结.md`。
- 日常迭代使用 `--validation-only`，不要根据测试集结果选择结构或超参数。

## 固定实验协议

- 数据：官方 train/valid/test。
- 默认 seed：2026；筛选阶段单种子，候选模型再运行 3 个种子。
- 默认训练：batch size 64、最多 50 epoch、patience 8。
- 核心指标：Macro-F1、MAE、Pearson；综合分数仅用于早停。
- 所有正式实验必须更新实验 Markdown、实验索引和本交接文件。

## 下一决策

1. 阅读 `reports/第1天实验阶段总结.md` 和 `outputs/exp006_error_analysis/hard_cases.csv`。
2. 将稳定困难样本与原始视频/标签、序列长度或可解释专项附件关联，先提出可验证的错误成因。
3. 不要直接继续调 dropout、类别权重或回归损失；这些方向第1天已经得到负结果。
