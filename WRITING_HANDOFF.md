# 问题1写作与跨会话交接

## 给合作者的结论

问题1工程交付已经冻结，可以开始写作。写作人员应先从 `reports/README.md` 和 `reports/writing_handoffs/README.md` 进入，不能把本文件单独当成最终口径。新的关键边界是：100条均有完整特征；78条未触发自动低置信阈值；22条低置信样本经人工视听复核后，2条仅部分匹配、16条与官方文本不一致、4条无有效语音。不得继续沿用“80条词级时间映射可用”或“100条全部精确对齐”的旧口径。

推荐阅读顺序：

1. `reports/问题1完整求解与写作材料.md`：题干拆解、模型公式、最终结果与写作边界；
2. `outputs/question1_submission/问题1_全量特征汇总.csv`：100条×3模态的论文表格原始数据；
3. `outputs/question1_submission/典型样本对齐验证.png`：正文典型样本图；
4. `outputs/question1_quality/quality_overview.png`：中文质量统计图；
5. `experiments/Q1-STAGE2_视觉回退与接口复验.md`：优化实验过程；
6. `outputs/question1_submission/问题1_可复现提交包.zip`：最终附件候选。
7. `reports/question1_stage_a/低对齐置信度样本ASR复核.csv`：22条低MMS分样本复核明细。

## 可直接交给新Codex会话的提示词

```text
请先完整阅读pythonproject/WRITING_HANDOFF.md和
pythonproject/reports/问题1完整求解与写作材料.md，再依据其中的最终数字撰写论文问题1章节。
写作结构为：问题重述、符号与假设、数据预处理、三模态特征定义、词级物理时间对齐、
双层输出、质量控制、结果与典型样本、局限性。图表必须使用已有中文图；不得把50写成
等长时间片，不得把34条质量预警写成失败，不得把20条低可靠候选区间写成可靠时间戳，
不得声称FaceMesh等同FACET，不得使用外部
情感数据或无标签伪标签。引用数字前核对pythonproject/outputs/question1_quality/summary.json。
```

## 写作中的固定口径

- “66条高置信、34条触发质量预警”，不能写“只有66条成功”。
- “78条未触发自动低置信阈值；22条低置信样本经人工视听复核后，2条仅部分匹配、16条与官方文本不一致、4条无有效语音”。2条部分匹配样本在没有补充精确子句起止时间前，也不能计入完整词级时间映射。
- “6条仅固定50位置兼容层截断，完整词级主层未截断”。
- 特征是BERT 768维、openSMILE 25维、FaceMesh 35维。
- 时间轴是原视频秒级物理时间，三模态按强制对齐词区间统一。
- 剩余真实无脸场景以0和mask表示，不伪造面部观测。
- 18条重合样本的BERT三类输入18/18一致，完整50位置特征RMSE为6.22×10⁻⁷；padding文本非零但必须由attention mask排除。
- 视频总时长与文本、音频、视觉有效时长是四个不同字段，禁止统称“原始有效时长”。

## 脚本导航

| 脚本 | 作用 | 是否修改官方数据 |
|---|---|---|
| `audit_attachment1.py` | 审计100条标签、视频、媒体流和附件2重合关系 | 否 |
| `extract_question1_features.py` | 完整提取三模态特征并建立词级物理时间轴 | 否 |
| `upgrade_question1_visual.py` | 只重算小脸/侧脸视觉回退，便于消融复验 | 否 |
| `finalize_question1_stage_a.py` | 恢复BERT padding接口并补充四类时长 | 否 |
| `review_question1_alignment.py` | 用Whisper独立复核22条低对齐置信度样本 | 否 |
| `analyze_question1_quality.py` | 校验NPZ并生成中文质量图和预警清单 | 否 |
| `validate_question1_reference.py` | 在18条重合样本上独立核验兼容接口 | 否 |
| `prepare_question1_submission.py` | 生成全量表、典型图、说明和压缩包 | 否 |

## 复现顺序

```powershell
conda activate mosei
python audit_attachment1.py
python extract_question1_features.py --output-dir outputs/question1_features
python upgrade_question1_visual.py
python finalize_question1_stage_a.py
python review_question1_alignment.py
python analyze_question1_quality.py
python validate_question1_reference.py
python prepare_question1_submission.py
```

首次复现会下载公开预训练权重；它们仅用于特征提取与时间对齐，不属于外部情感数据集。
