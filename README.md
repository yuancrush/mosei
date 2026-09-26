# CMU-MOSEI 多模态情感分析实验工程

本目录用于竞赛期间的数据审计、模型训练、消融实验和结果留档。核心原则是：**一次实验只回答一个问题，以验证集指导迭代，测试集只用于阶段性冻结模型的最终报告。**

## 1. 环境与入口

### 合作者放置数据

仓库包含代码、实验记录、结果和 `*.pt` 权重，但不包含原始数据。推荐把仓库和数据目录放在同一个父目录中：

```text
工作目录/
├─ pythonproject/                         # 本仓库
└─ E题数据/
   └─ E题数据/
      └─ 附件2-数据集特征文件/
         ├─ aligned_50.pkl                # 当前实验必需
         └─ unaligned_50.pkl              # 当前基线暂不使用
```

程序会从 `pythonproject` 的父目录递归查找唯一的 `aligned_50.pkl`。如果数据放在其他位置，可显式指定：

```powershell
python audit_data.py --data "D:\你的数据目录\aligned_50.pkl"
python train_baselines.py --data "D:\你的数据目录\aligned_50.pkl" --models text_bigru --validation-only
```

第一问还需要放置附件1：

```text
工作目录/
├─ pythonproject/
└─ E题数据/E题数据/
   ├─ 附件1-数据集原始多模态样本/
   │  └─ MOSEI数据集部分原始视频-100条/
   │     ├─ label-100.xlsx
   │     └─ <video_id>/<clip_id>.mp4
   └─ 附件2-数据集特征文件/aligned_50.pkl
```

第一问完整复现命令为：

```powershell
python audit_attachment1.py
python extract_question1_features.py --output-dir outputs/question1_features
python upgrade_question1_visual.py
python finalize_question1_stage_a.py
python review_question1_alignment.py
python analyze_question1_quality.py
python validate_question1_reference.py
python prepare_question1_submission.py
```

问题3阶段A的附件4审计与时间轴复现命令为：

```powershell
python audit_attachment4.py
python align_attachment4_timeline.py
python review_attachment4_alignment.py
python summarize_question3_stage_a.py
```

阶段A已发现必须在解释阶段单独降级的样本：15号禁用逐词精确秒数，13号不能声称视觉影响正式模型，07/18号只解释进入50-token接口的文本范围。完整策略见[`reports/第3问阶段A_数据审计与时间轴映射.md`](reports/第3问阶段A_数据审计与时间轴映射.md)。

问题1工程产物已完成。100条均有完整特征；人工复核后采用更严格口径：78条自动高置信词级映射、2条仅部分匹配、20条不支持完整词级证据。前78条没有经过本轮逐条人工听辨，不能写成“人工确认”。题干拆解和基础材料见 [`reports/问题1完整求解与写作材料.md`](reports/问题1完整求解与写作材料.md)，写作人员必须先查看 [`reports/README.md`](reports/README.md) 中的增量交接顺序。附件1生成的25维openSMILE音频与35维FaceMesh视觉特征不等同于附件2的74维COVAREP和35维FACET，不得直接混用已训权重。

首次运行建议先确认审计通过：

```powershell
conda activate mosei
cd pythonproject
python audit_data.py
```

基础训练入口为 `train_baselines.py`。例如：

```powershell
# 端到端冒烟测试
python train_baselines.py --models text_bigru --run-name exp001_text_bigru --smoke --validation-only

# 正式迭代实验：不查看测试集
python train_baselines.py --models text_bigru --run-name exp001_text_bigru --epochs 50 --patience 8 --validation-only
```

## 2. 数据与防泄漏约定

- 使用 `aligned_50.pkl` 的官方 `train/valid/test`，不随机重划分。
- 音频和视觉的标准化参数只使用训练集有效帧计算。
- padding 不参与池化、标准化或损失计算。
- 模型结构、超参数和实验方向只依据验证集调整。
- 测试集指标不用于日常迭代；只有方案冻结后才统一评估。
- 附件3仅有 `text_bert` 时，先用 `prepare_attachment3.py` 重建 `text`；不能直接读取不存在的键。
- 连续回归值 `0` 只表示中性；当前三分类文件编码为 `0=负向、1=中性、2=正向`。
- 附件3、附件4无标签，不用于伪标签训练；不引入外部情感数据集。

数据审计结果见 [`reports/data_audit.json`](reports/data_audit.json)。

## 3. 评价指标解释

| 指标 | 含义 | 方向 | 使用方式 |
|---|---|---|---|
| Accuracy | 三分类预测正确的样本比例 | 越大越好 | 直观，但类别不平衡时可能偏向多数类 |
| Macro-F1 | 分别计算三类 F1 后等权平均 | 越大越好 | 主要分类指标，关注少数类表现 |
| Weighted-F1 | 按各类样本数加权的 F1 | 越大越好 | 反映总体分类效果，容易受多数类影响 |
| MAE | 回归预测与真实情感分数的平均绝对误差 | 越小越好 | 主要回归误差指标 |
| Pearson | 预测分数与真实分数的线性相关系数 | 越大越好 | 衡量情感变化趋势是否一致 |

当前 checkpoint 选择分数为：

```text
selection_score = Macro-F1 + Pearson - MAE / 3
```

它只用于同一实验协议下的早停与 checkpoint 选择，不作为竞赛官方指标。分类与回归同等重要时，必须同时报告 Macro-F1、MAE 和 Pearson，不能只用综合分数掩盖单项退化。

## 4. 实验管理结构

```text
pythonproject/
├─ experiments/                 # 中文实验记录、模板和索引
├─ reports/README.md            # 写作材料权威入口与版本顺序
├─ reports/writing_handoffs/    # 问题1/2/3按日期追加的写作纠偏
├─ reports/                     # 数据审计与阶段性结论
├─ outputs/<run_name>/          # 权重、指标、预测、曲线和 TensorBoard 日志
├─ src/mosei/                   # 数据、模型、指标与公共函数
├─ train_baselines.py           # 统一训练入口
└─ NEXT_SESSION.md              # 新会话首先阅读的交接文件
```

每个正式实验都必须保留：实验目的、对照关系、唯一变量、运行命令、验证结果、现象分析、结论和下一步。统一模板见 [`experiments/TEMPLATE.md`](experiments/TEMPLATE.md)，实验总索引见 [`experiments/README.md`](experiments/README.md)。

## 5. 当前阶段结论

- 问题1工程已完成：100条均通过结构检查；严格对齐口径为78条自动高置信、2条部分匹配、20条不支持完整词级证据。典型样本图已重构为完整的可核验对齐证据链。
- 问题2正式分类模型为三种子三模态输出残差BiGRU，正式回归模型为三种子Text+Audio-BiGRU。验证集Accuracy/Macro-F1/MAE/Pearson为0.6525/0.6458/0.5648/0.6772，附件3的30条预测已冻结。
- 问题3已完成反事实模态贡献、连续块局部证据和多种子稳定性验证；附件4已有20条预测、99条局部证据和18张关键帧。
- 补充的轻量校准未通过统计更新门槛，正式模型保持不变；随机连续缺失、不确定性和重复随机解释实验均已完成。
- 官方test已在所有选择冻结后唯一一次评估：Accuracy/Macro-F1/Weighted-F1=0.6726/0.6499/0.6811，MAE/Pearson=0.6129/0.6837。`RUN_COMPLETE.json`存在时脚本拒绝再次运行，禁止据此调参。
- 三问唯一算法附件已按白名单生成于`submission/研究生数学建模E题_最终提交附件.zip`；禁止提交整个`outputs/`目录或历史实验权重。
- 写作人员必须从[`reports/README.md`](reports/README.md)和[`reports/writing_handoffs/README.md`](reports/writing_handoffs/README.md)进入最新口径，不得从历史实验中重新挑选结果。

当前完整状态见[`reports/当前解题进度与剩余阶段.md`](reports/当前解题进度与剩余阶段.md)。历史基线与筛选过程分别保留在[`reports/baseline_report.md`](reports/baseline_report.md)和`experiments/`，仅用于实验追溯。跨会话继续工作时，先阅读[`NEXT_SESSION.md`](NEXT_SESSION.md)。
