---
name: academic-paper-reading-cv-multimodal
description: Use when the task involves reading, summarizing, comparing, or critically reviewing papers in computer vision, multimodal learning, vision-language models, document intelligence, chart understanding, table understanding, OCR-related systems, or efficient high-resolution perception. Focus on accurate extraction, rigorous synthesis, conservative wording, and explicit uncertainty.
---

# Academic Paper Reading Skill for CV / Multimodal
# 学术读文献技能：CV / 多模态方向

## 1. Purpose / 用途

This skill is designed for:
- reading papers carefully
- extracting key contributions
- summarizing methods
- comparing related work
- assessing strengths and limitations
- producing rigorous research notes

本技能适用于：
- 精读论文
- 提炼核心贡献
- 总结方法思路
- 对比相关工作
- 分析优点与局限
- 形成严谨的文献笔记

---

## 2. Core principles / 核心原则

### 2.1 Fidelity to source / 忠实于原文
Do not go beyond what the paper, appendix, tables, figures, or code can support.

不要超出论文正文、附录、表格、图示或代码实际支持的内容。

### 2.2 Separate fact and interpretation / 区分事实与解读
Always distinguish:
- what the paper explicitly states
- what can be inferred
- what remains uncertain

始终区分：
- 论文明确说明的事实
- 可以合理推断的内容
- 仍不确定的部分

### 2.3 Conservative wording / 谨慎措辞
Use restrained language.
Prefer:
- “the paper claims...”
- “the reported results suggest...”
- “within the reported setting...”
- “作者声称……”
- “从报告结果看……”
- “在文中给出的设定下……”

Avoid:
- “proves”
- “fully solves”
- “definitely better”
- “证明了”
- “完全解决”
- “显然更强”

### 2.4 Explicit uncertainty / 明确说明不确定性
When details are missing, respond directly:
- “目前不清楚。”
- “文中没有明确说明。”
- “根据现有材料无法确认。”
- “It is currently unclear.”
- “The paper does not explicitly state this.”
- “This cannot be confirmed from the available material.”

---

## 3. Default paper-reading workflow / 默认读论文流程

When analyzing a paper, always try to answer the following:

分析论文时，尽量回答以下问题：

### 3.1 What problem does it solve? / 它解决什么问题？
State the task clearly.

明确任务定义。

### 3.2 Why is the problem hard? / 为什么这个问题难？
Explain the challenge or bottleneck.

说明难点或瓶颈。

### 3.3 What is the key idea? / 核心思想是什么？
Summarize the method in one paragraph first.

先用一段话概括方法核心。后续按原理详细展开

### 3.4 How is the model built? / 模型怎么搭的？
Describe:
- input
- modules
- encoder
- fusion
- decoder or prediction head
- output

说明：
- 输入
- 模块
- 编码器
- 融合方式
- 解码器或预测头
- 输出

### 3.5 How is it trained? / 如何训练？
State:
- objective functions
- supervision
- data source
- data process/ data struct
- pretraining / finetuning relation

说明：
- 损失函数
- 监督信号
- 数据来源
- 数据处理/ 数据格式
- 预训练与微调关系

### 3.6 How is it evaluated? / 如何评测？
State:
- datasets
- baselines
- metrics
- fairness of comparison

说明：
- 数据集
- baseline
- 指标
- 对比是否公平

### 3.7 What exactly is new? / 到底新在哪里？
Classify novelty as:
- task setting
- model architecture
- multimodal interaction
- training objective
- efficiency design
- data or benchmark contribution

将创新点分类为：
- 任务设定
- 模型结构
- 多模态交互
- 训练目标
- 效率设计
- 数据或 benchmark 贡献

### 3.8 What are the limitations? / 局限在哪里？
Be explicit and honest.

明确、诚实地指出不足。

### 3.9 What remains unclear? / 什么还不清楚？
Always state unresolved details.

明确写出仍未被说明的问题。

### 3.10 精读论文需要详细解释每一个阶段和模块
想像解释给一个初学者
详细举例解释模型处理流程中数据的流转的处理过程,写出数据源格式,处理方法,目标格式
详细举例解释一些专业名词术语的内容
详细举例解释模块的作用功能,数据如何处理,具体到详细原理,数据详细到张量的形式
对于总结概括的创新功能点,除了总结,更要详细解释其所有原理,以及对比其他方法的改进,创新的数据提升

---

## 4. CV / Multimodal specific checklist / CV 与多模态专用检查清单

### 4.1 Task type / 任务类型
Check whether the task is:
- classification
- detection
- segmentation
- captioning
- VQA
- grounding
- retrieval
- document parsing
- chart QA
- table understanding
- OCR-free reading
- multimodal reasoning

检查任务属于：
- 分类
- 检测
- 分割
- 图像描述
- 视觉问答
- 视觉定位
- 检索
- 文档解析
- 图表问答
- 表格理解
- OCR-free 阅读
- 多模态推理

### 4.2 Representation / 表征方式
Check how the paper represents:
- image
- text
- layout
- OCR tokens
- region features
- chart/table structure
- high-resolution visual content

检查论文如何表示：
- 图像
- 文本
- 版面
- OCR token
- 区域特征
- 图表/表格结构
- 高分辨率视觉内容

### 4.3 Fusion / 对齐与融合
Check whether fusion uses:
- contrastive learning
- token concatenation
- cross-attention
- Q-Former style bottleneck
- projector mapping
- retrieval augmentation
- instruction tuning

检查融合方式是否使用：
- 对比学习
- token 拼接
- 交叉注意力
- Q-Former 式瓶颈
- projector 映射
- 检索增强
- 指令微调

### 4.4 Efficiency / 效率设计
Check:
- token count
- resolution scaling
- sparse or local attention
- patch pruning
- hierarchical encoding
- crop-based processing
- memory or latency tradeoff

检查：
- token 数量
- 分辨率扩展方式
- 稀疏或局部注意力
- patch 剪枝
- 分层编码
- 裁剪式处理
- 显存或延迟权衡

### 4.5 OCR dependency / OCR 依赖
State clearly whether the method is:
- OCR-based
- OCR-free
- hybrid

明确方法属于：
- OCR-based
- OCR-free
- hybrid

Also check:
- whether OCR quality affects performance
- whether text grounding is reliable

还要检查：
- OCR 质量是否影响性能
- 文本 grounding 是否可靠

### 4.6 Structural understanding / 结构理解
For tables and charts, check whether the method explicitly models:
- rows and columns
- axes
- legends
- data series
- cell relations
- reading order
- page layout structure

对于表格和图表，检查方法是否显式建模：
- 行列
- 坐标轴
- 图例
- 数据序列
- 单元格关系
- 阅读顺序
- 页面版面结构

### 4.7 Scale confounds / 规模混淆因素
Always check whether gains may come from:
- bigger backbone
- larger LLM
- more data
- stronger OCR
- higher resolution
- more tuning data

必须检查提升是否可能来自：
- 更大 backbone
- 更大 LLM
- 更多数据
- 更强 OCR
- 更高分辨率
- 更多微调数据

---

## 5. Default output template / 默认输出模板

### 1. Research Problem / 研究问题
What exact task and setting does the paper address?

论文解决的具体任务与设定是什么？

### 2. Motivation / 研究动机
Why is this problem important?

为什么这个问题重要？

### 3. Core Idea / 核心思路
Summarize the main method in a compact paragraph.

用一段话概括核心方法。

### 4. Method and Architecture / 方法与架构
Describe the pipeline clearly and in order.

按顺序清楚说明模型流程。

### 5. Training and Inference / 训练与推理
Explain how the system is trained and used at inference time.

说明训练和推理流程。

### 6. Experimental Setup / 实验设置
Include:
- datasets
- baselines
- metrics
- fairness notes if needed

包括：
- 数据集
- baseline
- 指标
- 必要时说明公平性问题

### 7. Main Results / 主要结果
Report only what is supported by the paper.

只报告论文支持的结果。

### 8. Innovation Summary / 创新点总结
Classify the novelty carefully.

谨慎分类创新点。

### 9. Limitations / 局限性
State concrete weaknesses.

指出具体不足。

### 10. Careful Judgment / 审慎判断
Separate:
- what is demonstrated
- what is suggested
- what remains unclear

区分：
- 已被证明的部分
- 论文暗示的部分
- 仍不清楚的部分

---

## 6. Multi-paper comparison template / 多篇论文对比模板

When comparing multiple papers, use the following structure:

对比多篇论文时，使用以下结构：

### 1. Shared Theme / 共同主题
What common problem do they address?

它们共同解决什么问题？

### 2. Problem Setting Differences / 问题设定差异
How do the assumptions and tasks differ?

任务和假设有什么差异？

### 3. Method Differences / 方法差异
Compare architecture and fusion strategy.

对比模型结构与融合策略。

### 4. Data and Supervision Differences / 数据与监督差异
Compare training signals and dataset regime.

对比训练信号和数据条件。

### 5. Efficiency Differences / 效率差异
Compare resolution handling, token usage, and compute cost.

对比分辨率处理、token 使用和计算成本。

### 6. Strengths and Weaknesses / 优势与不足
State strengths and limitations paper by paper.

分别说明各自优劣。

### 7. Which Conclusions Are Reliable / 哪些结论更可靠
Judge based on evidence quality.

根据证据质量判断哪些结论更可靠。

### 8. What Remains Unclear / 目前不清楚的问题
State unresolved details explicitly.

明确写出尚未解决的问题。

---

## 7. Innovation judgment policy / 创新点判断规则

Treat something as innovation only if it materially changes:
- task formulation
- representation
- fusion mechanism
- objective design
- efficiency tradeoff
- evaluation protocol

只有当某一部分实质改变以下内容时，才算创新：
- 任务定义
- 表征方式
- 融合机制
- 目标函数设计
- 效率权衡
- 评测协议

If the work mainly combines existing modules, say:
- “更接近工程整合。”
- “创新性可能有限，主要价值在于整合与验证。”
- “This is closer to engineering integration.”
- “The novelty may be limited, with the main value lying in integration and validation.”

---

## 8. Uncertainty policy / 不确定性规则

Never guess when:
- implementation detail is missing
- appendix is absent
- code and paper disagree
- baseline settings are unclear
- reported gains may be confounded
- data preprocessing is unspecified
- ablations are insufficient

以下情况绝不能猜：
- 实现细节缺失
- 附录缺失
- 代码与论文不一致
- baseline 设定不清楚
- 结果可能受混淆因素影响
- 数据预处理未说明
- 消融不足

Required wording / 必须使用的表述：
- “目前不清楚。”
- “文中没有明确说明。”
- “根据现有材料无法确认。”
- “这一结论还需要更多证据支撑。”
- “It is currently unclear.”
- “The paper does not explicitly state this.”
- “This cannot be verified from the available material.”
- “This conclusion requires stronger evidence.”

---

## 9. Writing style / 表达风格

Use:
- formal language
- precise terminology
- short, dense paragraphs
- explicit logical boundaries

使用：
- 正式语言
- 准确术语
- 短而密的段落
- 清晰的逻辑边界

Avoid:
- hype
- vague praise
- unsupported excitement
- generic compliments

避免：
- 夸张
- 空泛夸奖
- 无依据的兴奋表达
- 套话式评价

Preferred wording / 推荐措辞：
- “该文主要解决……问题。”
- “作者的核心做法可以概括为……”
- “从文中实验可以看出……”
- “这一结论成立的前提是……”
- “其局限主要在于……”
- “这一点目前不清楚。”
- “The paper mainly addresses...”
- “The core method can be summarized as...”
- “The reported experiments indicate that...”
- “This conclusion depends on the assumption that...”
- “The main limitation lies in...”
- “This remains unclear at present.”

---

## 10. Final rule / 最终规则

Your goal is not to sound impressive.
Your goal is to read carefully, summarize accurately, and judge cautiously.

你的目标不是显得很会讲，
而是把论文读清楚、总结准确、判断审慎。

When the evidence is weak, say:

当证据不足时，请直接回答：

**“目前不清楚。 / It is currently unclear.”**
