# TinyLLM 教程

从零实现一个能够训练、生成文本并接受基本评估的 TinyLLM。本项目既讲模型结构，也讲让模型**可训练、稳定训练和正确评估**所需的数据、初始化、优化、精度与工程方法。每章计划包含 Markdown 教案及对应的 PyTorch notebook。

**如果你刚学完大学基础数学与 Python：**先看“从哪里开始”和 Chapter 0 中的“小词典”，再运行 notebook 的前几个实验。下方 24 章是整门课的地图，不要求现在认识每个名词；遇到 MoE、μP 等先知道它们是后续专题即可。

## 已完成章节

| 章节 | 内容 | 状态 |
| --- | --- | --- |
| [Chapter 0：Normalization](./chapter0_Normalization/) | LayerNorm、RMSNorm、ScaleNorm、QK Norm、DeepNorm；BF16/FP16 数值实验 | 教案与 [ch0.ipynb](./chapter0_Normalization/ch0.ipynb) 已完成 |

## 课程路线（规划）

下面是**教学目录**，尚未建立的章节只是规划。现有 Chapter 0 保留原编号。课程按“基础工具 → 构造模型 → 训练模型 → 使用与扩展”推进；实践中可以先完成最小语言模型，再回头深入各类训练策略。

### 第一阶段：张量与稳定训练基础

- **Chapter 0 · 归一化**：已有 LayerNorm、RMSNorm、ScaleNorm、QK Norm、DeepNorm、Pre/Post-Norm 与 FP16/BF16 实验；后续补充 BatchNorm、GroupNorm 的对照。
- **Chapter 1 · PyTorch 张量与自动求导**：维度、广播、矩阵乘法、参数、计算图、反向传播、梯度检查。已有 PyTorch 基础可跳读。
- **Chapter 2 · 初始化与残差路径**：正态/截断正态、Xavier、He 初始化；Residual Connection、Pre/Post-Norm、残差缩放、深度增加时的激活和梯度。
- **Chapter 3 · 激活函数与门控前馈层**：ReLU、GELU、SiLU/Swish、SwiGLU；函数曲线、梯度、参数量与 MLP 结构。

### 第二阶段：构造 Decoder-only 语言模型

- **Chapter 4 · 文本与词元化**：Unicode/字节、词表、BPE、特殊 token、编码和解码。
- **Chapter 5 · 词嵌入与输出层**：token ID 到向量、logits、输出投影、Weight Tying（输入输出权重共享）。
- **Chapter 6 · 位置表示**：绝对位置编码、RoPE，位置作用于注意力的哪个环节。
- **Chapter 7 · 自注意力**：Q/K/V、缩放点积、因果 mask、多头注意力、张量转置与形状核对。
- **Chapter 8 · Transformer Block**：注意力、前馈层、残差、归一化的连接方式；搭建多个 Block。
- **Chapter 9 · 完整 TinyLLM**：组装 Decoder-only 模型，检查参数量、前向形状和下一 token logits。

### 第三阶段：数据、目标函数与训练

- **Chapter 10 · 语料与数据管线**：清洗、去重、训练/验证划分、shuffle、batching、bucketing、数据来源混合、数据泄漏检查。
- **Chapter 11 · 语言模型目标函数**：输入与目标错位、softmax、交叉熵、负对数似然、困惑度、padding 与 loss mask。
- **Chapter 12 · 优化器**：SGD、Momentum、Adam、AdamW、Adafactor；参数组、优化器状态、AdamW 与 L2 正则的区别。
- **Chapter 13 · 学习率策略**：Warmup、Cosine Decay、Linear Decay、OneCycle；按 step 更新与学习率曲线。
- **Chapter 14 · 正则化与泛化**：Weight Decay、L1/L2、Dropout、Label Smoothing、Early Stopping；训练损失与验证损失。区分通用方法与语言模型预训练的具体选择。
- **Chapter 15 · 梯度与数值精度**：FP32、FP16、BF16、Mixed Precision、autocast；Gradient Clipping、Gradient Scaling（包含 FP16 Loss Scaling）、梯度累积与梯度数值监测。
- **Chapter 16 · 训练循环与检查点**：DataLoader、forward/backward、optimizer/scheduler step、随机种子、保存与恢复。
- **Chapter 17 · 训练诊断与评估**：先过拟合一个小 batch，再观察 loss、梯度、验证集困惑度和生成样例。

### 第四阶段：生成、效率和进阶结构

- **Chapter 18 · 自回归生成**：逐 token 解码、温度、top-k、top-p、停止条件与重复问题。
- **Chapter 19 · 推理效率**：Prefill/decode、KV cache、注意力时间与显存开销、批量生成。
- **Chapter 20 · 指令微调**：预训练与监督微调、对话模板、只对答案计算 loss、基础评估。
- **Chapter 21 · 参数高效与稀疏结构**：LoRA、MoE、结构化/非结构化稀疏、剪枝；分别说明节省的是训练参数、推理计算还是存储。
- **Chapter 22 · 大规模训练稳定性**：μP、DeepNorm/残差缩放、深度与宽度扩展；在 Chapter 0 的公式基础上比较完整训练配置。
- **Chapter 23 · 扩展专题**：GQA、FlashAttention、量化、分布式训练、RAG 与对齐方法；根据课程进展拆为独立章节。

## 从哪里开始

**最小闭环**：准备文本 → 词元化（把文字变成整数编号）→ 构建 Decoder-only 模型（只能看当前及之前的词）→ 训练它预测下一个词元 → 用没训练过的数据验证 → 逐词生成文本。第一阶段和第三阶段中的许多方法会让这个闭环更稳、更可解释，但不要求在第一个小模型里一次用完所有技术。Chapter 0 已提供归一化理论与固定数据实验；后续章节会沿用“公式/参数解释 → PyTorch 调用 → 可复现实验 → 如何读结果”的教案格式。



## 其他

- **Pre-LN/Post-LN 是归一化放置方式**，Residual Connection 是结构，RMSNorm 是具体归一化算法；三者可以组合，不应作为互斥选项。
- **Weight Decay 与 L2 惩罚不总等价**。特别是在 AdamW 中，权重衰减与梯度中的 L2 项是不同做法，课程会用参数更新算例对照。[PyTorch AdamW 文档](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) 提供了算法定义。
- **Label Smoothing、Early Stopping、Dropout 并非每个 LLM 预训练都必须采用**；要结合目标函数、训练预算与验证策略判断。
- **Gradient Scaling 是泛称，Loss Scaling 是其中用于缩放 loss/梯度的具体做法**，在 FP16 混合精度训练中特别常见。梯度裁剪限制梯度范数，梯度累积合并多个 micro-batch；这三类操作的目的与调用顺序需要分别解释。
- **BatchNorm/GroupNorm 值得对照学习**，但本项目的 Decoder-only 主线优先理解逐 token 的 LayerNorm/RMSNorm；[PyTorch 归一化 API](https://docs.pytorch.org/docs/stable/nn.html#normalization-layers) 可用于检查各层统计维度。

数据去重和混合不只是加载技巧，也会影响语言模型质量与评估可信度，可参见 [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) 和 [DataComp-LM](https://arxiv.org/abs/2406.11794)。训练循环的基础调用顺序可参考 [PyTorch 官方优化教程](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial)。
