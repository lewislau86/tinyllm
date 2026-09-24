# TinyLLM 教程

从零实现一个能够训练、生成文本并接受基本评估的 TinyLLM。课程既解释模型内部的计算，也覆盖数据、训练和推理。每章先讲原理，再用 PyTorch 做可运行的实验。

## 课程目录

| 章节 | 内容 | 状态 |
| --- | --- | --- |
| [Chapter 0：Normalization](./chapter0_Normalization/) | LayerNorm、RMSNorm、ScaleNorm、QK Norm、DeepNorm 及相关发展 | 教案与 PyTorch 数据实验已完成 |

## 课程路线（规划）

下面是计划覆盖的知识点，并不表示后续章节文档已经完成。Chapter 0 从归一化入手；后面先补齐构造模型所需的组件，再把它训练起来。

### 第一阶段：理解模型怎样计算

1. **Chapter 0 · 归一化**：LayerNorm、RMSNorm、ScaleNorm、QK Norm、DeepNorm，以及归一化位置与新变体。
2. **Chapter 1 · 张量、参数与自动求导**：维度、矩阵乘法、计算图、梯度和参数更新；有 PyTorch 基础的读者可跳读。
3. **Chapter 2 · 文本、词元化与词表**：Unicode/字节、BPE 的基本思路、特殊 token、编码与解码。
4. **Chapter 3 · 词嵌入与输出层**：token ID 如何变成向量、输出 logits、权重共享。
5. **Chapter 4 · 位置表示**：绝对位置编码和 RoPE；理解为什么注意力本身不知道顺序。
6. **Chapter 5 · 注意力机制**：Q/K/V、缩放点积、因果掩码、多头注意力、形状推导。
7. **Chapter 6 · 前馈网络与残差连接**：MLP、常见激活函数、残差路径及归一化位置。
8. **Chapter 7 · Decoder-only Transformer**：把前面模块接成可输出下一 token 概率的模型。

### 第二阶段：让模型学会预测下一词元

9. **Chapter 8 · 训练数据与样本构造**：语料清洗、训练/验证集划分、上下文窗口、输入和目标错位。
10. **Chapter 9 · 语言模型目标函数**：softmax、交叉熵、负对数似然、困惑度以及 padding/忽略标签。
11. **Chapter 10 · 训练循环**：batch、反向传播、AdamW、学习率计划、梯度裁剪、混合精度、检查点与复现。
12. **Chapter 11 · 训练诊断与评估**：过拟合一个小 batch、观察 loss、验证集评估和生成样例检查。

### 第三阶段：使用模型并理解常见扩展

13. **Chapter 12 · 自回归生成**：逐 token 解码、温度、top-k、top-p、停止条件和重复问题。
14. **Chapter 13 · 推理效率**：KV cache、prefill/decode、注意力计算与显存开销。
15. **Chapter 14 · 指令微调入门**：预训练与微调的区别、对话格式、监督微调的数据与 loss mask。

**最小闭环**是：准备文本 → 词元化 → 构建 Decoder-only 模型 → 用下一 token 目标训练 → 生成文本 → 在验证集上评估。完成这个闭环之后，可再按需要加入 GQA、FlashAttention、LoRA、量化、分布式训练、RAG 和对齐方法；这些属于扩展专题，不是理解第一个 TinyLLM 的先决条件。

每章完成时，应包含该章的 README、PyTorch 示例代码，以及能说明概念的实验或检查。
