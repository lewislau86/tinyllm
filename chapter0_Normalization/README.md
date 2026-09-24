# TinyLLM 教程 · 第一课：归一化

本课程从零实现一个小型语言模型。第一课先讲清 Transformer 中的归一化：**对什么数做归一化、怎么算、放在哪里，以及为什么要做**。本课先写文档，PyTorch 代码后续补充。

## 学习目标

1. 看懂形状为 `[batch, seq_len, hidden_size]` 的张量如何按最后一维归一化。
2. 手算 LayerNorm 和 RMSNorm，解释二者的差别。
3. 区分“归一化算法”和“归一化在残差块中的位置”。
4. 认识 LLM 中的其他相关方案。

## 1. 归一化在处理什么？

每个 token 都对应一个含 `d` 个数的隐藏向量。常见的 LayerNorm / RMSNorm 会**独立处理每个 token 的隐藏向量**，在 `hidden_size` 维度上计算统计量；不会把整个 batch 或所有 token 混在一起求均值。归一化有助于控制送入后续层的数值尺度，但训练是否稳定还取决于残差连接、初始化、学习率和数值实现。

以下令一个隐藏向量为 $x=(x_1,\ldots,x_d)$，$\epsilon>0$ 是防止分母为零的小常数，$\gamma$ 是可学习的逐元素缩放参数，$\beta$ 是可学习的逐元素平移参数。

## 2. 两个核心方案

### LayerNorm：先居中，再缩放

$$
\mu=\frac{1}{d}\sum_{i=1}^{d}x_i,\qquad
\sigma^2=\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2
$$

$$
y_i=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i
$$

它先减去均值，再按标准差调整尺度。公式中的方差分母为 `d`。常见的 PyTorch `LayerNorm` 默认有 `weight` 和 `bias` 两组可学习参数。

### RMSNorm：只按均方根缩放

$$
\operatorname{RMS}(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon},\qquad
y_i=\gamma_i\frac{x_i}{\operatorname{RMS}(x)}
$$

它不减均值，也不计算围绕均值的方差。常见实现只有 `weight`，没有 `bias`。这里把 $\epsilon$ 放在平方根内；阅读具体模型代码时仍要核对其实现约定。

### 手算例子

取 $x=[1,2,3]$，暂设 $\epsilon=0$、$\gamma=[1,1,1]$、$\beta=[0,0,0]$：

- LayerNorm：均值 $\mu=2$，方差 $\sigma^2=2/3$，输出约为 $[-1.225,0,1.225]$。
- RMSNorm：均方根 $\sqrt{14/3}\approx2.160$，输出约为 $[0.463,0.926,1.389]$。

这组数展示了关键区别：**RMSNorm 的输出通常不以 0 为中心**。真实计算必须保留正的 $\epsilon$，以处理全零向量等情况。

### 对比表

| 特性 | LayerNorm | RMSNorm |
| --- | --- | --- |
| 归一化统计量 | 均值、围绕均值的方差 | 输入平方的均值，即均方根 |
| 是否减均值 | 是 | 否 |
| 常见可学习参数 | $\gamma$ 和 $\beta$，各 `d` 个 | $\gamma$，`d` 个 |
| 计算步骤 | 求均值 → 减均值 → 求方差 → 缩放和平移 | 求平方均值 → 缩放 |
| 计算性能 | 需计算均值及方差 | 算术步骤更少；实际速度取决于实现和硬件 |
| 混合精度 | 需要妥善处理统计量计算精度 | 同样需要妥善处理平方、累加和 $\epsilon$ |
| 代表性模型 | BERT、GPT-2、GPT-3 | LLaMA 系列等 |

**不要把 RMSNorm 的“步骤更少”理解为任何设备、任何模型上都固定快某个百分比。**原始论文报告在其不同模型和实现中观察到约 7%–64% 的加速；这不是本项目的性能承诺。也不能简单说 LayerNorm 在 FP16 下必然不稳定、RMSNorm 在 BF16 下必然稳定：两者都要留意累加精度、$\epsilon$ 和具体内核。

## 3. 还有其他归一化方案吗？

有。本课程先掌握上面两种最常见的隐藏状态归一化，再认识这些方案：

| 名称 | 核心想法 | 与本课主线的关系 |
| --- | --- | --- |
| ScaleNorm | 用向量的 $\ell_2$ 范数和一个可学习尺度归一化 | 是另一种替代 LayerNorm 的向量归一化 |
| QK Norm | 在注意力计算中归一化 query 和 key | 作用于注意力内部的 Q/K，位置不同于残差流上的 LayerNorm/RMSNorm |
| DeepNorm | 调整深层 Transformer 的残差连接、归一化和初始化 | 面向极深网络的结构方案，不只是替换一个公式 |

这些方案说明“LLM 归一化”并非只有两种，但也不是都要放入 TinyLLM 的第一个实现。BatchNorm 在深度学习中很重要；它跨 batch 统计的做法与本课逐 token 的隐藏向量归一化不同，因此仅作为背景知识。

## 4. 归一化放在哪里？

**Pre-Norm / Post-Norm 说的是位置，LayerNorm / RMSNorm 说的是算法。**设 $F$ 是注意力层或前馈层：

```text
Pre-Norm:   output = x + F(Norm(x))
Post-Norm:  output = Norm(x + F(x))
```

例如，LLaMA 使用 **Pre-Norm + RMSNorm**。不要把 Pre-Norm 当作第三种与 RMSNorm 并列的计算公式。一个 Transformer block 通常有注意力和前馈两个子层，两处都需要明确归一化位置；模型末尾还可能有额外的归一化层。

## 5. 后续 PyTorch 代码要做的实验

1. 手写 LayerNorm 与 RMSNorm，并与 PyTorch 对应模块比较输出和梯度。
2. 用 `[batch, seq_len, hidden_size]` 张量确认只在最后一维计算统计量。
3. 比较有偏移的输入、全零输入及不同数据类型下的结果；性能测试记录设备、dtype、形状和实现方式，再讨论速度。

## 参考资料

- [Layer Normalization，Ba 等，2016](https://arxiv.org/abs/1607.06450)
- [Root Mean Square Layer Normalization，Zhang 与 Sennrich，2019](https://arxiv.org/abs/1910.07467)
- [Transformers without Tears：ScaleNorm，Nguyen 与 Salazar，2019](https://arxiv.org/abs/1910.05895)
- [Query-Key Normalization for Transformers，Henry 等，2020](https://aclanthology.org/2020.findings-emnlp.379/)
- [DeepNet：DeepNorm，Wang 等，2022](https://arxiv.org/abs/2203.00555)
- [LLaMA 论文，Touvron 等，2023](https://arxiv.org/abs/2302.13971)
- [GPT-3 论文，Brown 等，2020](https://arxiv.org/abs/2005.14165)
- [PyTorch LayerNorm 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) · [PyTorch RMSNorm 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)
