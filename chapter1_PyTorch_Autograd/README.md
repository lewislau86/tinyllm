# Chapter 1：PyTorch 张量与自动求导

上一章研究了归一化如何处理一个 token 的隐藏向量。本章先退一步：那个向量在代码里是什么？形状为何重要？模型又如何知道参数该往哪个方向更新？答案分别是**张量**、**张量运算**和**自动求导**。

本文把 [Datawhale《深入浅出 PyTorch》第二章](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E4%BA%8C%E7%AB%A0/index.html)的张量、自动求导、并行与硬件内容浓缩到 TinyLLM 的学习场景。配套 [`ch1.ipynb`](./ch1.ipynb) 提供固定数据、预期输出和逐格解释；不要求先有 GPU。

## 学习目标

1. 读懂 `[batch, seq, hidden]` 等形状，分清索引、转置、变形和复制。
2. 预测广播后的形状，分清逐元素乘法与矩阵乘法。
3. 用链式法则解释 `loss.backward()`，检查梯度是否正确、是否累积。
4. 让模型和数据处于同一设备；知道 MLX 与 PyTorch 的自动求导接口不同。

## What：张量是带形状的一组数

一个数字是 0 维张量，列表可视为 1 维张量，表格是 2 维张量。LLM 的中间结果常有更多维。例如 `[2, 3, 4]` 可以读成“2 个样本，每个样本 3 个 token，每个 token 有 4 个特征”。**轴的名字是我们赋予的语义；PyTorch 只知道各轴长度。**

| 对象 | 常见形状 | 含义 |
| --- | --- | --- |
| token ID | `[batch, seq]` | 每个位置对应词表中的整数索引 |
| 隐藏状态 | `[batch, seq, hidden]` | 每个 token 的浮点特征向量 |
| 注意力 Q/K | `[batch, heads, seq, head_dim]` | 每个头、每个位置的向量 |
| 注意力分数 | `[batch, heads, query_len, key_len]` | 每个 query 对各 key 的分数 |

`shape` 回答“每条轴多长”，`ndim` 回答“有几条轴”，`dtype` 回答“每个数怎样存”，`device` 回答“数据在哪个计算设备”。词元 ID 通常为整数，隐藏状态和可训练权重通常为浮点数。`torch.tensor(...)` 从现有数据创建张量；`torch.zeros(...)` 创建指定形状的零张量；`torch.randn(...)` 创建随机张量。后者适合实验，但真实模型参数的初始化规则将在 [Chapter 2](../chapter2_Initialization_Residuals/README.md) 学习。

```python
import torch

token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
print(token_ids.shape, token_ids.dtype)  # torch.Size([2, 3]), torch.int64
print(hidden.shape, hidden.ndim)          # torch.Size([2, 3, 4]), 3
```

### 索引、变形、转置：别把轴换错

`hidden[0, 1, :]` 是第一个样本、第二个 token 的 4 维向量。`reshape(2, 12)` 改变形状；`transpose(1, 2)` **交换轴**，从 `[2, 3, 4]` 得到 `[2, 4, 3]`。两者不是同一种操作。`view` 要求兼容的内存布局；转置后的张量通常不连续，直接 `view` 可能报错，可先 `contiguous()` 或使用 `reshape()`。

```python
q = torch.randn(2, 3, 4, 5)       # [batch, heads, query_len, head_dim]
k = torch.randn(2, 3, 6, 5)       # [batch, heads, key_len, head_dim]
scores = q @ k.transpose(-2, -1)  # [2, 3, 4, 6]：最后两轴变成 [head_dim, key_len]
```

这里必须转置 K 的**末尾两轴**。`k.T` 会反转所有轴，四维注意力里通常不对。`reshape()` 可能返回视图，也可能复制数据，不能假定它一定创建独立副本；需要独立数据时明确调用 `clone()`。索引得到的切片也可能共享底层存储。参见 [PyTorch reshape 文档](https://docs.pytorch.org/docs/stable/generated/torch.reshape.html)、[view 文档](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html)。

## How：广播和矩阵乘法如何组合

**广播**让某个长度为 1 或缺失的轴在运算时扩展。比较两个形状时，从右往左看：对应轴相等，或其中一个为 1，才可广播。

```python
x = torch.zeros(2, 3, 4)      # [batch, seq, hidden]
bias = torch.tensor([1., 2., 3., 4.])  # [hidden]
y = x + bias                 # [2, 3, 4]：每个 token 加同一条 bias
```

`x * bias` 是逐元素乘法；`x @ weight` 是矩阵乘法。如果 `weight` 为 `[4, 6]`，那么 `x @ weight` 为 `[2, 3, 6]`：最后一维从 4 个输入特征变成 6 个输出特征。`nn.Linear(in_features=4, out_features=6)` 存储的 `weight` 则是 `[6, 4]`，其前向计算相当于 `x @ linear.weight.T + linear.bias`。**读网络代码先核对形状，再看数值。** 参见 [PyTorch 广播规则](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)、[`nn.Linear` 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)。

```python
linear = torch.nn.Linear(
    in_features=4,   # 输入向量最后一维
    out_features=6,  # 输出向量最后一维
    bias=True,       # 创建长度为 6 的可学习偏置
)
out = linear(x)      # [2, 3, 6]
```

## How：自动求导怎样找到更新方向

想象一个只有一个参数的模型，预测值为 $wx$，目标为 $t$，损失为 $L=(wx-t)^2$。若把 $w$ 略微增大，损失怎样变化？这个变化率就是梯度：

$$
\frac{\partial L}{\partial w}=2(wx-t)x
$$

这是链式法则：先求损失对预测值的变化，再乘预测值对参数的变化。PyTorch 在前向运算时记录计算关系；`loss.backward()` 沿图反向传播，把叶子参数的梯度放进 `.grad`。`requires_grad=True` 表示追踪这个张量参与的运算；由运算产生的结果通常带有 `grad_fn`。若输出是向量，需先通过 `.sum()`、`.mean()` 等得到标量损失，或者向 `backward` 提供同形状的上游梯度。参见 [PyTorch Autograd 教程](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)。

```python
w = torch.nn.Parameter(torch.tensor(2.0))  # 可学习的叶子参数
x = torch.tensor(3.0)
target = torch.tensor(1.0)
loss = (w * x - target).square()             # (2×3−1)² = 25
loss.backward()                             # dL/dw = 2×(6−1)×3 = 30
print(w.grad)                               # tensor(30.)
```

**关键细节：`.backward()` 默认把新梯度加进 `.grad`，不会自动清零。** 训练每一步前通常用 `optimizer.zero_grad(set_to_none=True)`；本章未引入优化器时可设 `w.grad = None`。故意多步累积梯度是另一种训练策略，将在 [Chapter 16](../chapter16_Gradient_Stability/README.md) 展开。不要通过 `.data` 修改参数或梯度；它会绕开自动求导需要的检查。

`torch.no_grad()` 让代码块内的运算不建立反向图，常用于推理或参数更新；`tensor.detach()` 从已有图分离一个张量，但可能仍共享存储。若还需要独立副本，用 `tensor.detach().clone()`。`model.eval()` 只切换 Dropout、BatchNorm 等模块的训练/评估行为，**不会自动关闭梯度记录**；推理时通常还需 `torch.no_grad()` 或 `torch.inference_mode()`。参见 [PyTorch Autograd 机制](https://docs.pytorch.org/docs/stable/notes/autograd.html)。

### 梯度检查：用小扰动核对公式

对一个标量参数，可用中心差分估计梯度：

$$
\frac{\partial L}{\partial w}\approx\frac{L(w+h)-L(w-h)}{2h}
$$

它只是数值近似；在 `float64` 和合适的小 $h$ 下，应该接近自动求导的结果。配套 notebook 同时打印手算值、自动求导值和有限差分值。复杂自定义算子可进一步用 [`torch.autograd.gradcheck`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html) 检查。

## Do：设备、实验与下一章

PyTorch 张量与模块要放在同一设备：CPU 最通用；有 NVIDIA CUDA 时可用 `cuda`；兼容的 Apple 设备可用 `mps`。**有设备不等于所有算子都支持或结果完全一致**，所以 notebook 先在 CPU 运行核心固定数据实验，再对可用的 CUDA/MPS 做一个小型前向与反向检查，不把它当性能基准。多卡数据并行、张量并行和硬件架构属于进阶主题，本章只建立“同一设备”这一必需概念。参见 [PyTorch 加速器说明](https://docs.pytorch.org/docs/stable/torch.html#accelerators) 与 [MPS 后端说明](https://docs.pytorch.org/docs/stable/notes/mps.html)。

```python
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
model = torch.nn.Linear(4, 6).to(device)
batch = torch.ones(2, 3, 4, device=device)
output = model(batch)  # 模型参数与输入在同一设备
```

**MLX 是另一套框架，不是 `torch.device("mlx")`。** 在 Apple Silicon 上，它使用 `mlx.core.grad` / `value_and_grad` 这样的函数式接口；notebook 的可选小实验单独演示 `mx.grad`，未安装 MLX 时明确跳过。它与 PyTorch 的 `.backward()` 不能直接混用。参见 [MLX 函数变换文档](https://ml-explore.github.io/mlx/build/html/usage/function_transforms.html)。

### 跟着 notebook 做

打开 [`ch1.ipynb`](./ch1.ipynb)，从上到下运行：

1. 观察形状、索引、转置与 `reshape` 的实际输出。
2. 预测广播、`@` 和 `nn.Linear` 的形状，再与代码核对。
3. 手算简单损失的梯度，观察两次 `backward()` 的累积。
4. 对照数值差分，最后检查本机可用的计算设备与 MLX。

读完本章，再看 [Chapter 2：初始化与残差路径](../chapter2_Initialization_Residuals/README.md)：你已知道参数怎样获得梯度，接下来要回答参数初始值与残差连接如何影响深层训练。

**阅读来源与边界。**本文借鉴 [Datawhale 第二章目录及四个小节](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E4%BA%8C%E7%AB%A0/index.html)的选题，示例、解释与 TinyLLM 形状均重新编写；API 细节以文中链接的 PyTorch/MLX 官方文档为准。返回 [课程总目录](../README.md#课程路线规划)。
