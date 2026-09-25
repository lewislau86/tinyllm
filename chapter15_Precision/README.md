# Chapter 15：数值精度与混合精度

本章回答三个问题：**What：浮点数到底能表示什么？How：计算误差从哪里来，混合精度怎样工作？Do：如何用代码观察并检查这些现象？**配套实验见 [`ch15.ipynb`](./ch15.ipynb)。只需了解张量、矩阵乘法和 [Chapter 0 的归一化](../chapter0_Normalization/README.md)。基础实验可在 PyTorch CPU 运行；有 CUDA、PyTorch MPS 或 MLX 时还可运行对应的加速实验。

学完后，你应能区分“范围太窄导致溢出/下溢”和“刻度太粗导致舍入”，说明为何输入、计算中间值和输出可能使用不同精度，并能读懂 `torch.autocast` 的基本用法。这里聚焦前向数值和混合精度；梯度裁剪、FP16 Loss Scaling 与训练循环放在规划中的 Chapter 16–17。

## What：浮点数是什么？

### 1. 科学计数法，但可用位数有限

把浮点数想成二进制科学计数法：符号决定正负，**指数**决定数值能覆盖多大的范围，**尾数**决定相近的数能分得多细。下面是常见格式的位数（不含隐藏的有效位）；较宽的指数意味着较大的动态范围，较多的尾数位意味着较密的刻度。

| 格式 | 总位数 | 符号/指数/尾数位 | 直观取舍 |
| --- | ---: | --- | --- |
| FP32（`torch.float32`） | 32 | 1 / 8 / 23 | 范围较宽，刻度较细；本章数值参照 |
| FP16（`torch.float16`） | 16 | 1 / 5 / 10 | 比 BF16 刻度细，但范围窄，易遇到溢出或下溢 |
| BF16（`torch.bfloat16`） | 16 | 1 / 8 / 7 | 范围接近 FP32，但刻度比 FP16 粗 |

三个重要数字可由 `torch.finfo(dtype)` 查询：`eps` 是 **1 与下一个可表示数的距离**，`tiny` 是最小**正规**正数，`max` 是最大有限数。`tiny` 不是最小非零数：有些设备/运算还能表示更小的非正规数。`torch.finfo(dtype).eps` 也不是 `nn.LayerNorm(eps=...)` 中为防止除零而设置的超参数。[PyTorch 类型信息](https://docs.pytorch.org/docs/stable/type_info.html)

```python
import torch
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    info = torch.finfo(dtype)
    print(dtype, info.eps, info.tiny, info.max)
```

**舍入**：实数落在两条刻度之间，只能存为附近的可表示数。例如 `1000` 与 `1001` 转成 BF16 后可能相同，而 FP16 仍能区分它们。**溢出**：结果大到超出范围，可能得到 `inf`；**下溢**：非零小数太小，在特定运算路径中可能变为零。`NaN` 表示无效数值结果，例如 `inf / inf`。具体舍入、非正规数和中间计算行为可能随硬件与算子而变。

### 2. 什么叫“数值稳定”？

数学上等价的写法，在有限精度的计算机里未必同样可靠。例如数学上 `softmax([1000,1001,1002])` 没问题；直接计算 `exp(1000)` 会溢出。先减去最大值，再求指数，数学结果不变，却让参与 `exp` 的数不大于 0：

$$
\operatorname{softmax}(x_i)=\frac{\exp(x_i-m)}{\sum_j\exp(x_j-m)},\qquad m=\max_j x_j.
$$

这里“稳定”是**改写计算路径，让中间值更容易保持有限**，不是给浮点格式增加有效位。PyTorch 的 `torch.softmax` 可直接调用；教学实验中的“直接 `exp`”只用于展示风险。即使结果有限，仍要与高精度参考比较误差。[PyTorch 数值精度说明](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)

## How：精度怎样影响 LLM 计算？

### 3. 输入、累加、输出是三个不同环节

看 `a @ b` 时至少问三件事：输入张量存为哪种 dtype？乘积与累加使用哪种精度？结果存为哪种 dtype？只看输入或输出的 `dtype`，无法推出内核所有中间值的精度。点积、注意力分数和归一化统计量都有求和过程；乘法单项可表示，也不保证和仍能表示。PyTorch 说明不同后端的算法与累加策略可能不同，因此数值实验要记录设备、版本和 dtype。[PyTorch 数值精度说明](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)

两个典型误区：

- **“BF16 范围宽，所以一定更准确。”**在较大的数附近，BF16 可能比 FP16 更早丢失微小差异。
- **“算出 `inf` 后转 FP32 就能修复。”**类型转换不会找回已经溢出或舍入的信息；应在敏感运算**之前**安排合适的范围和计算 dtype。

### 4. 归一化能帮什么，不能帮什么？

[Chapter 0 第 10 节](../chapter0_Normalization/README.md#10-bf16fp16-对归一化有什么影响)已有具体例子。LayerNorm/RMSNorm 可以在某个位置把过大的激活尺度带回较合适的范围，从而降低**后续**运算的溢出风险；它们不改变 FP16/BF16 的刻度密度，也不能恢复输入舍入时丢掉的差异。归一化自身的统计量仍可能需要更高精度；可学习缩放、线性层和残差相加仍可能再次放大激活。判断效果时分别检查有限值、相对参考的误差及训练表现，不把“归一化后约为 1”理解成整个网络始终安全。

### 5. 混合精度与 `autocast`

全部使用 FP32 简单，但显存和计算成本可能较高。混合精度让框架按**算子**选择执行 dtype：例如某些矩阵乘法适合低精度，而某些求和、归一化或损失计算可能采用更高精度路径。`torch.autocast` 会按设备和算子策略自动选择；**并非所有算子都会转为同一种 dtype**，具体策略可随设备和 PyTorch 版本变化。[PyTorch AMP 文档](https://docs.pytorch.org/docs/stable/amp.html)

```python
# model 保持 FP32 参数；输入也从 FP32 开始。
# CUDA/MPS 上先试 FP16；CPU 示例使用 BF16。
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
amp_dtype = torch.float16 if device.type in ("cuda", "mps") else torch.bfloat16
model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
x = torch.randn(2, 4)
model, x = model.to(device), x.to(device)
with torch.autocast(device_type=device.type, dtype=amp_dtype):
    y = model(x)
print(y.dtype)  # 这是最终输出 dtype，不代表所有内部算子的 dtype
```

`autocast` **不是**把模型整体调用 `.half()` 或 `.bfloat16()`：后者直接改变参数/输入的存储 dtype，还可能使敏感运算失去 FP32 输入。本章 notebook 用固定矩阵比较 FP32 与手动转换后的误差，再用同一小模型比较 FP32 与 autocast；CUDA/MPS 可用时还会计时。MPS 的算子策略可能与 CUDA 不同，须打印实际输出 dtype。**不预设低精度必定更快或占用更少内存**；框架还可能为转换、临时缓冲区和缓存额外分配内存。

### 6. PyTorch MPS 与 MLX

Apple silicon 上有两条独立路径：PyTorch 的 `device="mps"` 仍使用 PyTorch 张量与 `torch.autocast`；[MLX](https://ml-explore.github.io/mlx/build/html/) 使用 `mlx.core`/`mlx.nn` 自己的数组与模块。**MLX 不是 PyTorch 的一个 device 字符串**，不能把 `torch.Tensor.to("mlx")` 当作调用方式。notebook 的 MLX 可选实验分别使用 `mx.array`、`mx.finfo`、`mlx.nn.LayerNorm` 和 `mx.eval`，与 PyTorch 实验观察同样的舍入、尺度及有限值问题。MLX 惰性求值，计时时必须 `mx.eval`。安装方式和平台要求见 [MLX 官方安装文档](https://ml-explore.github.io/mlx/build/html/install.html)。

FP16 训练中的 `torch.amp.GradScaler` 用来降低小梯度下溢风险；它不修复前向计算中的溢出，也不是 BF16 一定需要的步骤。梯度缩放、反向传播以及裁剪顺序留给 Chapter 16。[PyTorch AMP 示例](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)

## Do：运行并解释实验

打开 [`ch15.ipynb`](./ch15.ipynb)，从上往下运行：

| 实验 | 主要 PyTorch 调用 | 该看什么 |
| --- | --- | --- |
| 1. 刻度与范围 | `torch.finfo`、`Tensor.to(dtype)` | `eps/tiny/max`；1000 附近的舍入 |
| 2. 下溢与溢出 | `Tensor.square`、`torch.isfinite` | 输入有限时，中间值是否仍有限 |
| 3. 稳定 softmax | `torch.exp`、`torch.softmax` | 同一数学目标的不同计算路径 |
| 4. 矩阵乘法误差 | `@`、`.double()`、`.float()` | 用 FP64 参考，比较绝对误差；不把一次结果推广到所有硬件 |
| 5. autocast | `torch.autocast`、`nn.Linear`、`nn.LayerNorm` | 记录各阶段输出 dtype；区分参数存储与算子输出 |
| 6. CUDA/MPS 性能（可选） | `torch.cuda.synchronize` / `torch.mps.synchronize` | 同一 PyTorch 模型的耗时；CUDA 记录峰值已分配显存，MPS 记录驱动当前分配量；CPU 会跳过 |
| 7. MLX（可选） | `mx.array`、`mx.finfo`、`mlx.nn.LayerNorm`、`mx.eval` | Apple silicon 上原生验证舍入、归一化及矩阵运算；未安装时提示并跳过 |

**如何看误差：**绝对误差是 `abs(结果 - 参考)`；当参考值接近零时，相对误差会被分母放大，不能单看百分比。本章同时观察最大/平均绝对误差和有限值。FP64 也不是“真实数学值”，只是这里更高精度的数值参照。CUDA 显存峰值与 MPS 驱动分配量口径不同，MLX 计时又是单个矩阵乘法；这些数字只供各自环境内观察，不能直接横向排名。

**练习：**把 softmax 的 logits 从 `[1000,1001,1002]` 改成 `[-1000,-999,-998]`，预测直接 `exp` 的结果；再把矩阵乘法的输入整体乘以 `1000`，观察 FP16 结果是否仍有限。notebook 末尾给出解释。

## 参考资料

- [PyTorch：数值精度](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
- [PyTorch：AMP/autocast](https://docs.pytorch.org/docs/stable/amp.html)
- [PyTorch：AMP 示例与梯度缩放](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
- [PyTorch：`torch.finfo` 类型信息](https://docs.pytorch.org/docs/stable/type_info.html)
- [MLX：设备与同步](https://ml-explore.github.io/mlx/build/html/python/devices_and_streams.html)
