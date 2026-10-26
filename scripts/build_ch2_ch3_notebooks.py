"""Build the small, output-free teaching notebooks for Chapters 2 and 3."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(True)}


def write(path: str, cells: list[dict]) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


write("chapter2_Initialization_Residuals/ch2.ipynb", [
    markdown("# Chapter 2：初始化与残差路径\n\n配套[教案](../chapter2_initialization_residuals.md)。按顺序运行，CPU 即可。图表使用 Plotly；需要 `torch` 和 `plotly`。随机种子只让本次演示易复现，不能代替多次试验。"),
    code("import math\nimport torch\nfrom torch import nn\nimport plotly.graph_objects as go\n\ntorch.manual_seed(7)\nprint('PyTorch:', torch.__version__)"),
    markdown("## 1. 检查权重形状与初始化分布\n\n先预测 `nn.Linear(64,128)` 的权重形状。截断正态的 `std` 是截断前分布的参数；观察实测值。"),
    code("layer = nn.Linear(64, 128)\nassert tuple(layer.weight.shape) == (128, 64)\nw = torch.empty(100_000)\nnn.init.trunc_normal_(w, mean=0.0, std=1.0, a=-2.0, b=2.0)\nprint('shape:', tuple(layer.weight.shape))\nprint('truncated range/std:', float(w.min()), float(w.max()), float(w.std()))\nassert w.min() >= -2 and w.max() <= 2 and w.std() < 1"),
    markdown("## 2. 比较固定正态、Xavier 和 He\n\n固定输入为单位方差，宽度依次为 16、64、256。表和图记录激活函数之前的线性层输出标准差；He 在等宽时接近根号 2 是预期现象，因为这里还未经过 ReLU。这不代表完整 Transformer。"),
    code("widths = [16, 64, 256]\nmethods = ['fixed 0.1', 'Xavier', 'He/ReLU']\nrows = []\nfor d in widths:\n    x = torch.randn(4096, d)\n    for name in methods:\n        layer = nn.Linear(d, d, bias=False)\n        with torch.no_grad():\n            if name == 'fixed 0.1':\n                nn.init.normal_(layer.weight, std=0.1)\n            elif name == 'Xavier':\n                nn.init.xavier_normal_(layer.weight)\n            else:\n                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')\n            weight_std = float(layer.weight.std())\n            out_std = float(layer(x).std())\n            rows.append((d, name, weight_std, out_std))\nfor row in rows:\n    print('width=%3d  %-10s weight_std=%.3f  output_std=%.3f' % row)\nassert all(math.isfinite(row[3]) for row in rows)"),
    code("fig = go.Figure()\nfor name in methods:\n    chosen = [r for r in rows if r[1] == name]\n    fig.add_trace(go.Scatter(x=[r[0] for r in chosen], y=[r[3] for r in chosen], mode='lines+markers', name=name))\nfig.update_layout(title='初始化时线性层输出的标准差', xaxis_title='输入/输出宽度 d', yaxis_title='输出标准差（无单位）')\nfig.show()"),
    markdown("## 3. 残差的局部梯度\n\n`F(x)=0.1x` 时，纯分支导数是 0.1，加上直接路径后是 1.1。"),
    code("x = torch.tensor(2.0, requires_grad=True)\ny = x + 0.1 * x\ny.backward()\nprint('output=', float(y.detach()), 'input gradient=', float(x.grad))\nassert abs(float(x.grad) - 1.1) < 1e-6"),
    markdown("## 4. 多层路径：记录激活和梯度\n\n同宽线性子层分别构成纯堆叠、Pre-LN、Post-LN 和缩放的 Pre-LN。每层随机权重重新生成，所以曲线只可看数量级与趋势；不训练参数。`h.grad` 是该层输出在最终平方损失下的梯度。"),
    code("def trace_stack(kind, depth=16, width=32, seed=7):\n    torch.manual_seed(seed)\n    layers = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(depth)])\n    norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])\n    for layer in layers:\n        nn.init.xavier_normal_(layer.weight)\n    x = torch.randn(8, width, requires_grad=True)\n    h = x\n    states = [h]\n    for layer, norm in zip(layers, norms):\n        if kind == 'plain':\n            h = layer(h)\n        elif kind == 'pre':\n            h = h + layer(norm(h))\n        elif kind == 'post':\n            h = norm(h + layer(h))\n        elif kind == 'scaled-pre':\n            h = h + layer(norm(h)) / math.sqrt(depth)\n        else:\n            raise ValueError(kind)\n        h.retain_grad()\n        states.append(h)\n    target = torch.linspace(-1, 1, width).expand_as(h)\n    (h - target).square().mean().backward()\n    rms = lambda t: float(t.square().mean().sqrt())\n    return [rms(s.detach()) for s in states], [rms(s.grad) for s in states]\n\nkinds = ['plain', 'pre', 'post', 'scaled-pre']\ntraces = {kind: trace_stack(kind) for kind in kinds}\nfor kind, (activation, gradient) in traces.items():\n    print(kind, 'last activation RMS=', round(activation[-1], 4), 'input gradient RMS=', round(gradient[0], 6))\n    assert all(math.isfinite(v) for v in activation + gradient)"),
    code("for metric, idx in [('激活 RMS', 0), ('梯度 RMS', 1)]:\n    fig = go.Figure()\n    for kind in kinds:\n        fig.add_trace(go.Scatter(x=list(range(len(traces[kind][idx]))), y=traces[kind][idx], mode='lines+markers', name=kind))\n    fig.update_layout(title=metric + ' 随深度的变化（随机初始化）', xaxis_title='层数（0 为输入）', yaxis_title=metric + '（无单位）')\n    fig.show()"),
    markdown("## 5. 改深度和种子\n\n改变 `depth` 与 `seed`，看单条曲线是否稳定。由于损失是对固定目标的均方误差，梯度大小还受最终激活大小影响；不能直接把曲线当作训练好坏。"),
    code("for depth in [4, 16, 32]:\n    for seed in [7, 11, 19]:\n        a, g = trace_stack('scaled-pre', depth=depth, seed=seed)\n        print('depth=%2d seed=%2d final_RMS=%.3f input_grad_RMS=%.5f' % (depth, seed, a[-1], g[0]))"),
    markdown("**思考**：如果把 `scaled-pre` 的系数改成 1，会怎样？如果把所有隐藏层权重设成零，它和仅把残差输出投影设零有什么区别？参见[教案的检查理解](../chapter2_initialization_residuals.md#检查理解)。")
])

write("chapter3_Activations_FFN/ch3.ipynb", [
    markdown("# Chapter 3：激活函数与门控前馈层\n\n配套[教案](../chapter3_activations_ffn.md)。按顺序运行，CPU 即可。图表使用 Plotly；需要 `torch` 和 `plotly`。"),
    code("import torch\nfrom torch import nn\nimport torch.nn.functional as F\nimport plotly.graph_objects as go\n\ntorch.manual_seed(7)\nprint('PyTorch:', torch.__version__)"),
    markdown("## 1. 从饱和到门控：函数与导数曲线\n\n对比 Sigmoid/Tanh 在区间两端的饱和、ReLU 负半轴的零梯度，以及 GELU/SiLU 的平滑变化。`autograd` 对每个采样点求导；ReLU 在 0 处不可导，PyTorch 采用约定的梯度。"),
    code("""grid = torch.linspace(-4, 4, 401)
functions = {
    'Sigmoid': torch.sigmoid,
    'Tanh': torch.tanh,
    'ReLU': F.relu,
    'GELU': F.gelu,
    'SiLU': F.silu,
}
curves = {}
for name, fn in functions.items():
    probe = grid.detach().clone().requires_grad_()
    output = fn(probe)
    derivative = torch.autograd.grad(output.sum(), probe)[0]
    curves[name] = (output.detach(), derivative.detach())
    print(name, 'grad(-1)=', round(float(derivative[150]), 4),
          'grad(4)=', round(float(derivative[-1]), 4))
for title, index in [('激活函数', 0), ('导数', 1)]:
    fig = go.Figure()
    for name, pair in curves.items():
        fig.add_trace(go.Scatter(x=grid.tolist(), y=pair[index].tolist(),
                                 mode='lines', name=name))
    fig.update_layout(title=title, xaxis_title='输入 x（无单位）',
                      yaxis_title=title + '（无单位）')
    fig.show()"""),
    markdown("## 2. 普通 FFN 和 SwiGLU：检查所有形状\n\n`[batch, seq, d]` 的最后一轴参与投影。两个模型都使用无偏置线性层以便准确核对权重参数公式。"),
    code("class PlainFFN(nn.Module):\n    def __init__(self, d, m, activation=nn.GELU()):\n        super().__init__()\n        self.up = nn.Linear(d, m, bias=False)\n        self.activation = activation\n        self.down = nn.Linear(m, d, bias=False)\n\n    def forward(self, x):\n        return self.down(self.activation(self.up(x)))\n\nclass SwiGLUFFN(nn.Module):\n    def __init__(self, d, m):\n        super().__init__()\n        self.gate = nn.Linear(d, m, bias=False)\n        self.value = nn.Linear(d, m, bias=False)\n        self.down = nn.Linear(m, d, bias=False)\n\n    def forward(self, x):\n        return self.down(F.silu(self.gate(x)) * self.value(x))\n\nx = torch.randn(2, 3, 12)\nplain = PlainFFN(12, 48)\ngated = SwiGLUFFN(12, 32)\nprint('plain up:', tuple(plain.up(x).shape), 'output:', tuple(plain(x).shape))\nprint('gate/value:', tuple(gated.gate(x).shape), tuple(gated.value(x).shape), 'output:', tuple(gated(x).shape))\nassert plain(x).shape == gated(x).shape == x.shape"),
    markdown("## 3. 相近权重预算的比较\n\n普通 FFN：`2dm`；SwiGLU：`3dm`。本例 `d=12`，中间宽度 48 与 32 时权重都为 1152 个。"),
    code("count = lambda model: sum(p.numel() for p in model.parameters())\nprint('plain:', count(plain), 'expected:', 2 * 12 * 48)\nprint('gated:', count(gated), 'expected:', 3 * 12 * 32)\nassert count(plain) == count(gated) == 1152"),
    markdown("## 4. 梯度要经过三条投影和残差路径\n\n使用一个标量平方损失检查 `backward()` 连通性。"),
    code("probe = x.detach().clone().requires_grad_()\nloss = gated(probe).square().mean()\nloss.backward()\nfor name, parameter in gated.named_parameters():\n    print(name, 'gradient norm=', float(parameter.grad.norm()))\n    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()\nassert probe.grad is not None and torch.isfinite(probe.grad).all()\n\nclass PreNormFFN(nn.Module):\n    def __init__(self, d, m):\n        super().__init__()\n        self.norm = nn.RMSNorm(d)\n        self.ffn = SwiGLUFFN(d, m)\n\n    def forward(self, x):\n        return x + self.ffn(self.norm(x))\n\nblock = PreNormFFN(12, 32)\nprobe = x.detach().clone().requires_grad_()\nblock(probe).square().mean().backward()\nprint('Pre-RMSNorm residual input gradient norm=', float(probe.grad.norm()))\nassert probe.grad is not None and torch.isfinite(probe.grad).all()"),
    markdown("## 5. 固定合成数据上的小型拟合\n\n为了检查训练代码能运行，构造 128 个 12 维向量及固定非线性目标。两个模型权重预算相同、训练步数相同；单个随机种子的短程曲线不是架构优劣基准。"),
    code("torch.manual_seed(21)\nfeatures = torch.randn(128, 12)\nwith torch.no_grad():\n    target = torch.sin(features @ torch.randn(12, 12) / 3)\n\ndef fit(model, steps=60):\n    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)\n    history = []\n    for _ in range(steps):\n        optimizer.zero_grad(set_to_none=True)\n        loss = F.mse_loss(model(features), target)\n        loss.backward()\n        optimizer.step()\n        history.append(float(loss.detach()))\n    return history\n\ntorch.manual_seed(7)\nplain_loss = fit(PlainFFN(12, 48))\ntorch.manual_seed(7)\ngated_loss = fit(SwiGLUFFN(12, 32))\nprint('plain first/last:', plain_loss[0], plain_loss[-1])\nprint('gated first/last:', gated_loss[0], gated_loss[-1])\nassert plain_loss[-1] < plain_loss[0] and gated_loss[-1] < gated_loss[0]\nfig = go.Figure()\nfig.add_trace(go.Scatter(x=list(range(1, 61)), y=plain_loss, mode='lines', name='普通 GELU FFN'))\nfig.add_trace(go.Scatter(x=list(range(1, 61)), y=gated_loss, mode='lines', name='SwiGLU FFN'))\nfig.update_layout(title='单种子合成拟合演示', xaxis_title='优化步骤', yaxis_title='均方误差（无单位）')\nfig.show()"),
    markdown("**思考**：若两个模型都取 `m=48`，参数量还能比较吗？`x + FFN(Norm(x))` 中哪条是直接路径？参见[教案的检查理解](../chapter3_activations_ffn.md#检查理解)。")
])

# The root lesson is imported into the website; the chapter-folder README is
# the same lesson with links adjusted for readers browsing that folder.
for number, folder in [(2, "chapter2_Initialization_Residuals"),
                       (3, "chapter3_Activations_FFN")]:
    filename = ("chapter2_initialization_residuals.md" if number == 2
                else "chapter3_activations_ffn.md")
    text = (ROOT / filename).read_text(encoding="utf-8")
    text = text.replace(f"{folder}/ch{number}.ipynb", f"./ch{number}.ipynb")
    text = text.replace(f"{folder}/activation-functions.svg", "./activation-functions.svg")
    text = text.replace(f"{folder}/activation-derivatives.svg", "./activation-derivatives.svg")
    text = re.sub(r"\]\((chapter\d+_[^/)]+\.md)(#[^)]+)?\)",
                  lambda m: "](../" + m[1] + (m[2] or "") + ")", text)
    text = text.replace("](README.md#课程路线规划)", "](../README.md#课程路线规划)")
    (ROOT / folder / "README.md").write_text(text, encoding="utf-8")
