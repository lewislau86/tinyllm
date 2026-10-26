"""Render the Chapter 3 activation curves as a standalone SVG figure."""

from pathlib import Path

import plotly.graph_objects as go
import torch
import torch.nn.functional as F


values = torch.linspace(-4, 4, 801)
functions = (
    ("ReLU", F.relu, "#2563eb", "solid"),
    ("GELU", F.gelu, "#d97706", "dash"),
    ("SiLU", F.silu, "#7c3aed", "dot"),
)

figure = go.Figure()
with torch.no_grad():
    for name, function, color, dash in functions:
        figure.add_trace(go.Scatter(
            x=values.tolist(),
            y=function(values).tolist(),
            mode="lines",
            name=name,
            line={"color": color, "width": 3.5, "dash": dash},
            hovertemplate="x=%{x:.2f}<br>f(x)=%{y:.3f}<extra>%{fullData.name}</extra>",
        ))

figure.update_layout(
    width=960,
    height=480,
    margin={"l": 82, "r": 34, "t": 70, "b": 72},
    paper_bgcolor="white",
    plot_bgcolor="white",
    font={"family": "Microsoft YaHei, Noto Sans CJK SC, sans-serif", "size": 16, "color": "#334155"},
    legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": 1.05, "yanchor": "bottom"},
    xaxis={"title": "输入 x", "range": [-4, 4], "dtick": 1, "zeroline": True,
           "zerolinecolor": "#64748b", "gridcolor": "#e2e8f0"},
    yaxis={"title": "输出 f(x)", "range": [-0.8, 4.2], "dtick": 1, "zeroline": True,
           "zerolinecolor": "#64748b", "gridcolor": "#e2e8f0"},
)

output = Path(__file__).resolve().parents[1] / "chapter3_Activations_FFN" / "activation-functions.svg"
figure.write_image(str(output))
print(output)
