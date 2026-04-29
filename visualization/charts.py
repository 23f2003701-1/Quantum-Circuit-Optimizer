"""
Visualization module.
All chart and figure generation for the Streamlit dashboard.
Uses Plotly for interactivity and Matplotlib for circuit drawings.
"""

import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from qiskit.circuit import QuantumCircuit
from qiskit.visualization import circuit_drawer

# ── Color palette ────────────────────────────────────────────────────────────

LEVEL_COLORS = {
    -1: "#94a3b8",   # original = slate
    0:  "#ef4444",   # red
    1:  "#f97316",   # orange
    2:  "#eab308",   # yellow
    3:  "#22c55e",   # green
}

LEVEL_LABELS = {
    -1: "Original",
    0:  "Level 0",
    1:  "Level 1",
    2:  "Level 2",
    3:  "Level 3",
}

BG = "#0f1117"
CARD = "#1e2130"
TEXT = "#e2e8f0"
ACCENT = "#7c3aed"


# ── Circuit drawing ──────────────────────────────────────────────────────────

def draw_circuit_to_image(qc: QuantumCircuit, title: str = "") -> bytes:
    """Draw a QuantumCircuit and return it as PNG bytes."""
    style = {
        "backgroundcolor": "#1a1f2e",
        "textcolor": "#e2e8f0",
        "gatefacecolor": "#3b4163",
        "gatetextcolor": "#f8fafc",
        "subtextcolor": "#94a3b8",
        "linecolor": "#4a5568",
        "creglinecolor": "#7c3aed",
        "latexdrawerstyle": True,
        "displaycolor": {
            "cx": ("#7c3aed", "#f8fafc"),
            "h":  ("#0ea5e9", "#f8fafc"),
            "x":  ("#ef4444", "#f8fafc"),
            "rz": ("#22c55e", "#f8fafc"),
            "rx": ("#f97316", "#f8fafc"),
            "swap": ("#eab308", "#0f1117"),
        },
    }

    fig = qc.draw(
        output="mpl",
        style=style,
        fold=40,
        plot_barriers=False,
    )

    if title:
        fig.suptitle(title, color=TEXT, fontsize=12, fontweight="bold", y=1.01)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor="#1a1f2e", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Metrics bar chart ────────────────────────────────────────────────────────

def plot_metrics_comparison(metrics_dict: dict, original_metrics) -> go.Figure:
    """
    Grouped bar chart: Original vs all optimization levels.
    Metrics: depth, total gates, CX count, SWAP count, weighted cost.
    """
    all_entries = {-1: original_metrics, **metrics_dict}
    labels = [LEVEL_LABELS[k] for k in all_entries]
    colors = [LEVEL_COLORS[k] for k in all_entries]

    metric_names = ["Depth", "Total Gates", "CX Gates", "SWAP Gates", "Weighted Cost"]
    metric_keys = ["depth", "total_gates", "cx_count", "swap_count", "weighted_cost"]

    fig = make_subplots(
        rows=1, cols=5,
        subplot_titles=metric_names,
        shared_yaxes=False,
    )

    for col_idx, (name, key) in enumerate(zip(metric_names, metric_keys), 1):
        values = [getattr(m, key) for m in all_entries.values()]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=[f"{v:.0f}" for v in values],
                textposition="outside",
                showlegend=False,
                name=name,
            ),
            row=1, col=col_idx,
        )

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        font=dict(color=TEXT, family="monospace"),
        height=380,
        margin=dict(t=50, b=20, l=10, r=10),
        title=dict(
            text="Circuit Metrics — Original vs Optimization Levels",
            font=dict(size=15, color=TEXT),
        ),
    )
    fig.update_xaxes(tickfont=dict(size=9))
    fig.update_yaxes(gridcolor="#2d3748", gridwidth=0.5)

    return fig


# ── Improvement trend line ───────────────────────────────────────────────────

def plot_improvement_trend(metrics_dict: dict, original_metrics) -> go.Figure:
    """Line chart showing how each metric improves across optimization levels 0→3."""
    levels = [0, 1, 2, 3]
    orig_depth = original_metrics.depth
    orig_cx = original_metrics.cx_count
    orig_cost = original_metrics.weighted_cost

    def pct_change(original, new):
        if original == 0:
            return 0
        return (original - new) / original * 100

    depth_pcts = [pct_change(orig_depth, metrics_dict[l].depth) for l in levels]
    cx_pcts = [pct_change(orig_cx, metrics_dict[l].cx_count) for l in levels]
    cost_pcts = [pct_change(orig_cost, metrics_dict[l].weighted_cost) for l in levels]

    fig = go.Figure()

    for name, values, color in [
        ("Depth reduction %", depth_pcts, "#0ea5e9"),
        ("CX reduction %", cx_pcts, "#7c3aed"),
        ("Cost reduction %", cost_pcts, "#22c55e"),
    ]:
        fig.add_trace(go.Scatter(
            x=levels,
            y=values,
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=2.5),
            marker=dict(size=9, symbol="circle"),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="#4a5568", line_width=1)

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        font=dict(color=TEXT, family="monospace"),
        height=340,
        xaxis=dict(
            title="Optimization Level",
            tickvals=[0, 1, 2, 3],
            ticktext=["Level 0", "Level 1", "Level 2", "Level 3"],
            gridcolor="#2d3748",
        ),
        yaxis=dict(
            title="% reduction from original (↑ better)",
            gridcolor="#2d3748",
            zeroline=False,
        ),
        title="Improvement Trend Across Optimization Levels",
        legend=dict(bgcolor=CARD, bordercolor="#2d3748", borderwidth=1),
        margin=dict(t=50, b=40, l=60, r=20),
    )

    return fig


# ── Gate breakdown donut ─────────────────────────────────────────────────────

def plot_gate_breakdown(metrics, label: str) -> go.Figure:
    """Donut chart of gate types for a single circuit version."""
    breakdown = metrics.gate_breakdown
    if not breakdown:
        return go.Figure()

    labels = list(breakdown.keys())
    values = list(breakdown.values())

    gate_colors = [
        "#7c3aed", "#0ea5e9", "#22c55e", "#f97316",
        "#ef4444", "#eab308", "#ec4899", "#14b8a6",
        "#8b5cf6", "#06b6d4",
    ]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker_colors=gate_colors[:len(labels)],
        textinfo="label+percent",
        textfont=dict(size=11),
    ))

    fig.update_layout(
        paper_bgcolor=BG,
        font=dict(color=TEXT, family="monospace"),
        height=300,
        margin=dict(t=30, b=10, l=10, r=10),
        showlegend=False,
        title=dict(text=f"Gate Breakdown — {label}", font=dict(size=13, color=TEXT)),
        annotations=[dict(
            text=f"{sum(values)}<br>gates",
            x=0.5, y=0.5,
            font=dict(size=14, color=TEXT),
            showarrow=False,
        )],
    )

    return fig


# ── Stochastic distribution ──────────────────────────────────────────────────

def plot_stochastic_distribution(stochastic_results: dict) -> go.Figure:
    """Violin + scatter plot of metric distributions across seeds."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Circuit Depth", "CX Gates", "SWAP Gates"],
    )

    datasets = [
        ("depths", "#0ea5e9", "rgba(14, 165, 233, 0.27)"),
        ("cx_counts", "#7c3aed", "rgba(124, 58, 237, 0.27)"),
        ("swap_counts", "#22c55e", "rgba(34, 197, 94, 0.27)"),
    ]

    for col, (key, color, fillcolor_rgba) in enumerate(datasets, 1):
        vals = stochastic_results[key]
        seeds = stochastic_results["seeds"]

        fig.add_trace(go.Violin(
            y=vals,
            box_visible=True,
            meanline_visible=True,
            fillcolor=fillcolor_rgba,
            line_color=color,
            name=key.replace("_", " "),
            showlegend=False,
        ), row=1, col=col)

        fig.add_trace(go.Scatter(
            x=[0] * len(vals),
            y=vals,
            mode="markers",
            marker=dict(color=color, size=6, opacity=0.7),
            showlegend=False,
            hovertext=[f"Seed {s}: {v}" for s, v in zip(seeds, vals)],
        ), row=1, col=col)

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        font=dict(color=TEXT, family="monospace"),
        height=380,
        title="Stochastic Analysis: Metric Distribution Across Random Seeds (Level 3)",
        margin=dict(t=60, b=30, l=40, r=20),
    )
    fig.update_yaxes(gridcolor="#2d3748")
    fig.update_xaxes(showticklabels=False)

    return fig


# ── Coupling map visualization ───────────────────────────────────────────────

def plot_coupling_map(backend_name: str) -> go.Figure:
    """
    Draw the hardware coupling map as an interactive graph.
    Shows which qubit pairs are natively connected.
    """
    from analysis.transpiler import BACKENDS

    info = BACKENDS[backend_name]
    n = info["n_qubits"]
    coupling = info["coupling"]

    # Position qubits in a circle
    angles = [2 * np.pi * i / n for i in range(n)]
    xs = [np.cos(a) for a in angles]
    ys = [np.sin(a) for a in angles]

    fig = go.Figure()

    # Draw edges
    seen = set()
    for (u, v) in coupling:
        pair = tuple(sorted([u, v]))
        if pair in seen:
            continue
        seen.add(pair)
        fig.add_trace(go.Scatter(
            x=[xs[u], xs[v], None],
            y=[ys[u], ys[v], None],
            mode="lines",
            line=dict(color="#4a5568", width=2.5),
            showlegend=False,
        ))

    # Draw nodes
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        marker=dict(
            size=30,
            color=ACCENT,
            line=dict(color=TEXT, width=2),
        ),
        text=[f"Q{i}" for i in range(n)],
        textfont=dict(color="white", size=11),
        textposition="middle center",
        showlegend=False,
        hovertext=[f"Qubit {i}" for i in range(n)],
    ))

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT),
        height=320,
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis=dict(visible=False, range=[-1.4, 1.4]),
        yaxis=dict(visible=False, range=[-1.4, 1.4]),
        title=dict(text=f"Hardware Coupling Map — {backend_name}", font=dict(size=13, color=TEXT)),
    )

    return fig


# ── Custom pass transformation waterfall ────────────────────────────────────

def plot_custom_pass_waterfall(log: list) -> go.Figure:
    """Waterfall chart showing how each custom pass step changed gate count."""
    if not log:
        return go.Figure()

    stages = [r.stage for r in log]
    gate_counts = [r.before_total for r in log] + [log[-1].after_total]
    deltas = [r.after_total - r.before_total for r in log]

    colors = []
    for d in deltas:
        if d < 0:
            colors.append("#22c55e")   # green = improvement
        elif d > 0:
            colors.append("#ef4444")   # red = overhead
        else:
            colors.append("#64748b")   # grey = no change

    fig = go.Figure(go.Waterfall(
        name="Gate count",
        orientation="v",
        measure=["absolute"] + ["relative"] * len(deltas),
        x=["Initial"] + stages,
        y=[gate_counts[0]] + deltas,
        text=[str(gate_counts[0])] + [
            f"{'−' if d < 0 else '+' if d > 0 else ''}{abs(d)}" for d in deltas
        ],
        textposition="outside",
        connector=dict(line=dict(color="#4a5568", width=1)),
        increasing=dict(marker=dict(color="#ef4444")),
        decreasing=dict(marker=dict(color="#22c55e")),
        totals=dict(marker=dict(color="#7c3aed")),
    ))

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        font=dict(color=TEXT, family="monospace"),
        height=340,
        title="Custom Pass: Gate Count Waterfall",
        yaxis=dict(title="Total Gates", gridcolor="#2d3748"),
        xaxis=dict(gridcolor="#2d3748"),
        margin=dict(t=50, b=40, l=60, r=20),
    )

    return fig
