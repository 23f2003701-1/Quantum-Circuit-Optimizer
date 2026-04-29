"""
Quantum Circuit Optimizer — Streamlit Dashboard
================================================
A visual tool to analyze and optimize quantum circuits for NISQ hardware using Qiskit's transpilation pipeline.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import plotly.graph_objects as go
import time

from circuits.benchmarks import BENCHMARKS
from analysis.transpiler import (
    run_transpilation, run_stochastic_analysis,
    BACKENDS, TranspilationResult,
)
from analysis.explainer import (
    apply_custom_pass, build_diff_summary,
    explain_optimization_levels,
)
from visualization.charts import (
    draw_circuit_to_image,
    plot_metrics_comparison,
    plot_improvement_trend,
    plot_gate_breakdown,
    plot_stochastic_distribution,
    plot_coupling_map,
    plot_custom_pass_waterfall,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Quantum Circuit Optimizer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    background-color: #0f1117;
    color: #e2e8f0;
    font-family: 'Space Grotesk', sans-serif;
  }

  .main-header {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4c1d95 100%);
    border: 1px solid #4c1d95;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }

  .main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(124,58,237,0.3) 0%, transparent 70%);
    pointer-events: none;
  }

  .main-header h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
  }

  .main-header p {
    color: #a5b4fc;
    font-size: 1rem;
    margin: 0;
  }

  .metric-card {
    background: #1e2130;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
  }

  .metric-card:hover { border-color: #7c3aed; }

  .metric-card .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #f8fafc;
  }

  .metric-card .label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.25rem;
  }

  .metric-card .delta-good {
    font-size: 0.85rem;
    color: #22c55e;
    font-family: 'JetBrains Mono', monospace;
  }

  .metric-card .delta-bad {
    font-size: 0.85rem;
    color: #ef4444;
    font-family: 'JetBrains Mono', monospace;
  }

  .section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    color: #a5b4fc;
    border-left: 3px solid #7c3aed;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
  }

  .explain-card {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
  }

  .explain-card .stage {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #7c3aed;
    font-weight: 700;
    margin-bottom: 0.3rem;
  }

  .explain-card .desc {
    font-size: 0.9rem;
    color: #94a3b8;
    line-height: 1.5;
  }

  .level-pill {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
  }

  .pass-tag {
    display: inline-block;
    background: #1e2130;
    border: 1px solid #4a5568;
    border-radius: 6px;
    padding: 0.15rem 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #94a3b8;
    margin: 0.2rem;
  }

  .finance-banner {
    background: linear-gradient(135deg, #14532d 0%, #166534 100%);
    border: 1px solid #16a34a;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
  }

  .stButton > button {
    background: linear-gradient(135deg, #4c1d95, #7c3aed);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    transition: opacity 0.2s;
    width: 100%;
  }

  .stButton > button:hover { opacity: 0.85; }

  div[data-testid="stSidebar"] {
    background-color: #141721;
    border-right: 1px solid #2d3748;
  }

  .sidebar-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #7c3aed;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
    margin-top: 1rem;
  }

  .tab-description {
    color: #64748b;
    font-size: 0.88rem;
    margin-bottom: 1.2rem;
  }
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
  <h1>⚛️ Quantum Circuit Optimizer</h1>
  <p>
    Analyze, optimize and visualize quantum circuits for NISQ hardware.
    Compare optimization levels 0–3, inspect custom transpiler passes,
    and explore a QAOA portfolio optimization demo.
  </p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-section">⚙️ Circuit Selection</div>', unsafe_allow_html=True)

    circuit_name = st.selectbox(
        "Benchmark Circuit",
        options=list(BENCHMARKS.keys()),
        format_func=lambda x: f"{BENCHMARKS[x]['icon']} {x}",
    )

    bench = BENCHMARKS[circuit_name]
    st.caption(bench["description"])

    st.markdown('<div class="sidebar-section">🖥️ Hardware Backend</div>', unsafe_allow_html=True)

    backend_name = st.selectbox(
        "Target Backend",
        options=list(BACKENDS.keys()),
    )
    st.caption(BACKENDS[backend_name]["description"])

    st.markdown('<div class="sidebar-section">🎲 Transpiler Seed</div>', unsafe_allow_html=True)

    seed = st.slider("Random Seed", min_value=0, max_value=99, value=42,
                     help="Routing heuristics are stochastic. Different seeds → different SWAP overhead.")

    st.markdown('<div class="sidebar-section">🔬 Stochastic Analysis</div>', unsafe_allow_html=True)

    n_seeds = st.slider("Number of seeds to test", min_value=5, max_value=30, value=15,
                        help="Run the same circuit with N different seeds to see variance in results.")

    st.markdown("---")
    run_btn = st.button("🚀 Run Optimization Analysis", type="primary")


# ── Session state ─────────────────────────────────────────────────────────────

if "result" not in st.session_state:
    st.session_state.result = None
if "stochastic" not in st.session_state:
    st.session_state.stochastic = None
if "custom_pass_result" not in st.session_state:
    st.session_state.custom_pass_result = None
if "last_config" not in st.session_state:
    st.session_state.last_config = None


# ── Run analysis ──────────────────────────────────────────────────────────────

current_config = (circuit_name, backend_name, seed)

if run_btn or (st.session_state.result is None):
    with st.spinner("Running transpilation at all optimization levels…"):
        circuit = bench["fn"](n_qubits=bench["default_qubits"])
        result: TranspilationResult = run_transpilation(circuit, backend_name, seed=seed)
        st.session_state.result = result

    with st.spinner("Running stochastic seed analysis…"):
        stochastic = run_stochastic_analysis(
            circuit, backend_name, n_seeds=n_seeds, level=3
        )
        st.session_state.stochastic = stochastic

    with st.spinner("Applying custom optimization pass…"):
        custom_circuit, custom_log = apply_custom_pass(circuit)
        st.session_state.custom_pass_result = (custom_circuit, custom_log)

    st.session_state.last_config = current_config
    st.success("Analysis complete!", icon="✅")


result: TranspilationResult = st.session_state.result
stochastic = st.session_state.stochastic
custom_circuit, custom_log = st.session_state.custom_pass_result

if result is None:
    st.info("Configure your settings in the sidebar and click **Run Optimization Analysis**.")
    st.stop()


# ── Quick metrics summary ─────────────────────────────────────────────────────

orig = result.original_metrics
best = result.metrics[3]

st.markdown('<div class="section-header">📊 Optimization Summary</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

def delta_html(before, after, lower_is_better=True):
    diff = after - before
    if diff == 0:
        return f'<div class="delta-good">±0</div>'
    improved = (diff < 0) if lower_is_better else (diff > 0)
    cls = "delta-good" if improved else "delta-bad"
    sign = "−" if diff < 0 else "+"
    return f'<div class="{cls}">{sign}{abs(diff)}</div>'

for col, (label, orig_val, best_val) in zip(
    [col1, col2, col3, col4, col5],
    [
        ("Depth", orig.depth, best.depth),
        ("Total Gates", orig.total_gates, best.total_gates),
        ("CX Gates", orig.cx_count, best.cx_count),
        ("SWAP Gates", orig.swap_count, best.swap_count),
        ("Weighted Cost", int(orig.weighted_cost), int(best.weighted_cost)),
    ]
):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="value">{best_val}</div>
          <div class="label">{label}</div>
          {delta_html(orig_val, best_val)}
        </div>
        """, unsafe_allow_html=True)

st.caption("Values show Level 3 result. Delta vs original (green = improvement).")


# ── Main tabs ─────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "📈 Metrics Analysis",
    "🔬 Circuit Viewer",
    "🧩 Custom Pass Explainer",
    "🖥️ Hardware View",
    "📈 QAOA Finance Demo",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — METRICS ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.markdown('<div class="tab-description">Compare all metrics across optimization levels. '
                'See how depth, gate count, and cost change from Level 0 to Level 3.</div>',
                unsafe_allow_html=True)

    # Optimization level explainer
    st.markdown('<div class="section-header">🔎 What Each Level Does</div>', unsafe_allow_html=True)

    level_info = explain_optimization_levels()
    lc = st.columns(4)
    for col, (level, info) in zip(lc, level_info.items()):
        with col:
            st.markdown(f"""
            <div class="explain-card">
              <div class="stage">{info['label']}</div>
              <div class="desc">{info['description']}</div>
              <div style="margin-top:0.5rem;">
                {''.join(f'<span class="pass-tag">{p}</span>' for p in info['passes'])}
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 Metrics Comparison</div>', unsafe_allow_html=True)
    fig_metrics = plot_metrics_comparison(result.metrics, result.original_metrics)
    st.plotly_chart(fig_metrics, use_column_width=True)

    st.markdown('<div class="section-header">📉 Improvement Trend</div>', unsafe_allow_html=True)
    fig_trend = plot_improvement_trend(result.metrics, result.original_metrics)
    st.plotly_chart(fig_trend, use_column_width=True)

    st.markdown('<div class="section-header">🎲 Stochastic Variance (Level 3)</div>', unsafe_allow_html=True)
    st.caption(f"Same circuit, {n_seeds} different random seeds. "
               "Routing is stochastic — different seeds produce different SWAP overhead.")
    fig_stochastic = plot_stochastic_distribution(stochastic)
    st.plotly_chart(fig_stochastic, use_column_width=True)

    # Gate breakdown side by side
    st.markdown('<div class="section-header">🍩 Gate Type Breakdown</div>', unsafe_allow_html=True)
    ga, gb = st.columns(2)
    with ga:
        fig_orig = plot_gate_breakdown(result.original_metrics, "Original")
        st.plotly_chart(fig_orig, use_column_width=True)
    with gb:
        fig_opt = plot_gate_breakdown(result.metrics[3], "Level 3 Optimized")
        st.plotly_chart(fig_opt, use_column_width=True)

    # Diff summary
    st.markdown('<div class="section-header">📋 Optimization Diff Summary</div>', unsafe_allow_html=True)
    diff_lines = build_diff_summary(result.original_metrics, result.metrics, best_level=3)
    for line in diff_lines:
        st.markdown(f"- {line}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — CIRCUIT VIEWER
# ════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown('<div class="tab-description">Visual circuit diagrams before and after optimization. '
                'Select which levels to compare.</div>', unsafe_allow_html=True)

    selected_level = st.select_slider(
        "Select optimization level to compare with original",
        options=[0, 1, 2, 3],
        value=3,
        format_func=lambda x: f"Level {x}",
    )

    c_orig, c_opt = st.columns(2)

    with c_orig:
        st.markdown(f"**Original Circuit** — depth={result.original_metrics.depth}, "
                    f"gates={result.original_metrics.total_gates}")
        with st.spinner("Drawing original circuit…"):
            orig_img = draw_circuit_to_image(
                result.original_circuit,
                title=f"{circuit_name} — Original"
            )
        st.image(orig_img, use_column_width=True)

    with c_opt:
        opt_m = result.metrics[selected_level]
        st.markdown(f"**Level {selected_level} Optimized** — depth={opt_m.depth}, "
                    f"gates={opt_m.total_gates}")
        with st.spinner(f"Drawing Level {selected_level} circuit…"):
            opt_img = draw_circuit_to_image(
                result.transpiled[selected_level],
                title=f"{circuit_name} — Level {selected_level}"
            )
        st.image(opt_img, use_column_width=True)

    # Metric table for all levels
    st.markdown('<div class="section-header">📋 Full Metrics Table</div>', unsafe_allow_html=True)

    import pandas as pd
    rows = []
    for lvl, label in [(-1, "Original"), (0, "Level 0"), (1, "Level 1"),
                        (2, "Level 2"), (3, "Level 3")]:
        m = result.original_metrics if lvl == -1 else result.metrics[lvl]
        rows.append({
            "Level": label,
            "Depth": m.depth,
            "Total Gates": m.total_gates,
            "CX Gates": m.cx_count,
            "SWAP Gates": m.swap_count,
            "1Q Gates": m.single_qubit_gates,
            "Weighted Cost": f"{m.weighted_cost:.0f}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.highlight_min(
            subset=["Depth", "Total Gates", "CX Gates", "SWAP Gates", "Weighted Cost"],
            color="#14532d",
        ).highlight_max(
            subset=["Depth", "Total Gates", "CX Gates", "SWAP Gates", "Weighted Cost"],
            color="#450a0a",
        ),
        hide_index=True,
    )
    st.caption("Green = best (minimum), Red = worst (maximum) across all rows.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — CUSTOM PASS EXPLAINER
# ════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown('<div class="tab-description">'
                'See a custom transpiler pass built from scratch — step by step. '
                'Each pass is applied sequentially and its effect is explained.'
                '</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-card">
      <div class="stage">What is a transpiler pass?</div>
      <div class="desc">
        Qiskit's transpiler is a <strong>pipeline of passes</strong> — small, composable
        transformations applied to a circuit one at a time. Each pass has a single responsibility:
        cancel redundant gates, merge rotations, reroute qubits, etc.
        Below, we built a custom PassManager from 4 individual passes and applied it to your circuit.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🔄 Custom Pass Pipeline</div>', unsafe_allow_html=True)

    # Pass pipeline diagram
    pipeline_cols = st.columns(4)
    pass_colors = ["#7c3aed", "#0ea5e9", "#22c55e", "#f97316"]
    pass_labels = ["1. Remove\nBarriers", "2. CX\nCancellation",
                   "3. 1Q Gate\nMerging", "4. Commutative\nCancellation"]

    for col, (label, color) in zip(pipeline_cols, zip(pass_labels, pass_colors)):
        with col:
            st.markdown(f"""
            <div style="background:{color}22; border:1px solid {color};
                        border-radius:10px; padding:0.8rem; text-align:center;">
              <div style="font-family:'JetBrains Mono',monospace; font-size:0.8rem;
                          color:{color}; font-weight:700; white-space:pre-line;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📋 Step-by-Step Transformation Log</div>',
                unsafe_allow_html=True)

    cat_icons = {
        "gate_cancel": "❌",
        "merge": "🔀",
        "routing": "🗺️",
        "decompose": "⚙️",
    }

    for i, record in enumerate(custom_log):
        icon = cat_icons.get(record.category, "🔧")
        gate_delta = record.after_total - record.before_total
        depth_delta = record.after_depth - record.before_depth
        delta_str = (f"Gates: {record.before_total}→{record.after_total} "
                     f"({'−'+str(abs(gate_delta)) if gate_delta < 0 else '+'+str(gate_delta) if gate_delta > 0 else 'no change'}) | "
                     f"Depth: {record.before_depth}→{record.after_depth}")

        st.markdown(f"""
        <div class="explain-card" style="border-left:3px solid
             {'#22c55e' if gate_delta < 0 else '#ef4444' if gate_delta > 0 else '#4a5568'}">
          <div class="stage">{icon} Pass {i+1}: {record.stage}</div>
          <div style="font-family:'JetBrains Mono',monospace; font-size:0.78rem;
                      color:#64748b; margin-bottom:0.4rem;">{delta_str}</div>
          <div class="desc">{record.description}</div>
        </div>
        """, unsafe_allow_html=True)

    # Waterfall chart
    st.markdown('<div class="section-header">📊 Gate Count Waterfall</div>', unsafe_allow_html=True)
    fig_waterfall = plot_custom_pass_waterfall(custom_log)
    st.plotly_chart(fig_waterfall, use_column_width=True)

    # Show the custom circuit
    st.markdown('<div class="section-header">🔬 Custom-Optimized Circuit</div>', unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Before custom pass**")
        img_before = draw_circuit_to_image(result.original_circuit, "Original (abstract)")
        st.image(img_before, use_column_width=True)
    with cb:
        st.markdown("**After custom pass**")
        img_after = draw_circuit_to_image(custom_circuit, "After Custom PassManager")
        st.image(img_after, use_column_width=True)

    # Code snippet
    st.markdown('<div class="section-header">💻 Custom Pass Code</div>', unsafe_allow_html=True)
    st.code("""
from qiskit.circuit.library import CXGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    RemoveBarriers,
    InverseCancellation,    # Cancels CX · CX = Identity
    Optimize1qGates,        # Merges consecutive 1Q gates
    CommutativeCancellation,# Cancels commuting gate pairs
)

# Build a custom pass manager from scratch
pm = PassManager()
pm.append(RemoveBarriers())              # Remove programmer barriers
pm.append(InverseCancellation([CXGate()]))  # Cancel CX · CX = Identity
pm.append(Optimize1qGates())             # Merge consecutive 1Q gates → 1 gate
pm.append(CommutativeCancellation())     # Cancel commuting gate pairs

# Apply to your circuit
optimized_circuit = pm.run(your_circuit)
""", language="python")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — HARDWARE VIEW
# ════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.markdown('<div class="tab-description">'
                'Visualize the hardware topology and understand how connectivity '
                'constraints force SWAP insertions during routing.'
                '</div>', unsafe_allow_html=True)

    hw_a, hw_b = st.columns([1, 1])

    with hw_a:
        st.markdown('<div class="section-header">🖥️ Coupling Map</div>', unsafe_allow_html=True)
        fig_coupling = plot_coupling_map(backend_name)
        st.plotly_chart(fig_coupling, use_column_width=True)

        info = BACKENDS[backend_name]
        st.markdown(f"""
        <div class="explain-card">
          <div class="stage">Backend: {backend_name}</div>
          <div class="desc">
            <b>Qubits:</b> {info['n_qubits']}<br>
            <b>Connectivity:</b> {len(info['coupling']) // 2} native qubit pairs<br>
            <b>Description:</b> {info['description']}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with hw_b:
        st.markdown('<div class="section-header">🗺️ SWAP Overhead by Backend</div>',
                    unsafe_allow_html=True)

        import plotly.graph_objects as go

        backend_names = list(BACKENDS.keys())
        circuit = BENCHMARKS[circuit_name]["fn"](
            n_qubits=BENCHMARKS[circuit_name]["default_qubits"]
        )

        swap_counts = []
        cx_counts = []
        for bk in backend_names:
            try:
                r = run_transpilation(circuit, bk, seed=seed)
                swap_counts.append(r.metrics[3].swap_count)
                cx_counts.append(r.metrics[3].cx_count)
            except Exception:
                swap_counts.append(0)
                cx_counts.append(0)

        short_labels = [b.split("(")[0].strip() + "\n(" + b.split("(")[1] for b in backend_names]

        fig_hw = go.Figure()
        fig_hw.add_trace(go.Bar(
            name="SWAP Gates", x=backend_names, y=swap_counts,
            marker_color="#ef4444",
            text=swap_counts, textposition="outside",
        ))
        fig_hw.add_trace(go.Bar(
            name="CX Gates", x=backend_names, y=cx_counts,
            marker_color="#7c3aed",
            text=cx_counts, textposition="outside",
        ))
        fig_hw.update_layout(
            paper_bgcolor="#0f1117",
            plot_bgcolor="#1e2130",
            font=dict(color="#e2e8f0", family="monospace"),
            height=320,
            barmode="group",
            title="Routing Cost Across All Backends (Level 3)",
            xaxis=dict(tickangle=-15, tickfont=dict(size=9)),
            yaxis=dict(gridcolor="#2d3748"),
            legend=dict(bgcolor="#1e2130"),
            margin=dict(t=50, b=60, l=40, r=20),
        )
        st.plotly_chart(fig_hw, use_column_width=True)

    st.markdown('<div class="section-header">📚 Why Connectivity Matters</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-card">
      <div class="stage">The SWAP Problem</div>
      <div class="desc">
        Real quantum hardware only allows two-qubit gates between <strong>physically connected</strong>
        qubit pairs. If your circuit needs a CX between qubit 0 and qubit 4, but they're not
        directly connected, the compiler must insert <strong>SWAP gates</strong> to move the
        quantum state closer.<br><br>
        A SWAP gate = 3 CX gates. So routing overhead is very expensive.
        The transpiler tries to find the best initial qubit layout and SWAP strategy to
        minimize this cost — but it's an NP-hard problem solved heuristically.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — QAOA FINANCE DEMO
# ════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown("""
    <div class="finance-banner">
      <strong>📈 Finance Application Demo</strong> — QAOA Portfolio Optimization<br>
      <span style="font-size:0.85rem; color:#86efac;">
        This section connects quantum circuit optimization to a real finance use case:
        selecting the optimal subset of assets to minimize correlated risk using QAOA.
      </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### What is QAOA Portfolio Optimization?

    **QAOA** (Quantum Approximate Optimization Algorithm) is a NISQ algorithm for
    solving combinatorial optimization problems. In finance, we can model
    **portfolio selection** as a graph problem:

    - Each **qubit = one asset** (e.g. stock)
    - **Edges = correlated pairs** (assets that move together = risk)
    - The algorithm finds a portfolio that minimizes pairwise correlation (risk)

    The QAOA circuit alternates between a **cost layer** (encodes the problem)
    and a **mixer layer** (explores the solution space).
    """)

    st.markdown('<div class="section-header">📊 The 4-Asset Portfolio Problem</div>',
                unsafe_allow_html=True)

    assets_col, graph_col = st.columns([1, 1])

    with assets_col:
        import pandas as pd
        asset_data = pd.DataFrame({
            "Asset": ["AAPL", "MSFT", "TSLA", "GOOGL"],
            "Qubit": ["Q0", "Q1", "Q2", "Q3"],
            "Sector": ["Tech", "Tech", "EV/Tech", "Tech"],
            "Correlation pairs": ["(Q0,Q1), (Q0,Q3)", "(Q1,Q0), (Q1,Q2)", "(Q2,Q1), (Q2,Q3)", "(Q3,Q0), (Q3,Q2)"],
        })
        st.dataframe(asset_data, hide_index=True)
        st.markdown("""
        <div class="explain-card">
          <div class="stage">Objective</div>
          <div class="desc">
            Find a portfolio (subset of assets) such that
            the total pairwise correlation cost is minimized.
            Each edge in the graph represents a correlated pair that adds risk
            if both assets are selected.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with graph_col:
        # Draw the asset correlation graph
        import plotly.graph_objects as go
        fig_assets = go.Figure()

        positions = {"AAPL": (0, 1), "MSFT": (1, 1), "TSLA": (1, 0), "GOOGL": (0, 0)}
        edges = [("AAPL", "MSFT"), ("MSFT", "TSLA"), ("TSLA", "GOOGL"), ("AAPL", "GOOGL")]

        for u, v in edges:
            x0, y0 = positions[u]
            x1, y1 = positions[v]
            fig_assets.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(color="#22c55e", width=3),
                showlegend=False,
            ))

        for asset, (x, y) in positions.items():
            fig_assets.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                marker=dict(size=45, color="#14532d", line=dict(color="#22c55e", width=2)),
                text=[asset],
                textfont=dict(color="white", size=11, family="JetBrains Mono"),
                textposition="middle center",
                showlegend=False,
            ))

        fig_assets.update_layout(
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            height=280,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(visible=False, range=[-0.3, 1.3]),
            yaxis=dict(visible=False, range=[-0.3, 1.3]),
            font=dict(color="#e2e8f0"),
            title=dict(text="Asset Correlation Graph", font=dict(size=13, color="#86efac")),
        )
        st.plotly_chart(fig_assets, use_column_width=True)

    # Now run the QAOA circuit through the optimizer
    st.markdown('<div class="section-header">⚛️ QAOA Circuit Optimization</div>',
                unsafe_allow_html=True)

    from circuits.benchmarks import make_qaoa

    qaoa_circuit = make_qaoa(n_qubits=4, p_layers=1)
    with st.spinner("Transpiling QAOA circuit…"):
        qaoa_result = run_transpilation(qaoa_circuit, backend_name, seed=seed)

    qa, qb = st.columns(2)
    with qa:
        st.markdown("**QAOA Circuit — Original**")
        qaoa_orig_img = draw_circuit_to_image(qaoa_circuit, "QAOA Portfolio Circuit")
        st.image(qaoa_orig_img, use_column_width=True)

    with qb:
        st.markdown("**QAOA Circuit — Level 3 Optimized**")
        qaoa_opt_img = draw_circuit_to_image(
            qaoa_result.transpiled[3], "QAOA Optimized for Hardware"
        )
        st.image(qaoa_opt_img, use_column_width=True)

    # QAOA metrics
    st.markdown('<div class="section-header">📊 QAOA Optimization Impact</div>',
                unsafe_allow_html=True)

    fig_qaoa = plot_metrics_comparison(qaoa_result.metrics, qaoa_result.original_metrics)
    st.plotly_chart(fig_qaoa, use_column_width=True)

    st.markdown("""
    <div class="explain-card">
      <div class="stage">Why optimize QAOA circuits?</div>
      <div class="desc">
        On real NISQ hardware, every additional gate introduces <strong>noise and decoherence</strong>.
        An unoptimized QAOA circuit may fail entirely due to accumulated errors before the
        computation completes. Circuit optimization directly improves the quality
        of quantum computation results — making the difference between a meaningful
        answer and random noise.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4a5568; font-size:0.8rem; font-family:'JetBrains Mono',monospace;">
  ⚛️ Quantum Circuit Optimizer · Built with Qiskit + Streamlit ·
  Course Project: Hardware-Aware Quantum Circuit Optimization
</div>
""", unsafe_allow_html=True)
