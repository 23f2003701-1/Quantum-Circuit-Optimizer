# ⚛️ Quantum Circuit Optimizer

**Course Project: Hardware-Aware Quantum Circuit Optimization and Visualization**

A Streamlit web app that analyzes and optimizes quantum circuits for NISQ hardware
using Qiskit's transpilation pipeline. Compare optimization levels 0–3, inspect
custom transpiler passes step-by-step, and explore a QAOA portfolio optimization demo.

---

## 🗂️ Project Structure

```
quantum_optimizer/
│
├── app.py                        ← Main Streamlit dashboard (run this)
├── requirements.txt              ← All Python dependencies
│
├── circuits/
│   └── benchmarks.py             ← GHZ, QFT, Grover, QAOA circuit definitions
│
├── analysis/
│   ├── transpiler.py             ← Transpilation engine, metrics, stochastic analysis
│   └── explainer.py             ← Custom pass pipeline + diff/explainability layer
│
└── visualization/
    └── charts.py                 ← All Plotly/Matplotlib chart functions
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📋 Features

| Feature | Description |
|---|---|
| **4 Benchmark Circuits** | GHZ State, QFT, Grover's Search, QAOA Portfolio |
| **4 Hardware Backends** | Star, Linear, Heavy-Hex, Fully-Connected topologies |
| **Optimization Levels 0–3** | Full Qiskit transpilation pipeline comparison |
| **Metrics Engine** | Depth, total gates, CX gates, SWAP count, weighted cost |
| **Custom Pass Explainer** | Step-by-step custom PassManager with human-readable log |
| **Stochastic Analysis** | Seed variance — routing is random, results differ! |
| **Circuit Viewer** | Side-by-side before/after circuit diagrams |
| **Hardware Map** | Interactive coupling graph + SWAP overhead by backend |
| **QAOA Finance Demo** | 4-asset portfolio optimization use case |

---

## 🧠 Key Concepts Explained

### Optimization Levels

| Level | What it does |
|---|---|
| 0 | Only layout + routing. No optimization passes. |
| 1 | + Single-qubit gate merging, redundant gate removal |
| 2 | + Commutative cancellation, noise-aware layout |
| 3 | + Block resynthesis, Clifford simplification, Sabre routing |

### Weighted Cost Function

```
cost = depth × 1 + CX_gates × 3 + SWAP_gates × 5
```

SWAPs are weighted highest because each SWAP = 3 CX gates on hardware.

### Custom Pass Pipeline

```python
PassManager([
    RemoveBarriers(),                  # Remove programmer hints
    InverseCancellation([CXGate()]),   # Cancel CX · CX = I
    Optimize1qGates(),                 # Merge sequences of 1Q gates
    CommutativeCancellation(),         # Cancel commuting pairs
])
```

---

## 📊 Benchmark Circuits

- **GHZ State**: `H ⊗ CNOT chain` — tests linear routing overhead
- **QFT**: Gate-dense with controlled-phase — tests 2Q gate optimization
- **Grover's Search**: Multi-controlled gates — tests decomposition
- **QAOA Portfolio**: Variational finance circuit — tests real application

---

## 🔧 Tech Stack

- **Qiskit 2.x** — Circuit construction, transpilation, pass management
- **Streamlit** — Interactive web dashboard
- **Plotly** — Interactive charts and graphs
- **Matplotlib** — Circuit diagrams
- **Pandas** — Metrics tables

---

## 📝 Report Outline (Suggested)

1. Introduction — NISQ hardware and the need for optimization
2. Background — Qiskit transpiler stages and passes
3. System Design — Architecture of this tool
4. Experiments — Results across 4 circuits × 4 backends × 4 levels
5. Custom Pass — Implementation and evaluation
6. QAOA Finance Demo — Application context
7. Conclusion — Findings and limitations
