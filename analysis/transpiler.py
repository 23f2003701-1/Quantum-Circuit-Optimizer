"""
Transpilation engine.
Takes a circuit + backend, runs optimization levels 0–3, and returns all metrics and transpiled circuits.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import GenericBackendV2


# ── Backend definitions ─────────────────────────────────────────────────────

BACKENDS = {
    "5-qubit (star topology)": {
        "n_qubits": 5,
        "description": "Star: qubit 0 connected to all others. Very restrictive routing.",
        "coupling": [[0, 1], [1, 0], [0, 2], [2, 0], [0, 3], [3, 0], [0, 4], [4, 0]],
    },
    "5-qubit (linear chain)": {
        "n_qubits": 5,
        "description": "Linear: 0–1–2–3–4. Long SWAP chains needed for distant qubits.",
        "coupling": [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]],
    },
    "7-qubit (heavy-hex)": {
        "n_qubits": 7,
        "description": "IBM heavy-hex inspired. Realistic and commonly used.",
        "coupling": [
            [0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2],
            [3, 4], [4, 3], [4, 5], [5, 4], [5, 6], [6, 5],
            [1, 4], [4, 1],
        ],
    },
    "5-qubit (fully connected)": {
        "n_qubits": 5,
        "description": "All-to-all connectivity. No SWAP overhead — best case scenario.",
        "coupling": [
            [i, j] for i in range(5) for j in range(5) if i != j
        ],
    },
}


def get_backend(backend_name: str) -> GenericBackendV2:
    """Return a fake backend matching the chosen topology."""
    info = BACKENDS[backend_name]
    from qiskit.transpiler import CouplingMap
    coupling = CouplingMap(info["coupling"])
    backend = GenericBackendV2(
        num_qubits=info["n_qubits"],
        coupling_map=coupling,
    )
    return backend


# ── Metric extraction ────────────────────────────────────────────────────────

@dataclass
class CircuitMetrics:
    level: int
    depth: int
    total_gates: int
    cx_count: int
    swap_count: int
    single_qubit_gates: int
    weighted_cost: float
    gate_breakdown: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        if self.level == -1:
            return "Original"
        return f"Level {self.level}"


def extract_metrics(qc: QuantumCircuit, level: int = -1) -> CircuitMetrics:
    """Extract all relevant metrics from a (transpiled) circuit."""
    ops = qc.count_ops()

    cx_count = ops.get("cx", 0) + ops.get("ecr", 0) + ops.get("cz", 0)
    swap_count = ops.get("swap", 0)
    total_gates = sum(ops.values())
    single_q = total_gates - cx_count - swap_count

    # Weighted cost: depth matters most, then 2Q gates, then SWAPs
    # w_depth=1, w_cx=3, w_swap=5  (SWAPs are especially costly on NISQ)
    weighted_cost = qc.depth() * 1 + cx_count * 3 + swap_count * 5

    return CircuitMetrics(
        level=level,
        depth=qc.depth(),
        total_gates=total_gates,
        cx_count=cx_count,
        swap_count=swap_count,
        single_qubit_gates=single_q,
        weighted_cost=weighted_cost,
        gate_breakdown=dict(ops),
    )


# ── Transpilation runner ─────────────────────────────────────────────────────

@dataclass
class TranspilationResult:
    original_circuit: QuantumCircuit
    original_metrics: CircuitMetrics
    transpiled: dict          # level -> QuantumCircuit
    metrics: dict             # level -> CircuitMetrics
    backend_name: str
    circuit_name: str
    seed: int = 42


def run_transpilation(
    circuit: QuantumCircuit,
    backend_name: str,
    seed: int = 42,
) -> TranspilationResult:
    """
    Transpile the given circuit at optimization levels 0–3 on the chosen backend.
    Returns a TranspilationResult with all circuits and metrics.
    """
    backend = get_backend(backend_name)

    original_metrics = extract_metrics(circuit, level=-1)
    transpiled = {}
    metrics = {}

    for level in range(4):
        pm = generate_preset_pass_manager(
            optimization_level=level,
            backend=backend,
            seed_transpiler=seed,
        )
        t_qc = pm.run(circuit)
        transpiled[level] = t_qc
        metrics[level] = extract_metrics(t_qc, level=level)

    return TranspilationResult(
        original_circuit=circuit,
        original_metrics=original_metrics,
        transpiled=transpiled,
        metrics=metrics,
        backend_name=backend_name,
        circuit_name=circuit.name,
        seed=seed,
    )


# ── Stochastic analysis ──────────────────────────────────────────────────────

def run_stochastic_analysis(
    circuit: QuantumCircuit,
    backend_name: str,
    n_seeds: int = 20,
    level: int = 3,
) -> dict:
    """
    Run the same circuit at a fixed optimization level with different random seeds.
    Returns distribution of depth, cx_count, swap_count, weighted_cost.

    This demonstrates that routing is stochastic — different seeds produce
    different SWAP overhead even for the same circuit.
    """
    backend = get_backend(backend_name)
    results = {
        "depths": [],
        "cx_counts": [],
        "swap_counts": [],
        "weighted_costs": [],
        "seeds": list(range(n_seeds)),
    }

    for seed in range(n_seeds):
        pm = generate_preset_pass_manager(
            optimization_level=level,
            backend=backend,
            seed_transpiler=seed,
        )
        t_qc = pm.run(circuit)
        m = extract_metrics(t_qc, level=level)
        results["depths"].append(m.depth)
        results["cx_counts"].append(m.cx_count)
        results["swap_counts"].append(m.swap_count)
        results["weighted_costs"].append(m.weighted_cost)

    return results
