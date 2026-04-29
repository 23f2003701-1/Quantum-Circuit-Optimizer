"""
Explainability layer for quantum circuit optimization.

This module analyses what transformations the transpiler applied, giving human-readable explanations of *why* and *how* the circuit changed.
It also implements a simple custom cancellation pass to demonstrate custom transpiler pass creation.
"""

from dataclasses import dataclass
from typing import Optional
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    InverseCancellation,      # Qiskit 2.x: replaces CXCancellation
    RemoveIdentityEquivalent, # Qiskit 2.x: removes near-identity gates
    Optimize1qGates,
    CommutativeCancellation,
    RemoveBarriers,
)


# ── Transformation record ────────────────────────────────────────────────────

@dataclass
class TransformRecord:
    """Represents a single observed transformation between two circuit versions."""
    stage: str
    before_depth: int
    after_depth: int
    before_cx: int
    after_cx: int
    before_total: int
    after_total: int
    description: str
    category: str  # 'gate_cancel' | 'routing' | 'merge' | 'decompose' | 'layout'


# ── Custom passes ────────────────────────────────────────────────────────────

def build_custom_cancellation_pass() -> PassManager:
    """
    A custom PassManager that chains three simple optimizations:
      1. Remove barriers (so gates across barriers can interact)
      2. Cancel adjacent CX gates (CX · CX = I) via InverseCancellation
      3. Merge consecutive single-qubit gates into one
      4. Cancel commuting gates
    """
    pm = PassManager()
    pm.append(RemoveBarriers())
    pm.append(InverseCancellation([CXGate()]))   # CX · CX = Identity
    pm.append(Optimize1qGates())
    pm.append(CommutativeCancellation())
    return pm


def apply_custom_pass(circuit: QuantumCircuit) -> tuple[QuantumCircuit, list[TransformRecord]]:
    """
    Apply the custom cancellation pass step-by-step, recording what changed.
    Returns the optimized circuit and a log of all transformations.
    """
    log = []

    def snap(qc):
        ops = qc.count_ops()
        return {
            "depth": qc.depth(),
            "cx": ops.get("cx", 0),
            "total": sum(ops.values()),
        }

    current = circuit.copy()
    s0 = snap(current)

    # Step 1: Remove barriers
    pm1 = PassManager([RemoveBarriers()])
    current = pm1.run(current)
    s1 = snap(current)
    if s1 != s0:
        log.append(TransformRecord(
            stage="Remove Barriers",
            before_depth=s0["depth"], after_depth=s1["depth"],
            before_cx=s0["cx"], after_cx=s1["cx"],
            before_total=s0["total"], after_total=s1["total"],
            description="Barriers are programmer hints, not real gates. Removing them "
                        "allows the optimizer to see across them and cancel adjacent gates.",
            category="merge",
        ))

    # Step 2: CX cancellation
    pm2 = PassManager([InverseCancellation([CXGate()])])
    before = snap(current)
    current = pm2.run(current)
    after = snap(current)
    if after != before:
        canceled = before["cx"] - after["cx"]
        log.append(TransformRecord(
            stage="CX Cancellation",
            before_depth=before["depth"], after_depth=after["depth"],
            before_cx=before["cx"], after_cx=after["cx"],
            before_total=before["total"], after_total=after["total"],
            description=f"Found {canceled} adjacent CX gate pair(s) that cancel: "
                        f"CX · CX = Identity. These pairs are redundant and removed entirely.",
            category="gate_cancel",
        ))
    else:
        log.append(TransformRecord(
            stage="CX Cancellation",
            before_depth=before["depth"], after_depth=after["depth"],
            before_cx=before["cx"], after_cx=after["cx"],
            before_total=before["total"], after_total=after["total"],
            description="No adjacent CX pairs found to cancel in this circuit.",
            category="gate_cancel",
        ))

    # Step 3: Single-qubit gate merging
    pm3 = PassManager([Optimize1qGates()])
    before = snap(current)
    current = pm3.run(current)
    after = snap(current)
    merged = before["total"] - after["total"]
    log.append(TransformRecord(
        stage="1Q Gate Merging",
        before_depth=before["depth"], after_depth=after["depth"],
        before_cx=before["cx"], after_cx=after["cx"],
        before_total=before["total"], after_total=after["total"],
        description=f"Merged consecutive single-qubit gates on the same qubit "
                    f"into one gate. Reduced total gate count by {merged}. "
                    f"Any sequence of 1Q gates = one rotation.",
        category="merge",
    ))

    # Step 4: Commutative cancellation
    pm4 = PassManager([CommutativeCancellation()])
    before = snap(current)
    current = pm4.run(current)
    after = snap(current)
    if after["total"] < before["total"]:
        log.append(TransformRecord(
            stage="Commutative Cancellation",
            before_depth=before["depth"], after_depth=after["depth"],
            before_cx=before["cx"], after_cx=after["cx"],
            before_total=before["total"], after_total=after["total"],
            description="Some gates commute (can be reordered without changing the result). "
                        "After reordering, additional cancellation opportunities were found.",
            category="gate_cancel",
        ))
    else:
        log.append(TransformRecord(
            stage="Commutative Cancellation",
            before_depth=before["depth"], after_depth=after["depth"],
            before_cx=before["cx"], after_cx=after["cx"],
            before_total=before["total"], after_total=after["total"],
            description="No additional commutation-based cancellations found.",
            category="gate_cancel",
        ))

    return current, log


# ── Difference analyzer ──────────────────────────────────────────────────────

def build_diff_summary(
    original_metrics,
    opt_metrics_dict: dict,
    best_level: int = 3,
) -> list[str]:
    """
    Build a human-readable diff summary comparing original vs best optimized circuit.
    Returns a list of bullet-point strings.
    """
    orig = original_metrics
    opt = opt_metrics_dict[best_level]
    lines = []

    depth_delta = orig.depth - opt.depth
    cx_delta = orig.cx_count - opt.cx_count
    swap_added = opt.swap_count
    gate_delta = orig.total_gates - opt.total_gates
    cost_delta = orig.weighted_cost - opt.weighted_cost
    cost_pct = (cost_delta / orig.weighted_cost * 100) if orig.weighted_cost else 0

    if depth_delta > 0:
        lines.append(f"✅ Circuit depth reduced by **{depth_delta}** "
                     f"({orig.depth} → {opt.depth})")
    elif depth_delta < 0:
        lines.append(f"⚠️ Circuit depth *increased* by {abs(depth_delta)} "
                     f"due to SWAP routing overhead ({orig.depth} → {opt.depth})")
    else:
        lines.append(f"↔️ Circuit depth unchanged at {orig.depth}")

    if cx_delta > 0:
        lines.append(f"✅ **{cx_delta}** two-qubit (CX/CNOT) gate(s) eliminated")
    elif cx_delta < 0:
        lines.append(f"⚠️ **{abs(cx_delta)}** extra CX gates added (routing decomposition)")
    else:
        lines.append("↔️ CX gate count unchanged")

    if swap_added > 0:
        lines.append(f"🔀 **{swap_added}** SWAP gate(s) inserted for hardware routing. "
                     f"Each SWAP = 3 CX gates under the hood — routing overhead is real cost.")
    else:
        lines.append("✅ No SWAP gates inserted — good qubit layout found")

    if gate_delta > 0:
        lines.append(f"✅ **{gate_delta}** total gate(s) removed via cancellation/merging")
    elif gate_delta < 0:
        lines.append(f"⚠️ Total gate count grew by {abs(gate_delta)} "
                     f"(decomposition into native gate set)")

    lines.append(f"📊 Weighted cost: **{orig.weighted_cost:.0f} → {opt.weighted_cost:.0f}** "
                 f"({'−' if cost_pct > 0 else '+'}{abs(cost_pct):.1f}%)")

    return lines


def explain_optimization_levels() -> dict[int, dict]:
    """Return explanations for each optimization level (0–3)."""
    return {
        0: {
            "label": "Level 0 — No Optimization",
            "color": "#ef4444",
            "description": (
                "Only maps qubits to hardware and inserts the minimum SWAPs needed "
                "to satisfy connectivity. No gate cancellation or merging. "
                "Fastest to compile but worst circuit quality."
            ),
            "passes": ["Layout (trivial)", "Routing (basic SWAP)", "Gate translation"],
        },
        1: {
            "label": "Level 1 — Light Optimization",
            "color": "#f97316",
            "description": (
                "Adds single-qubit gate merging and some redundant-gate removal. "
                "Good for quick runs where compile time matters. "
                "Default level for most use cases."
            ),
            "passes": ["Layout", "Routing", "1Q gate merging", "Redundant gate removal"],
        },
        2: {
            "label": "Level 2 — Medium Optimization",
            "color": "#eab308",
            "description": (
                "Enables commutative cancellation, SWAP network optimization, "
                "and better layout heuristics. Noticeable improvement in CX count."
            ),
            "passes": ["Layout (noise-aware)", "Routing", "CX cancellation",
                       "Commutative cancellation", "1Q merging"],
        },
        3: {
            "label": "Level 3 — Maximum Optimization",
            "color": "#22c55e",
            "description": (
                "Full suite: noise-adaptive layout, advanced SWAP routing, "
                "Clifford synthesis, resynthesis of 2Q gate blocks, and all cancellations. "
                "Slowest to compile but best circuit quality for NISQ hardware."
            ),
            "passes": ["Noise-adaptive layout", "Sabre SWAP routing",
                       "CX + commutative cancellation", "1Q merging",
                       "2Q block resynthesis", "Clifford simplification"],
        },
    }
