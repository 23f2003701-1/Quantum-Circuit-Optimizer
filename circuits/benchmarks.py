"""
Benchmark quantum circuits for optimization analysis.
Each function returns a named QuantumCircuit ready for transpilation.
"""

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import QFT
import numpy as np


def make_ghz(n_qubits: int = 4) -> QuantumCircuit:
    """
    GHZ (Greenberger–Horne–Zeilinger) state circuit.
    Creates maximal entanglement across all qubits.
    Abstract depth = n_qubits (one H + chain of CNOTs).
    """
    qc = QuantumCircuit(n_qubits, name="GHZ")
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def make_qft(n_qubits: int = 4) -> QuantumCircuit:
    """
    Quantum Fourier Transform circuit.
    Gate-intensive benchmark with many controlled-phase gates.
    """
    qc = QFT(n_qubits, name="QFT")
    return qc.decompose()


def make_grover(n_qubits: int = 4) -> QuantumCircuit:
    """
    Grover's algorithm oracle + diffuser (1 iteration).
    Tests how the optimizer handles multi-controlled gates.
    """
    qc = QuantumCircuit(n_qubits, name="Grover")

    # Oracle: mark the |1111> state
    qc.h(range(n_qubits))
    qc.barrier()

    # Multi-controlled Z as oracle (flip target state)
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.barrier()

    # Diffuser
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))

    return qc


def make_qaoa(n_qubits: int = 4, p_layers: int = 1) -> QuantumCircuit:
    """
    QAOA (Quantum Approximate Optimization Algorithm) circuit.
    Used for combinatorial optimization — here modelling a small
    portfolio optimization / MaxCut instance.

    The cost Hamiltonian represents a 4-asset correlation graph:
    minimize risk = sum of ZZ interactions (correlated assets).
    """
    qc = QuantumCircuit(n_qubits, name="QAOA-Portfolio")

    # Fixed angles (pre-optimized for demo; in practice these are variational)
    gamma = np.pi / 4   # cost layer angle
    beta = np.pi / 8    # mixer layer angle

    # Initial superposition
    qc.h(range(n_qubits))

    for _ in range(p_layers):
        # Cost layer: ZZ interactions between connected assets
        # Edges represent correlated asset pairs
        edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
        qc.barrier()
        for u, v in edges:
            qc.cx(u, v)
            qc.rz(2 * gamma, v)
            qc.cx(u, v)

        # Mixer layer: X rotations
        qc.barrier()
        for q in range(n_qubits):
            qc.rx(2 * beta, q)

    return qc


def make_random_circuit(n_qubits: int = 4, depth: int = 10, seed: int = 42) -> QuantumCircuit:
    """
    A random circuit with mixed gate types.
    Good for stress-testing the optimizer on arbitrary circuits.
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits, name="Random")

    single_gates = ["h", "x", "y", "z", "s", "t", "sdg", "tdg"]

    for _ in range(depth):
        gate_type = rng.choice(["single", "cx", "rz"])
        q = int(rng.integers(0, n_qubits))

        if gate_type == "single":
            gate = rng.choice(single_gates)
            getattr(qc, gate)(q)
        elif gate_type == "cx":
            q2 = int(rng.integers(0, n_qubits))
            if q2 != q:
                qc.cx(q, q2)
        elif gate_type == "rz":
            angle = float(rng.uniform(0, 2 * np.pi))
            qc.rz(angle, q)

    return qc


# Registry of all available benchmark circuits
BENCHMARKS = {
    "GHZ State": {
        "fn": make_ghz,
        "description": "Maximal entanglement circuit. Tests CNOT chain optimization.",
        "icon": "🔗",
        "default_qubits": 4,
    },
    "QFT": {
        "fn": make_qft,
        "description": "Quantum Fourier Transform. Gate-dense with controlled-phase gates.",
        "icon": "🌀",
        "default_qubits": 4,
    },
    "Grover's Search": {
        "fn": make_grover,
        "description": "1-iteration Grover oracle + diffuser. Tests multi-controlled gate decomposition.",
        "icon": "🔍",
        "default_qubits": 4,
    },
    "QAOA Portfolio": {
        "fn": make_qaoa,
        "description": "QAOA circuit for a 4-asset portfolio optimization problem (finance demo).",
        "icon": "📈",
        "default_qubits": 4,
    },
}
