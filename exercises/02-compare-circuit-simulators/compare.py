"""Compare Qiskit and PennyLane on the three-qubit ladder ansatz."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector


ANGLES = np.linspace(0, 2 * np.pi, 201)
THETA = Parameter("θ")
QISKIT_CIRCUIT = QuantumCircuit(3)
for _ in range(2):
    for qubit in range(3):
        QISKIT_CIRCUIT.ry(THETA, qubit)
    QISKIT_CIRCUIT.cx(0, 1)
    QISKIT_CIRCUIT.cx(1, 2)

DEVICE = qml.device("default.qubit", wires=3)


@qml.qnode(DEVICE)
def pennylane_state(theta: float):
    for _ in range(2):
        for qubit in range(3):
            qml.RY(theta, wires=qubit)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
    return qml.state()


def qiskit_state(theta: float) -> np.ndarray:
    state = Statevector.from_instruction(
        QISKIT_CIRCUIT.assign_parameters({THETA: theta})
    ).data
    # Qiskit indexes |q2 q1 q0>; PennyLane and our course use |q0 q1 q2>.
    return state.reshape(2, 2, 2).transpose(2, 1, 0).reshape(-1)


def observables(state: np.ndarray) -> tuple[float, float]:
    probabilities = np.abs(state) ** 2
    zzz = sum(
        (-1) ** index.bit_count() * probability
        for index, probability in enumerate(probabilities)
    )
    return float(probabilities[0]), float(zzz)


def compare() -> tuple[np.ndarray, np.ndarray]:
    qiskit_states = np.array([qiskit_state(theta) for theta in ANGLES])
    pennylane_states = np.array([pennylane_state(theta) for theta in ANGLES])
    qiskit_values = np.array([observables(state) for state in qiskit_states])
    pennylane_values = np.array([observables(state) for state in pennylane_states])

    assert np.allclose(np.linalg.norm(qiskit_states, axis=1), 1)
    assert np.allclose(np.linalg.norm(pennylane_states, axis=1), 1)
    assert np.allclose(qiskit_states, pennylane_states, atol=1e-10)
    assert np.allclose(qiskit_values, pennylane_values, atol=1e-10)
    assert np.allclose(qiskit_values[0], [1, 1])
    assert np.all((0 <= qiskit_values[:, 0]) & (qiskit_values[:, 0] <= 1))
    assert np.all(np.abs(qiskit_values[:, 1]) <= 1 + 1e-10)
    return qiskit_values, pennylane_values


def plot(qiskit_values: np.ndarray, pennylane_values: np.ndarray):
    figure, axes = plt.subplots(2, sharex=True, layout="constrained")
    labels = ("P(000)", r"$\langle Z_1Z_2Z_3 \rangle$")
    for column, axis in enumerate(axes):
        axis.plot(ANGLES, qiskit_values[:, column], label="Qiskit")
        axis.plot(ANGLES, pennylane_values[:, column], "--", label="PennyLane")
        axis.set(ylabel=labels[column], ylim=(-1.05 if column else -0.05, 1.05))
    axes[0].legend()
    axes[1].set_xlabel(r"$\theta$")
    return figure


def main() -> None:
    qiskit_values, pennylane_values = compare()
    output = Path(__file__).with_name("ladder_ansatz_comparison.png")
    plot(qiskit_values, pennylane_values).savefig(output, dpi=150)
    print(f"Qiskit and PennyLane agree across {len(ANGLES)} angles; wrote {output}")


if __name__ == "__main__":
    main()
