"""Compare Qiskit and PennyLane on the three-qubit ladder ansatz."""

import csv
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector

sys.path.insert(0, str(Path(__file__).parents[1] / "01-quantum-circuit-simulator"))
from simulator import CNOT, I, basis_state, simulate  # noqa: E402

ANGLES = np.linspace(0, 2 * np.pi, 201)
SHOT_ANGLES = np.linspace(0, 2 * np.pi, 60)
VARIANT_ANGLES = np.linspace(0, 2 * np.pi, 180, endpoint=False)
SHOTS = 2**15
THETA = Parameter("θ")
QISKIT_CIRCUIT = QuantumCircuit(3)
for _ in range(2):
    for qubit in range(3):
        QISKIT_CIRCUIT.ry(THETA, qubit)
    QISKIT_CIRCUIT.cx(0, 1)
    QISKIT_CIRCUIT.cx(1, 2)


def ladder(theta, num_qubits, layers, rotation):
    for _ in range(layers):
        for qubit in range(num_qubits):
            rotation(theta, wires=qubit)
        for qubit in range(num_qubits - 1):
            qml.CNOT(wires=[qubit, qubit + 1])


def make_pennylane_state(num_qubits=3, layers=2, rotation=qml.RY):
    device = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(device)
    def circuit(theta):
        ladder(theta, num_qubits, layers, rotation)
        return qml.state()

    return circuit


pennylane_state = make_pennylane_state()
SHOT_DEVICE = qml.device("default.qubit", wires=3, seed=7)


@qml.set_shots(SHOTS)
@qml.qnode(SHOT_DEVICE)
def pennylane_samples(theta):
    ladder(theta, 3, 2, qml.RY)
    return qml.sample(wires=range(3))


def own_state(theta: float) -> np.ndarray:
    cosine, sine = np.cos(theta / 2), np.sin(theta / 2)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    rotation_layer = np.kron(np.kron(rotation, rotation), rotation)
    gates = [rotation_layer, np.kron(CNOT, I), np.kron(I, CNOT)] * 2
    return simulate(basis_state("000"), gates)


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


def compare_own() -> np.ndarray:
    qiskit_states = np.array([qiskit_state(theta) for theta in ANGLES])
    own_states = np.array([own_state(theta) for theta in ANGLES])
    assert np.allclose(own_states, qiskit_states, atol=1e-10)
    return np.array([observables(state) for state in own_states])


def benchmark() -> list[dict]:
    exact_times = {}
    for name, state_function in (
        ("Qiskit", qiskit_state),
        ("PennyLane", pennylane_state),
    ):
        state_function(ANGLES[0])
        durations = []
        for _ in range(5):
            start = perf_counter()
            [state_function(theta) for theta in ANGLES]
            durations.append(perf_counter() - start)
        exact_times[name] = float(np.median(durations))

    qiskit_probabilities = []
    state = Statevector.from_instruction(QISKIT_CIRCUIT.assign_parameters({THETA: 0}))
    state.seed(7)
    state.sample_counts(SHOTS)
    start = perf_counter()
    for index, theta in enumerate(SHOT_ANGLES):
        state = Statevector.from_instruction(
            QISKIT_CIRCUIT.assign_parameters({THETA: theta})
        )
        state.seed(7 + index)
        qiskit_probabilities.append(state.sample_counts(SHOTS).get("000", 0) / SHOTS)
    qiskit_shot_time = perf_counter() - start

    pennylane_samples(SHOT_ANGLES[0])
    start = perf_counter()
    pennylane_probabilities = [
        np.mean(np.all(pennylane_samples(theta) == 0, axis=1))
        for theta in SHOT_ANGLES
    ]
    pennylane_shot_time = perf_counter() - start

    exact_probabilities = np.array(
        [observables(qiskit_state(theta))[0] for theta in SHOT_ANGLES]
    )
    errors = {
        "Qiskit": float(
            np.max(np.abs(np.array(qiskit_probabilities) - exact_probabilities))
        ),
        "PennyLane": float(
            np.max(np.abs(np.array(pennylane_probabilities) - exact_probabilities))
        ),
    }
    assert max(errors.values()) < 0.02

    return [
        {
            "mode": "exact",
            "framework": name,
            "angles": len(ANGLES),
            "shots": "",
            "repeats": 5,
            "seconds": exact_times[name],
            "max_p000_error": 0.0,
        }
        for name in exact_times
    ] + [
        {
            "mode": "shots",
            "framework": name,
            "angles": len(SHOT_ANGLES),
            "shots": SHOTS,
            "repeats": 1,
            "seconds": seconds,
            "max_p000_error": errors[name],
        }
        for name, seconds in (
            ("Qiskit", qiskit_shot_time),
            ("PennyLane", pennylane_shot_time),
        )
    ]


def architecture_data() -> tuple[dict, dict]:
    qubits_layers = {}
    for num_qubits in range(2, 5):
        for layers in range(1, 4):
            circuit = make_pennylane_state(num_qubits, layers)
            qubits_layers[num_qubits, layers] = np.array(
                [observables(circuit(theta))[1] for theta in VARIANT_ANGLES]
            )

    rotations = {}
    for name, gate in (("RX", qml.RX), ("RY", qml.RY), ("RZ", qml.RZ)):
        circuit = make_pennylane_state(rotation=gate)
        rotations[name] = np.array(
            [observables(circuit(theta)) for theta in VARIANT_ANGLES]
        )

    assert all(np.all(np.abs(values) <= 1 + 1e-10) for values in qubits_layers.values())
    assert all(np.all(np.abs(values) <= 1 + 1e-10) for values in rotations.values())
    assert np.allclose(rotations["RZ"], 1)
    return qubits_layers, rotations


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


def plot_architectures(qubits_layers: dict):
    figure, axes = plt.subplots(1, 3, sharey=True, layout="constrained")
    for num_qubits, axis in zip(range(2, 5), axes):
        for layers in range(1, 4):
            axis.plot(
                VARIANT_ANGLES,
                qubits_layers[num_qubits, layers],
                label=f"{layers} layer{'s' if layers > 1 else ''}",
            )
        axis.set(title=f"{num_qubits} qubits", xlabel=r"$\theta$")
    axes[0].set_ylabel(r"$\langle Z^{\otimes n} \rangle$")
    axes[-1].legend()
    return figure


def plot_rotations(rotations: dict):
    figure, axes = plt.subplots(2, sharex=True, layout="constrained")
    for name, values in rotations.items():
        axes[0].plot(VARIANT_ANGLES, values[:, 0], label=name)
        axes[1].plot(VARIANT_ANGLES, values[:, 1], label=name)
    axes[0].set(ylabel="P(000)", ylim=(-0.05, 1.05))
    axes[1].set(
        xlabel=r"$\theta$",
        ylabel=r"$\langle Z_1Z_2Z_3 \rangle$",
        ylim=(-1.05, 1.05),
    )
    axes[0].legend()
    return figure


def main() -> None:
    qiskit_values, pennylane_values = compare()
    own_values = compare_own()
    assert np.allclose(own_values, qiskit_values, atol=1e-10)

    directory = Path(__file__).parent
    figure = plot(qiskit_values, pennylane_values)
    figure.savefig(directory / "ladder_ansatz_comparison.png", dpi=150)
    plt.close(figure)

    rows = benchmark()
    with (directory / "benchmark.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    qubits_layers, rotations = architecture_data()
    figure = plot_architectures(qubits_layers)
    figure.savefig(directory / "qubits_layers.png", dpi=150)
    plt.close(figure)
    figure = plot_rotations(rotations)
    figure.savefig(directory / "rotation_gates.png", dpi=150)
    plt.close(figure)
    print("Qiskit, PennyLane, and the NumPy simulator agree; wrote all optional results.")


if __name__ == "__main__":
    main()
