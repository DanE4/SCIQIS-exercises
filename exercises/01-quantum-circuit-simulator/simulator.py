"""A small state-vector quantum circuit simulator built with NumPy."""

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


StateVector = NDArray[np.complex128]

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
T = np.diag([1, np.exp(1j * np.pi / 4)]).astype(complex)

CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=complex,
)
CZ = np.diag([1, 1, 1, -1]).astype(complex)


def _state_vector(state: ArrayLike) -> StateVector:
    vector = np.asarray(state, dtype=complex)
    if vector.ndim != 1 or vector.size == 0 or vector.size & (vector.size - 1):
        raise ValueError("state must be a one-dimensional vector with 2**n entries")
    if not np.isclose(np.linalg.norm(vector), 1.0):
        raise ValueError("state must be normalized")
    return vector.copy()


def basis_state(bits: str) -> StateVector:
    """Return an MSB-first computational basis state such as |011>."""
    if not bits or set(bits) - {"0", "1"}:
        raise ValueError("bits must be a non-empty binary string")

    state = np.zeros(2 ** len(bits), dtype=complex)
    state[int(bits, 2)] = 1
    return state


def simulate(initial_state: ArrayLike, gates: Iterable[ArrayLike]) -> StateVector:
    """Apply full-system gate matrices in chronological order."""
    state = _state_vector(initial_state)
    expected_shape = (state.size, state.size)

    for gate in gates:
        matrix = np.asarray(gate, dtype=complex)
        if matrix.shape != expected_shape:
            raise ValueError(f"each gate must have shape {expected_shape}")
        state = matrix @ state

    if not np.isclose(np.linalg.norm(state), 1.0):
        raise ValueError("gates must preserve state normalization")
    return state


def measurement_probabilities(state: ArrayLike) -> NDArray[np.float64]:
    """Return computational-basis measurement probabilities."""
    probabilities = np.abs(_state_vector(state)) ** 2
    return probabilities / probabilities.sum()


def sample_measurements(
    state: ArrayLike, shots: int = 1, seed: int | None = None
) -> NDArray[np.int64]:
    """Sample computational-basis indices from a state vector."""
    if not isinstance(shots, int) or shots < 1:
        raise ValueError("shots must be a positive integer")

    probabilities = measurement_probabilities(state)
    return np.random.default_rng(seed).choice(
        probabilities.size, size=shots, p=probabilities
    )


def qft_matrix(num_qubits: int) -> NDArray[np.complex128]:
    """Return the quantum Fourier transform matrix for ``num_qubits``."""
    if not isinstance(num_qubits, int) or num_qubits < 1:
        raise ValueError("num_qubits must be a positive integer")

    size = 2**num_qubits
    indices = np.arange(size)
    omega = np.exp(2j * np.pi / size)
    return omega ** np.outer(indices, indices) / np.sqrt(size)


def demo() -> None:
    """Run a compact self-check for the simulator."""
    zero = basis_state("0")
    one = basis_state("1")
    plus = (zero + one) / np.sqrt(2)
    t_state = (zero + np.exp(1j * np.pi / 4) * one) / np.sqrt(2)

    assert np.allclose(simulate(zero, [X]), one)
    assert np.allclose(simulate(zero, [H]), plus)
    assert np.allclose(simulate(plus, [T]), t_state)

    bell = simulate(basis_state("00"), [np.kron(H, I), CNOT])
    assert np.allclose(bell, (basis_state("00") + basis_state("11")) / np.sqrt(2))
    assert np.allclose(measurement_probabilities(bell), [0.5, 0, 0, 0.5])

    qft = qft_matrix(3)
    omega = np.exp(2j * np.pi / 8)
    assert np.allclose(simulate(basis_state("000"), [qft]), np.ones(8) / np.sqrt(8))
    assert np.allclose(
        simulate(basis_state("011"), [qft]),
        omega ** (3 * np.arange(8)) / np.sqrt(8),
    )

    superposition = (basis_state("000") + basis_state("011")) / np.sqrt(2)
    output = simulate(superposition, [qft])
    assert np.allclose(output, (1 + omega ** (3 * np.arange(8))) / 4)
    probabilities = measurement_probabilities(output)
    assert np.isclose(probabilities[0], 0.25)
    assert np.isclose(probabilities[4], 0.0)

    samples = sample_measurements(bell, shots=32, seed=7)
    assert samples.shape == (32,) and set(samples) <= {0, 3}
    print("All quantum circuit simulator checks passed.")


if __name__ == "__main__":
    demo()
