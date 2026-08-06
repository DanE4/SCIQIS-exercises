# SCIQIS exercises

Exercises for the Scientific Computing in Quantum Information Science course.

## Setup

Install [uv](https://docs.astral.sh/uv/), then create the project environment:

```sh
uv sync
```

Run an exercise's Python self-check:

```sh
uv run python exercises/01-quantum-circuit-simulator/simulator.py
```

Or explore the notebooks:

```sh
uv run jupyter lab
```

Each exercise lives in its own numbered folder under `exercises/` and may
contain Python files, Jupyter notebooks, or both.

## Exercises

1. [Quantum circuit simulator](exercises/01-quantum-circuit-simulator/)
2. [Compare circuit simulators](exercises/02-compare-circuit-simulators/)
