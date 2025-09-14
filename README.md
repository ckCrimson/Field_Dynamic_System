# Field Dynamic Systems (FDS)
*A Python OOP framework for building, evolving, and studying dynamic systems with **weighted states** (“fields”).*

---

## Overview
**FDS** models systems as **State Spaces + Fields + Operators**, then evolves them with an **Evolution Engine** (path-integral / sum-over-histories style). It’s designed for research and experimentation where transition **weights** (not just fixed matrices) control dynamics. The framework includes patterns and examples spanning biased random walks, sampling/“dice” processes, symmetry/mappings between systems, and stability tooling.

For the full theory and notation, see the companion paper:
- `paper/On_the_formulation_and_application_of_dynamic_systems_with_weighted_states.pdf`

---

## Features
- **Clean OOP design**: `State`, `StateSpace`, `Field` (real/complex), `Operator`, `EvolutionEngine`, `Ensemble`, `Mapping`, etc.
- **Path-integral recurrence**: multi-step distributions over reachable sets/layers.
- **Deterministic & stochastic evolution**: expectation operators and samplers.
- **Composability**: map fields between spaces, couple multiple systems, aggregate global fields.
- **Stability & metrics**: hooks for Lyapunov-style analysis on probabilistic evolution.
- **Research-ready**: minimal, hackable, and easy to extend.

---

## Install
```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>

# (recommended) create a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

# install dependencies (if provided)
pip install -U pip
pip install -r requirements.txt

# or develop locally
pip install -e .
