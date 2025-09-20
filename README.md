# Field Dynamic Systems (FDS)
*A Python OOP framework for building, evolving, and studying dynamic systems with **weighted states** (“fields”).*  
This README includes short **definitions** and a **simple, practical flow** for defining classes so you can build a complete FDS stack from scratch.

---

## 1) What is this?
FDS treats a system as:  
- a **State Space** *(the where)*,  
- one or more **Fields** *(weights or potentials on states)*, and  
- **Operators** *(the how)* that transform distributions using those fields,  
all orchestrated by an **Evolution Engine** that advances the system in discrete steps (path-integral / sum-over-histories style). You can plug in different spaces (ℤ, ℤ², lattices), field types (real/complex), and operators (deterministic expectation, stochastic sampling) to model biased random walks, sampling processes (“dice”), coupled systems, and symmetry/mappings between systems.

---

## 2) Mini-Glossary (short, useful definitions)

- **State**  
  A single configuration point (e.g., integer `x` on ℤ, coordinate `(i, j)` on a grid). Prefer immutable dataclasses.

- **State Space**  
  The set of all states **plus** topology: neighbors, metric/distance, membership. Example: 1D line with step size 1.

- **Distribution**  
  A weight/probability assignment over states (pmf). Provides `normalize()`, `sample()`, and simple stats (mean, var) where applicable.

- **FieldValue**  
  The atomic value type held at each state by a field (e.g., real number, complex number, vector). Keep arithmetic minimal and consistent.

- **Field**  
  A (possibly time-varying) mapping `state → FieldValue`.  
  - *Global field*: background influence accessible everywhere.  
  - *Local/temporary field*: contextual or operator-specific weights.

- **Kernel (Transition Weights)**  
  A function that turns local info/fields into neighbor transition weights (e.g., biasing a random walk).

- **Operator**  
  A rule that transforms a **Distribution** using **Field/Kernel**. Two staples:
  - `ExpectationOperator`: deterministic update using weighted averages.
  - `SamplingOperator`: stochastic step based on transition probabilities.

- **Evolution Engine**  
  Multi-step coordinator that applies operators, tracks reachable/reaching sets and **layers** (e.g., by Chebyshev distance), and logs history.

- **Mapping**  
  A relation between spaces; supports push/pull of **Fields** and **Distributions** across spaces (for symmetry or analogy).

- **Aggregator**  
  Combines multiple fields/signals (e.g., sum, max, composition) with conflict/priority rules.

- **Ensemble**  
  A coupled collection of systems that publish/subscribe fields or otherwise influence one another.

- **Metrics/Stability**  
  Distances/divergences over distributions and simple Lyapunov-style indicators for probabilistic evolution.

---
---

## Quick Install
```bash
git clone https://github.com/ckCrimson/Field_Dynamic_System
.git
cd Field_Dynamic_System
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt  # if present
# or for editable dev
pip install -e .
