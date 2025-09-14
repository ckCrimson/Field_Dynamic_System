# Field Dynamic Systems (FDS)
*A Python OOP framework for building, evolving, and studying dynamic systems with **weighted states** (“fields”).*  
> This README includes short **definitions** and a **simple, practical flow** for defining classes so you can build a complete FDS stack from scratch.

---

## What is this, in one paragraph?
FDS treats a system as a **State Space** (the where), a **Field** (the weights on states), and **Operators** (the how), all orchestrated by an **Evolution Engine** that advances distributions over time. You can plug in different spaces (ℤ, ℤ², lattices), field types (real/complex), and operators (deterministic expectation, stochastic sampling) to model biased random walks, sampling (“dice”) processes, coupled systems, and symmetry/mappings between systems.

---

## Mini-Glossary (short, useful definitions)

- **State**: A single configuration point (e.g., integer `x` on ℤ, coordinate `(i,j)` on a grid).
- **State Space**: The set of all states **plus** topology (neighbors/metric). Example: 1D line with step size 1.
- **Distribution**: A weight/probability assignment over states (pmf). Supports normalize, mean, var, etc.
- **FieldValue**: The atomic value held at each state by a field (real, complex, vector, etc.).
- **Field**: A (possibly time-varying) mapping `state → FieldValue`.  
  - *Global field*: background influence accessible everywhere.  
  - *Local/temporary field*: contextual or operator-specific weights.
- **Kernel (Transition Weights)**: Function turning local info/fields into transition weights to neighbors.
- **Operator**: Rule that transforms a **Distribution** using **Field/Kernel** (e.g., expectation, sampling).
- **Evolution Engine**: Multi-step coordinator that applies operators, tracks reachable sets/layers, logs history.
- **Mapping**: Relation between spaces; supports push/pull of **Fields** and **Distributions** across spaces.
- **Aggregator**: Combines many fields/signals (sum, max, composition) with conflict/priority rules.
- **Ensemble**: A coupled collection of systems that publish/subscribe fields or otherwise influence each other.
- **Metrics/Stability**: Distances/divergences over distributions and Lyapunov-style indicators.

---

## Build Flow (define classes in this order)
> Follow this sequence to avoid circular deps and keep interfaces clean. Each step shows the **goal** and a **minimal stub**.

### 1) Foundations (value types & ids)
**Goal:** shared types/utilities without domain logic.
```python
# fds/core/types.py
from typing import Protocol, Any

class FieldValue(Protocol):
    def __add__(self, other: Any) -> "FieldValue": ...
    def __mul__(self, scalar: float) -> "FieldValue": ...
    @staticmethod
    def zero() -> "FieldValue": ...
