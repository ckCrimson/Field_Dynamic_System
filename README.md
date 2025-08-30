# Field Dynamic Systems (FDS)

A modular Python framework for **Field Dynamic Systems (FDS)** — where **fields defined on state spaces** drive system evolution. FDS unifies stochastic/Markov-style updates and quantum-like path-interference into one toolkit: **kernels** define local transition rules, a **path-integral engine** aggregates over paths, and **ensembles** couple multiple systems via global fields. Comes with symmetry-aware lattices (e.g., 6-/8-fold), reachable/reaching utilities, and random-walker demos.

---

## Why FDS?
Traditional Markov chains specify transition probabilities explicitly. **FDS flips that**: we store *field values on states* and let **kernels** map fields → transitions. This makes it easy to:
- model **bias, potentials, or momentum** as fields,
- switch between **classical** (real PDFs) and **quantum-like** (complex amplitudes) updates,
- compose **multi-system ensembles** where systems influence each other through **global fields**.

---

## Core ideas (quick tour)
- **State spaces**: integer lattices (ℤ, ℤ²), triangular/square grids, or your own `StateSpace`.
- **Fields**: values attached to every state (real, complex, PDFs, custom types).
- **Kernels**: local rules `K(s | s0, field)`; includes basic Markovian, phase-only, and custom kernels.
- **Path integrals**: accumulate contributions over all paths to get the next state distribution.
- **Reachable/Reaching**: utilities to generate one-step neighborhoods and inverse neighborhoods.
- **Ensembles & global fields**: systems publish/consume global fields (e.g., `GlobalFieldAffectingEnsemble`).
- **Symmetry modules**: helpers for 6-fold/8-fold lattices and distance metrics (Chebyshev, Manhattan, Euclidean).

---

## Installation
```bash
# clone your private repo first, then:
pip install -e ".[dev]"   # or just: pip install -e .
