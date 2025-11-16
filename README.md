# Field Dynamic Systems (FDS) Framework

> A general framework for modeling dynamic systems using **weighted states (fields)**, **generalized chain-rule style evolution**, and **relationships between multiple systems** (affected systems and ensembles).

This repository contains a Python implementation of the **Field Dynamic Systems (FDS)** framework.

This README presents the **theoretical foundation only**, independent of implementation details.

---

## Table of Contents

- [1. Motivation](#1-motivation)
- [2. High-Level Overview](#2-high-level-overview)
- [3. Core FDS Concepts](#3-core-fds-concepts)
  - [3.1 State and State Space](#31-state-and-state-space)
  - [3.2 Fields and Field Values](#32-fields-and-field-values)
  - [3.3 Kernels, Operators, and Single-Step Evolution](#33-kernels-operators-and-single-step-evolution)
  - [3.4 Paths, Contributions, and Multi-Step Recurrence](#34-paths-contributions-and-multi-step-recurrence)
  - [3.5 Dynamic System](#35-dynamic-system)
- [4. Affected Systems Framework](#4-affected-systems-framework)
- [5. Ensemble Framework](#5-ensemble-framework)
- [6. Conceptual Diagram](#6-conceptual-diagram)
- [7. Where to Go Next](#7-where-to-go-next)

---

# 1. Motivation

Many systems in nature and engineering evolve over time:

- a particle moving on a lattice,
- a sample mean changing with more observations,
- cooling of a warm body,
- interacting subsystems under joint constraints.

Classical tools (differential equations, Markov chains) handle special cases, but often fail to:

- treat deterministic/probabilistic behavior in a unified way,
- handle state-dependent or time-dependent transitions cleanly,
- express correlations across multiple systems,
- encode field-based transformations and compositions.

**Field Dynamic Systems (FDS)** unify these concepts using:

- fields on states or transitions,
- transforms,
- compositions,
- generalized chain-rule based multi-step evolution,
- affected-system relationships,
- and ensembles.

---

# 2. High-Level Overview

An FDS consists of:

- a **state space** \( S \),
- one or more **fields** \( F \),
- **transforms** \( Z^p, Z^t, Z^c \),
- **compositions** \( \cdot \) (internal) and \( + \) (external),
- **operators** controlling evolution,
- a **multi-step recurrence** analogous to a path integral.

Two major extensions:

1. **Affected Systems Framework** — directional influence between systems  
2. **Ensemble Framework** — collections of systems for comparison & joint analysis

---

# 3. Core FDS Concepts

## 3.1 State and State Space

A **state** is a single configuration of a system.

Examples:

- a lattice point,
- a pair \( (x, \kappa) \),
- a histogram / count vector,
- temperature in a thermodynamic model.

The **state space** \( S \) determines:

- all valid states,
- allowable transitions,
- reachable and reaching sets.

**Reachable states** (from \( s_0 \) in \( l \) steps):  
states that can be reached via valid transitions.

**Reaching states** (for \( s' \)):  
states \( s \) such that a single-step transition \( s \to s' \) exists.

---

## 3.2 Fields and Field Values

A **field** assigns values to states or transitions:

$$
F : S \to \mathcal{V}
\qquad\text{or}\qquad
F : S \times S \to \mathcal{V}.
$$

Field values may represent:

- weights or preferences,
- amplitudes,
- potentials / kernels,
- contributions to multi-step evolution.

**Field Values**

- \( F(s) \) — value at state \( s \)
- \( F(s' : s) \) — value for transition \( s \to s' \)

---

## 3.3 Kernels, Operators, and Single-Step Evolution

A **kernel** is simply the transition field:

$$
K(s \to s') = F_t(s' : s).
$$

### Transforms

- **\( Z^p \)** — internal transform applied to previous-step fields  
- **\( Z^t \)** — external transform applied to single-step fields  
- **\( Z^c \)** — cleanup transform applied after accumulation  

### Compositions

- **Internal composition**: \( a \cdot b \)  
  (combines transformed previous and single-step contributions)

- **External composition**: \( a + b \)  
  (accumulates contributions from different reaching states)

No `*` is used — GitHub formatting-safe.

---

## 3.4 Paths, Contributions, and Multi-Step Recurrence

### Path

A **path** is:

$$
\pi = (s_0, s_1, \dots, s_l).
$$

### Path Contribution

Each transition contributes via:

- single-step field \( F_t \)
- transforms \( Z^p, Z^t \)
- internal composition \( \cdot \)

A path's total contribution is the structured combination of its transitions.

### Conceptual Path Integral

$$
F^{l}(s_l : s_0)
=
\sum_{\pi : s_0 \to s_l}
\text{Contribution}(\pi).
$$

### Why We Do Not Enumerate Paths

Path enumeration is exponential.

FDS instead uses a **generalized chain-rule recurrence** using reaching states.

---

## 3.4.1 **General Multi-Step Recurrence (GitHub-Safe)**

For each step \( l \), for each state \( s' \):

#### **Accumulation step**

$$
F'^{l}(s' : s_0)
=
F^{l}(s' : s_0)
+
Z^{t}\left( F_{t}(s' : s) \right)
\cdot
Z^{p}\left( F^{l-1}(s : s_0) \right)
$$

for every reaching state \( s \).

#### **Finalize step**

$$
F^{l}(s' : s_0)
=
Z^{c}\left( F'^{l}(s' : s_0) \right)
$$

This recurrence **is** multi-step evolution in FDS:

- \( Z^p \): prepares previous-step field  
- \( Z^t \): prepares single-step field  
- \( \cdot \): internal composition  
- \( + \): external accumulation  
- \( Z^c \): cleanup / normalization  

---

## 3.5 Dynamic System

A **Dynamic System** in FDS consists of:

- a state space \( S \),
- fields \( F \),
- transforms \( Z^p, Z^t, Z^c \),
- compositions \( \cdot \) and \( + \),
- evolution operators implementing the recurrence.

Examples:

- biased random walkers,  
- sample-mean / dice systems,  
- thermodynamic or coarse-grained systems,  
- mapped or correlated systems.

---

# 4. Affected Systems Framework

Models **directional influence**:

- Affecting system: \( A_1 = (S_1, F_1, \Theta_1) \)
- Affected system:  \( A_2 = (S_2, F_2, \Theta_2) \)

Influence appears through:

- modified fields,
- restricted transitions,
- joint constraints,
- directional mappings.

### Joint Field

$$
F_{12}(s_1, s_2 : s_{10}, s_{20})
$$

captures combined influence or correlation.

### State Space Mapping

$$
M_{12}: S_1 \to \mathcal{P}(S_2)
$$

assigns regions in \( S_2 \) under the influence of states in \( S_1 \).

Inverse mapping:

$$
M_{12}^{-1}(s_2) = \{ s_1 \mid s_2 \in M_{12}(s_1) \}
$$

---

### Classed State Spaces

Partition \( S \) into equivalence classes:

- same energy,  
- same count sum,  
- same conserved quantity.

A **classified field** aggregates fields over each class, revealing symmetries.

### Grouped Evolution

Subsystems or state blocks may evolve jointly under shared influence.

---

# 5. Ensemble Framework

An **ensemble** is a collection:

$$
\mathcal{E} = \{ A_1, A_2, \dots, A_n \}.
$$

Used for:

- comparing systems,
- evolving multiple systems in parallel,
- multi-scale modeling,
- exploring symmetry or divergence.

### Ensemble Operators

- synchronous evolution  
- asynchronous evolution  
- ensemble-conditioned evolution  

### Mapping Register

Stores:

- state space mappings,
- field correspondences,
- system distances,
- symmetry relationships.

---

# 6. Conceptual Diagram

```mermaid
graph TD

  %% Core
  StateSpace["State Space"]
  State["State"]
  Field["Field"]
  FieldValue["Field Value"]
  Kernel["Kernel"]
  Operator["Operator"]
  Path["Path"]
  PathIntegral["Path Integral (Conceptual)"]
  DynamicSystem["Dynamic System"]

  %% Affected systems
  AffectedFramework["Affected Systems Framework"]
  JointField["Joint / Combined Field"]
  Mapping["State Space Mapping"]
  ClassedSpace["Classed State Space"]
  ClassifiedField["Classified Field"]
  GroupedEvolution["Grouped Evolution"]

  %% Ensemble
  Ensemble["Ensemble of Systems"]
  EnsembleOp["Ensemble Operator"]
  MappingRegister["Mapping Register"]
  Symmetry["Symmetry / Distance / Relationships"]

  %% Core relationships
  StateSpace --> State
  StateSpace --> Field
  Field --> FieldValue
  Field --> Kernel
  Kernel --> Operator
  Operator --> Path
  Path --> PathIntegral
  StateSpace --> DynamicSystem
  Field --> DynamicSystem
  Operator --> DynamicSystem

  %% Affected relationships
  DynamicSystem --> AffectedFramework
  AffectedFramework --> JointField
  AffectedFramework --> Mapping
  AffectedFramework --> ClassedSpace
  AffectedFramework --> GroupedEvolution
  ClassedSpace --> ClassifiedField
  Mapping --> ClassedSpace

  %% Ensemble relationships
  DynamicSystem --> Ensemble
  Ensemble --> EnsembleOp
  Ensemble --> MappingRegister
  MappingRegister --> Symmetry
  AffectedFramework --> Symmetry
