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
  - [3.3 Kernels, Transforms, and Single-Step Evolution](#33-kernels-transforms-and-single-step-evolution)
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

- unify deterministic and probabilistic behavior,
- handle state-dependent transitions cleanly,
- express correlations between systems,
- describe field-based transformations and multi-step contributions.

**Field Dynamic Systems (FDS)** unify these ideas using:

- fields on states and transitions,
- transforms and compositions,
- generalized multi-step evolution,
- affected-system relationships,
- ensemble-level structure.

---

# 2. High-Level Overview

An FDS consists of:

- a **state space** \( S \),
- one or more **fields** \( F \),
- **transforms** \( Z^p, Z^t, Z^c \),
- **compositions** \( \cdot \) (internal), \( + \) (external),
- **operators** governing single-step updates,
- a **multi-step recurrence** analogous to a path integral.

Two major extensions:

1. **Affected Systems Framework** — directional influence between systems  
2. **Ensemble Framework** — collections of systems for comparison and joint analysis

---

# 3. Core FDS Concepts

## 3.1 State and State Space

A **state** is a single configuration of a system.

Examples:

- a lattice location,
- a pair \( (x, \kappa) \),
- a histogram or frequency vector,
- temperature in a thermodynamic model.

The **state space** \( S \) determines:

- all valid states,
- allowable transitions,
- reachable states,
- reaching states (states that can transition into a given state).

---

## 3.2 Fields and Field Values

A **field** assigns values to states or transitions:

$$ F : S \to \mathcal{V} $$

or

$$ F : S \times S \to \mathcal{V}. $$

Field values represent:

- weights,
- amplitudes,
- potentials,
- kernel-like contributions.

**Field Values**

- \( F(s) \) — field at state  
- \( F(s' : s) \) — field for transition \( s \to s' \)

---

## 3.3 Kernels, Transforms, and Single-Step Evolution

A **kernel** is the transition field:

$$ K(s \to s') = F_t(s' : s). $$

### Transforms

- \( Z^p \) — internal transform (applied to previous-step fields)  
- \( Z^t \) — external transform (applied to single-step fields)  
- \( Z^c \) — cleanup/normalization after accumulation  

### Compositions

- **Internal composition:** \( a \cdot b \)  
- **External accumulation:** \( a + b \)  

These govern how contributions combine during evolution.

---

## 3.4 Paths, Contributions, and Multi-Step Recurrence

### Path
A **path** is:

$$ \pi = (s_0, s_1, \ldots, s_l). $$

### Path Contribution

Each transition contributes through:

- the single-step field \( F_t \),
- transforms \( Z^p \), \( Z^t \),
- internal composition \( \cdot \).

The total contribution of a path is the structured combination of its transitions.

### Conceptual Path Integral

$$ F^{l}(s_l : s_0) = \sum_{\pi : s_0 \to s_l} \text{Contribution}(\pi) $$

### Why Not Enumerate Paths
Path enumeration grows exponentially.

FDS uses a **generalized chain-rule recurrence** over *reaching states* instead.

---

## 3.4.1 Multi-Step Recurrence (GitHub-Safe)

For each step \( l \), and each state \( s' \):

### Accumulation Step

$$ F'^{l}(s' : s_0) = F^{l}(s' : s_0) + Z^{t}(F_t(s' : s)) \cdot Z^{p}(F^{l-1}(s : s_0)) $$

(where the term is applied for every reaching state \( s \))

### Finalize Step

$$ F^{l}(s' : s_0) = Z^{c}(F'^{l}(s' : s_0)) $$

### Interpretation

- \( Z^p \): prepares previous-step field  
- \( Z^t \): prepares single-step field  
- \( \cdot \): internal composition  
- \( + \): accumulation of contributions  
- \( Z^c \): cleanup/normalization  

This chain-rule-like recurrence **is** multi-step evolution in FDS.

---

## 3.5 Dynamic System

A **Dynamic System** in the FDS sense includes:

- state space \( S \),
- fields \( F \),
- transforms \( Z^p, Z^t, Z^c \),
- compositions \( \cdot \), \( + \),
- operators implementing the recurrence.

Examples:

- biased random walkers,  
- dice/sample-mean systems,  
- thermodynamic/coarse-grained systems,  
- mapped or correlated systems.  

---

# 4. Affected Systems Framework

Models **directional influence** between dynamic systems.

Given two systems:

- \( A_1 = (S_1, F_1, \Theta_1) \)  
- \( A_2 = (S_2, F_2, \Theta_2) \)

We may define:

### Joint Field

$$ F_{12}(s_1, s_2 : s_{10}, s_{20}) $$

### State Space Mapping

$$ M_{12} : S_1 \to \mathcal{P}(S_2) $$

Inverse-like mapping:

$$ M_{12}^{-1}(s_2) = \{ s_1 \mid s_2 \in M_{12}(s_1) \} $$

### Classed State Spaces

Partition \( S \) into equivalence classes based on:

- conserved quantities,  
- symmetry properties,  
- aggregated characteristics.  

A **classified field** aggregates field values over classes, revealing symmetries.

### Grouped Evolution
Subsystems or blocks of states may evolve jointly under shared influence.

---

# 5. Ensemble Framework

An **ensemble** is:

$$ \mathcal{E} = \{ A_1, A_2, \ldots, A_n \}. $$

Used for:

- comparison of models,  
- multi-scale analysis,  
- joint evolution,  
- symmetry detection.  

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
classDiagram
    %% ======================
    %% CORE / STATIC MODULE
    %% ======================
    class State {
      <<core>>
    }

    class StateSpace {
      <<core>>
    }

    class Reachable {
      <<core>>
    }

    class MultiStepReaching {
      <<core>>
    }

    class SingleFieldValue {
      <<core>>
    }

    class FieldValue {
      <<core>>
    }

    class Field {
      <<core>>
    }

    class FieldTransform {
      <<core>>
    }

    class FieldComposition {
      <<core>>
    }

    StateSpace "1" o-- "*" State
    Reachable --> State
    Reachable --> StateSpace
    MultiStepReaching --> Reachable
    FieldValue --> SingleFieldValue
    Field --> FieldValue
    Field --> StateSpace
    FieldTransform --> Field
    FieldComposition --> Field

    %% ======================
    %% DYNAMIC MODULE
    %% ======================
    class Kernel {
      <<dynamic>>
    }

    class SingleStepField {
      <<dynamic>>
    }

    class MultiStepField {
      <<dynamic>>
    }

    Kernel --> State
    Kernel --> FieldValue

    SingleStepField --> Kernel
    SingleStepField --> Reachable
    SingleStepField --> Field

    MultiStepField --> SingleStepField
    MultiStepField --> FieldComposition
    MultiStepField --> FieldTransform
    MultiStepField --> MultiStepReaching

    %% ======================
    %% DYNAMIC SYSTEMS MODULE
    %% ======================
    class StaticDynamicSystem {
      <<system>>
    }

    class FieldStaticDynamicSystem {
      <<system>>
    }

    class FieldDynamicSystem {
      <<system>>
    }

    StaticDynamicSystem --> StateSpace
    StaticDynamicSystem --> Reachable
    StaticDynamicSystem --> MultiStepReaching

    FieldStaticDynamicSystem --|> StaticDynamicSystem
    FieldStaticDynamicSystem --> Field

    FieldDynamicSystem --|> FieldStaticDynamicSystem
    FieldDynamicSystem --> MultiStepField

    %% ======================
    %% AFFECTING FRAMEWORK
    %% ======================
    class CorrelatedState {
      <<affecting>>
    }

    class CorrelatedStateSpace {
      <<affecting>>
    }

    class AffectingGroupState {
      <<affecting>>
    }

    class AffectingGroupStateSpace {
      <<affecting>>
    }

    class AffectedSystemsMapping {
      <<affecting>>
    }

    class AffectedReachable {
      <<affecting>>
    }

    class AffectingMultiStepReaching {
      <<affecting>>
    }

    class AffectingSingleStep {
      <<affecting>>
    }

    class AffectingMultiStepField {
      <<affecting>>
    }

    class AffectedSystemsOperator {
      <<affecting>>
    }

    class AffectingFDS {
      <<affecting>>
    }

    CorrelatedState --|> State
    CorrelatedStateSpace --|> StateSpace
    AffectingGroupState --|> State
    AffectingGroupStateSpace --|> StateSpace

    AffectedSystemsMapping --> CorrelatedStateSpace
    AffectedSystemsMapping --> AffectingGroupStateSpace

    AffectedReachable --|> Reachable
    AffectingMultiStepReaching --|> MultiStepReaching

    AffectingSingleStep --|> SingleStepField
    AffectingMultiStepField --|> MultiStepField
    AffectedSystemsOperator --> AffectingMultiStepField

    AffectingFDS --|> FieldDynamicSystem
    AffectingFDS --> AffectingGroupState
    AffectingFDS --> AffectingGroupStateSpace
    AffectingFDS --> AffectedSystemsMapping
    AffectingFDS --> AffectedReachable
    AffectingFDS --> AffectingMultiStepReaching
    AffectingFDS --> AffectedSystemsOperator

    %% ======================
    %% ENSEMBLE MODULE (STUB)
    %% ======================
    class EnsembleOperator {
      <<ensemble>>
      + channels : List[AffectingGroupsEvolution]
      + stepAll()
      + collectResults()
    }

    class Ensemble {
      <<ensemble>>
      + systems : List[FieldDynamicSystem]
      + channels : List[AffectingGroupsEvolution]
      + operator : EnsembleOperator
      + run()
    }

    Ensemble --> FieldDynamicSystem
    EnsembleOperator --> Ensemble
    MappingRegister --> Ensemble
    MappingRegister --> FieldDynamicSystem

