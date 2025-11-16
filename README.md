# Field Dynamic Systems (FDS) Framework

> A general framework for modeling dynamic systems using **weighted states (fields)**, **generalized chain-rule style evolution**, and **relationships between multiple systems** (affected systems and ensembles).

This repository contains a Python implementation of the **Field Dynamic Systems (FDS)** framework.

This README presents the **theoretical foundation only**, independent of implementation details, so the conceptual structure stands clearly on its own.

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

## 1. Motivation

Many systems in nature and engineering evolve over time:

- a particle moving on a lattice,
- a sample mean changing as more data points are added,
- a body cooling toward its environment,
- multiple systems interacting under conservation laws.

Traditional approaches (differential equations, Markov chains, stochastic processes) handle specific patterns of evolution but often struggle to:

- unify deterministic and probabilistic evolution,
- incorporate state-dependent or time-dependent behavior cleanly,
- handle complex correlation between systems,
- express evolution through flexible algebraic structures (fields, transforms, compositions).

**Field Dynamic Systems (FDS)** generalize these ideas by introducing:

- **fields** attached to states or transitions,
- **operators** that use structured transforms and compositions,
- a **path-integral viewpoint** implemented via a recurrence rather than explicit path enumeration,
- extensions to **affected systems** and **ensembles**.

This framework is designed to be **modular**, **extensible**, and **theoretically clean**.

---

## 2. High-Level Overview

At a high level, an FDS consists of:

- a **state space** $S$,
- one or more **fields** $F$ defined on states or transitions,
- **transforms** (internal / external) that modify field values,
- **compositions** (internal / external) for combining field values,
- **operators** $\Theta$ that evolve the system,
- a **multi-step recurrence** that generalizes path-integral evolution.

Built on top of this are:

1. the **Affected Systems Framework** — directional influence between systems,
2. the **Ensemble Framework** — collections of systems studied jointly.

---

## 3. Core FDS Concepts

### 3.1 State and State Space

**State**  
A **state** is a configuration of a system at a given instant. It is the basic unit of description.

Examples (conceptual):

- a position on a 1D or 2D lattice,
- a pair $(x, \kappa)$, where $\kappa$ is a “bias” or internal parameter,
- a histogram / count vector,
- a macroscopic variable like temperature or sample mean.

**State Space**  
The **state space** $S$ is the set of all possible states.

It determines:

- which states are allowed,
- which transitions are allowed,
- how reachable / reaching sets are formed,
- the domain on which fields are defined.

**Reachable States**  
Given an initial state $s_0$, the **reachable states** at step $\ell$ are states that can be reached in $\ell$ steps from $s_0$.

**Reaching States**  
For a given state $s'$, the **reaching states** are those $s$ such that a single-step transition $s \to s'$ is allowed.

These concepts are central to the multi-step recurrence.

---

### 3.2 Fields and Field Values

A **field** associates a value to a state, or to a transition between states:

$$
F : S \to \mathcal{V}
\qquad\text{or}\qquad
F : S \times S \to \mathcal{V},
$$

where $\mathcal{V}$ may be:

- real numbers,
- complex numbers,
- vectors,
- or any structured value space.

Field values may represent:

- weights or preferences,
- amplitudes,
- potential-like quantities,
- kernel values controlling transitions,
- contributions to multi-step evolution.

**Field Value**

- $F(s)$ is the field at state $s$.
- $F(s' : s)$ is the field associated to the transition $s \to s'$.

Fields are the primary carriers of information that drive the evolution in FDS.

---

### 3.3 Kernels, Operators, and Single-Step Evolution

A **kernel** in FDS is simply a field on transitions that expresses the “raw tendency” or contribution of moving from one state to another:

$$
K(s \to s') = F_t(s' : s).
$$

This is not required to be a probability; it is a **contribution** that will later be transformed and combined.

#### Transforms

FDS distinguishes three conceptual transforms:

- **$Z^p$ (internal transform)**  
  Applied to **previous-step field values** $F^{\,l-1}(s : s_0)$.
- **$Z^t$ (external transform)**  
  Applied to **single-step field values** $F_t(s' : s)$.
- **$Z^c$ (cleanup / final transform)**  
  Applied to the accumulated field $F'^{\,l}(s' : s_0)$ to produce $F^{\,l}(s' : s_0)$.

These transforms may encode normalization, biasing, scaling, damping, stability adjustments, or other problem-specific structure.

#### Compositions

Two kinds of compositions are used:

- **Internal composition** $*$  
  Combines transformed previous-step fields and transformed single-step fields.

- **External composition** $+$  
  Accumulates contributions from different reaching states $s$ into a single $F'^{\,l}(s' : s_0)$.

The combination of transforms and compositions defines the **single-step evolution rule** for contributions.

---

### 3.4 Paths, Contributions, and Multi-Step Recurrence

#### Path

A **path** is an ordered sequence of states:

$$
\pi = (s_0, s_1, \dots, s_\ell).
$$

Each edge $s_{k-1} \to s_k$ corresponds to a single-step transition and has an associated field contribution.

#### Path Contribution

Instead of working directly with probabilities, FDS works with **contributions** built from:

- the single-step field $F_t(s_{k} : s_{k-1})$,
- internal / external transforms ($Z^p$, $Z^t$),
- and internal composition $*$.

A path’s total contribution is the structured combination of its segment contributions.

#### Path Integral (Conceptual View)

Conceptually, the total contribution (or weight) supporting state $s_\ell$ after $\ell$ steps from $s_0$ is a sum over all paths:

$$
F^{\,\ell}(s_\ell : s_0)
=
\sum_{\pi : s_0 \to s_\ell}
\text{Contribution}(\pi).
$$

This is analogous to a path integral or sum-over-histories picture.

#### Why FDS Does Not Enumerate Paths

Explicitly enumerating all paths is usually exponential in the number of steps.

Instead, FDS uses a **generalized chain-rule style recurrence** that operates only on **reaching states** and **multi-step fields**.

#### General Multi-Step Recurrence

For each step $l$ and for each state $s'$:

1. For each **reaching state** $s$ of $s'$ (i.e. each $s$ with a valid transition $s \to s'$), update the intermediate field:

$$
F'^{l}(s' : s_0)
=
F^{l}(s' : s_0)
$+$
Z^{t}\left( F_{t}(s' : s) \right)
$*$
Z^{p}\left( F^{l-1}(s : s_0) \right)
$$

2. After aggregating contributions from all reaching states $s$, finalize the $l$-step field at $s'$:

   $$
   F^{\,l}(s' : s_0)
   \leftarrow
   Z^c\!\big[F'^{\,l}(s' : s_0)\big].
   $$

This recurrence is the FDS analogue of a **chain rule** for multi-step evolution:

- $Z^p$ prepares the previous-step field at $s$,
- $Z^t$ prepares the single-step field from $s$ to $s'$,
- $*$ combines them into a contribution for paths passing through $s$,
- $+$ sums contributions from all such $s$,
- $Z^c$ cleans up / normalizes the resulting $l$-step field.

Thus, FDS realizes a path-integral style evolution **without** enumerating paths explicitly.

---

### 3.5 Dynamic System

A **Dynamic System** in FDS is defined by:

- a state space $S$,
- a collection of fields over states and/or transitions (e.g. $F, F_t, \dots$),
- a set of transforms $(Z^p, Z^t, Z^c)$,
- compositions $(*, +)$,
- and one or more evolution operators $\Theta$ that implement the multi-step recurrence.

Conceptually, a dynamic system is the pairing of:

- a **structural layer** (state space + fields), and
- an **evolution layer** (transforms + compositions + operator).

Examples (conceptual only):

- A biased random walker where fields encode directional preference and magnitude.
- A dice-based system where fields count how many ways a mean or configuration can occur.
- A thermodynamic system whose macroscopic evolution is induced from a finer “microscopic” system via mappings.

The **Affected Systems** and **Ensemble** frameworks build upon this notion of dynamic system.

---

## 4. Affected Systems Framework

The **Affected Systems Framework** captures situations where one system’s behavior depends on another system (or multiple systems).

Consider:

- Affecting system: $A_1 = (S_1, F_1, \Theta_1)$  
- Affected system: $A_2 = (S_2, F_2, \Theta_2)$  

Influence can appear as:

- field values in $A_2$ depending on states / fields in $A_1$,
- allowable transitions in $A_2$ depending on $A_1$,
- joint constraints (e.g. conservation) linking both systems,
- coupled evolution rules.

### 4.1 Systems and Joint Fields

A **joint field** defines combined contributions or preferences for pairs of states:

$$
F_{12}(s_1, s_2 : s_{10}, s_{20}),
$$

or for joint transitions.

- If it factorizes into independent parts, the systems are effectively uncorrelated.
- If not, the systems are correlated or coupled.

In the affected framework, we especially care about **directionality**: which system is affecting and which is affected.

---

### 4.2 State Space Mappings

To express how an affecting system relates to an affected system, we introduce a **state space mapping**:

$$
M_{12} : S_1 \to \mathcal{P}(S_2),
$$

where $\mathcal{P}(S_2)$ is the power set of $S_2$.

This mapping assigns to each $s_1 \in S_1$ a subset of $S_2$ (a region, block, or fiber) which is “under the influence” of $s_1$.

An inverse-like relation:

$$
M_{12}^{-1}(s_2) = \{ s_1 \in S_1 \mid s_2 \in M_{12}(s_1) \},
$$

identifies which states in $S_1$ affect a given $s_2$.

These mappings:

- define **affected reachable / reaching sets** across systems,
- constrain evolution in the affected system,
- encode structural relationships such as conservation or shared invariants.

---

### 4.3 Classed State Spaces and Classified Fields

Sometimes, influence and symmetry live at the level of **equivalence classes of states** rather than individual states.

- Define an equivalence relation $\sim$ on $S$, e.g. “same energy”, “same total count”.
- Partition $S$ into **equivalence classes** $C_1, C_2, \dots$.

The resulting **classed state space** treats each class as a macro-state.

A **classifier mapping** between systems respects these classes, mapping classes in $S_1$ to classes in $S_2$.

A **classified field** aggregates field values over each class (for example, summing or averaging contributions). This reveals:

- symmetry at the class level,
- relationships reminiscent of “symmetry $\leftrightarrow$ conserved quantity”.

Classed state spaces are natural tools for expressing **thermodynamic-like** or **coarse-grained** views within the FDS framework.

---

### 4.4 Grouped Evolution

Real systems often involve **groups** of states or subsystems that evolve together.

In the affected framework we can:

- define groups (blocks) of states or subsystems,
- define **grouped operators** that evolve entire groups coherently,
- track **grouped evolution**, where each step advances all affected subsystems under joint rules.

This is useful when:

- modeling multiple interacting subsystems with shared constraints,
- evolving correlated walkers,
- expressing coupled thermodynamic bodies, etc.

The Affected Systems Framework thus provides a structured way to talk about **who affects whom**, and **how that influence propagates through the fields and evolution operators**.

---

## 5. Ensemble Framework

While the affected framework focuses on directional influence, the **Ensemble Framework** focuses on **collections** of dynamic systems studied together.

An **ensemble** is:

$$
\mathcal{E} = \{ A_1, A_2, \dots, A_n \},
$$

where each $A_i$ is a valid dynamic system in the FDS sense.

Ensembles allow us to:

- compare systems side-by-side,
- simulate multiple systems in parallel,
- study fine vs. coarse models of the same phenomenon,
- detect symmetries or approximate equivalences between systems.

---

### 5.1 Ensemble Operators

An **ensemble operator** acts on multiple systems in the ensemble:

- **Synchronous**: evolve all systems by one step.
- **Asynchronous**: evolve a subset of systems according to a schedule or condition.
- **Coupled**: evolution rules that depend on ensemble-wide quantities or statistics.

Ensemble operators are natural when exploring:

- how different parameter sets behave,
- how multiple candidate models evolve,
- how systems might converge or diverge in behavior.

---

### 5.2 Mapping Register and System Relationships

Within an ensemble, we often need to know **how systems correspond to each other**.

A **mapping register** conceptually stores:

- mappings between state spaces,
- correspondences between observables,
- rules for mapping or transforming fields,
- metrics or distances between systems.

This enables:

- defining **distances** between systems,
- analyzing **path evolution symmetry** (do two systems follow the same evolution up to a mapping?),
- capturing **classification-based symmetry** across multiple systems.

The ensemble framework, together with mapping registers and metrics, turns FDS into a tool for **model comparison, symmetry detection, and structural analysis** of dynamic systems.

---

## 6. Conceptual Diagram

Below is a high-level conceptual diagram of the FDS framework, emphasizing the role of **Affected Systems** and **Ensembles**.

> GitHub supports Mermaid diagrams natively.

```mermaid
graph TD

  %% Core layer
  StateSpace["State Space"]
  State["State"]
  Field["Field"]
  FieldValue["Field Value"]
  Kernel["Kernel"]
  Operator["Operator"]
  Path["Path"]
  PathIntegral["Path Integral (Conceptual)"]
  DynamicSystem["Dynamic System"]

  %% Affected systems layer
  AffectedFramework["Affected Systems Framework"]
  JointField["Joint / Combined Field"]
  Mapping["State Space Mapping"]
  ClassedSpace["Classed State Space"]
  ClassifiedField["Classified Field"]
  GroupedEvolution["Grouped Evolution"]

  %% Ensemble layer
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

  %% Affected systems relationships
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
