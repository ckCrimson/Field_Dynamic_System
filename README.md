# Field Dynamic Systems (FDS) Framework

> A conceptual framework for modeling dynamic systems with **weighted states (fields)**, **path-integral style evolution**, and **relationships between multiple systems** (affected systems and ensembles).

This repository contains a Python implementation of the **Field Dynamic Systems (FDS)** framework.  
This README focuses on the **theoretical foundations only** – no implementation details – so that the core ideas stand on their own.

---

## Table of Contents

- [1. Motivation](#1-motivation)
- [2. High-Level Overview](#2-high-level-overview)
- [3. Core FDS Concepts](#3-core-fds-concepts)
  - [3.1 State and State Space](#31-state-and-state-space)
  - [3.2 Fields and Field Values](#32-fields-and-field-values)
  - [3.3 Kernels, Operators, and Evolution](#33-kernels-operators-and-evolution)
  - [3.4 Paths and Path Integrals](#34-paths-and-path-integrals)
  - [3.5 Dynamic System](#35-dynamic-system)
- [4. Affected Systems Framework](#4-affected-systems-framework)
  - [4.1 Systems and Joint Fields](#41-systems-and-joint-fields)
  - [4.2 Affecting vs Affected Systems](#42-affecting-vs-affected-systems)
  - [4.3 State Space Mappings](#43-state-space-mappings)
  - [4.4 Classed State Spaces and Classified Fields](#44-classed-state-spaces-and-classified-fields)
  - [4.5 Grouped Evolution](#45-grouped-evolution)
- [5. Ensemble Framework](#5-ensemble-framework)
  - [5.1 Ensemble of Dynamic Systems](#51-ensemble-of-dynamic-systems)
  - [5.2 Ensemble Operators](#52-ensemble-operators)
  - [5.3 Mapping Register and System Relationships](#53-mapping-register-and-system-relationships)
- [6. Conceptual Diagram](#6-conceptual-diagram)
- [7. Where to Go Next](#7-where-to-go-next)

---

## 1. Motivation

Many systems in nature and engineering evolve over time:

- a particle moving in space,
- a sample mean changing as more data points are added,
- a body cooling toward its environment,
- multiple systems interacting under conservation laws.

Classical tools cover special cases:

- **Deterministic dynamical systems** use recurrence relations or differential equations.
- **Markov chains** use probability vectors and transition matrices.

However, these often struggle to:

- handle time-varying or state-dependent transition structure cleanly,
- unify deterministic and probabilistic behavior,
- express correlations and symmetries between different systems.

**Field Dynamic Systems (FDS)** generalize these ideas by attaching **fields** to states, and using these fields to define how the system evolves. The framework then extends to **affected systems** (systems influencing each other) and **ensembles** (collections of systems analyzed together).

---

## 2. High-Level Overview

At a high level, an FDS consists of:

- a **state space** $S$: all possible states a system can be in,
- one or more **fields** $F$: values associated to states or transitions, which encode “how much the system likes” those states or transitions,
- **operators** $\Theta$: rules that use fields to advance the system in time (probabilistic or deterministic),
- a **path-integral style** view of multi-step evolution: probabilities are built by summing contributions over all possible paths.

On top of this, the framework has two higher-level constructions:

1. **Affected Systems Framework**: multiple systems where one system’s evolution is influenced by another (or by several others).
2. **Ensemble Framework**: structured collections of systems, used to compare, couple, or jointly evolve different dynamic systems.

---

## 3. Core FDS Concepts

### 3.1 State and State Space

**State**  
A *state* is a single configuration of a system at an instant. It is the basic unit of description.

Examples (conceptual, not code):

- a position on a 1D or 2D lattice,
- a pair $(x, \kappa)$ describing position and a bias/“momentum-like” parameter,
- a vector of frequencies representing how many times each dice face has appeared,
- a macroscopic variable like temperature or sample mean.

**State Space**  
The **state space** $S$ is the set of all states a system can occupy.

It determines:

- which states are allowed,
- which transitions between states are possible,
- what kind of distances or topologies we can define.

**Reachable States**  
Given an initial state $s_0$, the **reachable states** are those that can be reached from $s_0$ by valid transitions in any finite number of steps.

**Reaching States**  
For a given state $s$, the **reaching states** are those that can transition directly *into* $s$.  
This concept is key when we build recurrence relations for multi-step evolution.

---

### 3.2 Fields and Field Values

**Field**  
A **field** assigns a value to each state or transition. In general,
$$
F: S \to \mathcal{V}
\quad \text{or} \quad
F: S \times S \to \mathcal{V},
$$
where $\mathcal{V}$ is some value space (e.g. real numbers, complex numbers, vectors, tensors).

A field value might represent:

- a weight or preference,
- a probability amplitude,
- a potential or energy-like quantity,
- a kernel value controlling transitions.

Fields are the main way we encode information that influences the dynamics.

**Field Value**  
The **field value** at state $s$ is $F(s)$.  
The **field value** for a transition $s \to s'$ is $F(s' \mid s)$.

For many systems, the field values (or their norms) define transition probabilities after normalization.

---

### 3.3 Kernels, Operators, and Evolution

**Kernel**  
A **kernel** is a rule (or a field) that assigns weights to candidate next states given a current state. Conceptually:
$$
K(s \to s') = \text{weight}(s, s').
$$

The kernel describes the *tendency* of the system to move from one state to another before normalization.

**Operator**  
An **operator** $\Theta$ updates the system by one step using fields and kernels. Depending on the context, it can:

- produce a new **probability distribution** over states, or
- produce a new **observed state**.

Two common types:

- **Probabilistic operator**: samples the next state from a distribution defined by fields/kernels.
- **Deterministic operator**: picks a single next state (e.g., via an expectation or argmax rule).

By applying $\Theta$ repeatedly, we get:
$$
s_{k+1} = \Theta(s_k)
\quad \text{or} \quad
P^{k+1} = \Theta(P^k),
$$
where $P^k$ is a probability distribution at step $k$.

---

### 3.4 Paths and Path Integrals

**Path**  
A **path** is a sequence of states:
$$
\pi = (s_0, s_1, \dots, s_\ell),
$$
obtained by applying the operator $\Theta$ step-by-step (possibly probabilistically).

**Path Probability**  
If transitions are Markovian, the probability of a path is:
$$
\mathbb{P}[\pi] = \prod_{k=1}^{\ell} P(s_k \mid s_{k-1}),
$$
where each $P(s_k \mid s_{k-1})$ is derived from fields and kernels.

**Path Integral / Sum Over Paths**  
The probability of being in a state $s_\ell$ after $\ell$ steps is obtained by summing over all paths that end at $s_\ell$:
$$
P^\ell(s_\ell \mid s_0) = \sum_{\pi : s_0 \to s_\ell} \mathbb{P}[\pi].
$$

In practice, FDS uses a **recurrence relation** over *reaching* states instead of explicitly enumerating all paths. This is the core multi-step construction.

---

### 3.5 Dynamic System

A **Dynamic System** in the FDS sense is a tuple containing:

- a state space $S$,
- one or more fields $F$ over $S$,
- one or more operators $\Theta$ that evolve states/distributions using these fields.

Examples of dynamic systems (conceptually):

- A classical random walker on a line where fields bias direction.
- A “dice system” where the state is the sample mean, and fields count how many ways the mean can arise.
- A thermodynamic system whose temperature evolution is related to a mapped dice system.

The rest of the framework – **affected systems** and **ensembles** – builds on these dynamic systems.

---

## 4. Affected Systems Framework

The **Affected Systems Framework** describes multiple dynamic systems that are **not isolated**: one system’s behavior is influenced by another.

Think in terms of:

- **primary systems** (each with its own state space, fields, and operators), and
- **relations** that describe how one system affects another.

### 4.1 Systems and Joint Fields

Consider two dynamic systems:

- $A_1 = (S_1, F_1, \Theta_1)$,
- $A_2 = (S_2, F_2, \Theta_2)$.

We can define a **joint (combined) field**:
$$
F_{12}(s_1, s_2 : s_{10}, s_{20}),
$$
which assigns a weight or amplitude to pairs of states (or transitions) from both systems.

- If the joint field factorizes cleanly into $F_1$ and $F_2$, the systems are *uncorrelated*.
- If not, the systems are *correlated* – their evolutions are statistically or structurally linked.

In the affected framework, we care especially about **directional influence**.

---

### 4.2 Affecting vs Affected Systems

We distinguish:

- **Affecting system**: a system whose state or evolution **modifies** the fields or allowable transitions of another system.
- **Affected system**: a system whose fields, kernels, or operators depend on the affecting system.

Examples (conceptual):

- A thermodynamic system whose temperature evolution depends on another system representing an environment or heat bath.
- A dice system whose rules depend on some external “control” system.

The influence can appear as:

- modifications to field values (e.g., scaling or bias based on another system’s state),
- activation or deactivation of certain transitions,
- joint constraints that couple the dynamics of both systems.

---

### 4.3 State Space Mappings

To formalize relations between systems, we introduce **state space mappings**:
$$
M_{12}: S_1 \to \mathcal{P}(S_2),
$$
where $\mathcal{P}(S_2)$ is the set of subsets of $S_2$.

This mapping says:

> To each state in $S_1$, we associate one or more states (or a region) in $S_2$.

We also consider an “inverse-like” mapping:
$$
M_{12}^{-1}(s_2) = \{ s_1 \in S_1 \mid s_2 \in M_{12}(s_1) \},
$$
which tells us which states in $S_1$ influence a particular state in $S_2$.

Within the affected framework, these mappings:

- identify which parts of the affected state space sit “under the influence” of which affecting states,
- define **affected reachable/reaching** structures across systems,
- can encode physical constraints (e.g., conservation laws, shared invariants).

---

### 4.4 Classed State Spaces and Classified Fields

Sometimes, influence and symmetry appear at the level of **equivalence classes of states** rather than individual states.

**Equivalence Relation and Classes**

- Define an equivalence relation $\sim$ on a state space $S$ (e.g., “same energy”, “same total count”).
- The state space is partitioned into **equivalence classes** $C_1, C_2, \dots$, each containing states that share some conserved or relevant quantity.

**Classed State Space**

- The **classed state space** is the set of these equivalence classes.
- Each class acts as a “macro-state”.

**Classifier Mapping**

A **classifier mapping** is a mapping between state spaces that respects equivalence classes, i.e., it sends each class in $S_1$ to a class in $S_2$.

**Classified Fields**

A **classified field** is a field that is aggregated over equivalence classes:

- Instead of assigning a field value to each microstate, we sum or combine them over an entire class.
- This often exposes symmetries more clearly, especially those linked to conserved quantities.

This structure can reveal:

- **Field symmetry** between systems at the level of classes.
- Relationships reminiscent of Noether’s theorem: symmetries ↔ conserved quantities.

---

### 4.5 Grouped Evolution

Real systems often involve **groups** of states or subsystems evolving together.

In the affected systems framework, we can:

- group states or subsystems into **blocks**,
- define **grouped operators** that evolve each block in a coordinated way,
- track **grouped evolution**, where each step advances all affected subsystems under joint constraints.

Conceptually, grouped evolution answers questions like:

- “How does this subsystem evolve if these other subsystems evolve alongside it?”
- “How does a group of correlated walkers behave under a shared field or constraint?”
- “How do multiple thermodynamic bodies influence each other under conservation laws?”

---

## 5. Ensemble Framework

While the Affected Systems Framework focuses on **influence and correlation**, the **Ensemble Framework** focuses on **collections** of dynamic systems considered together.

### 5.1 Ensemble of Dynamic Systems

An **ensemble** is a (finite or countable) collection of dynamic systems:
$$
\mathcal{E} = \{ A_1, A_2, \dots, A_n \},
$$
where each $A_i = (S_i, F_i, \Theta_i)$ is a dynamic system in the FDS sense.

Ensembles allow us to:

- compare different systems side-by-side,
- run multiple systems in parallel,
- study families of models (e.g., fine vs coarse approximations),
- explore relationships and symmetries between systems.

---

### 5.2 Ensemble Operators

An **ensemble operator** acts on all (or a selected subset of) systems in the ensemble.

Conceptually:

- **Synchronous evolution**: evolve all systems one step simultaneously.
- **Asynchronous evolution**: evolve specific systems according to some schedule.
- **Coupled evolution**: apply different operators depending on ensemble-wide conditions or statistics.

This is useful when:

- testing different parameter choices for the same conceptual model,
- comparing the evolution of fine-grained and coarse-grained versions of a system,
- evolving multiple interacting subsystems that are not strictly in an affected relationship but still belong to a single study.

---

### 5.3 Mapping Register and System Relationships

Within an ensemble, we often care about **how systems correspond to each other**.

A **mapping register** is a conceptual registry that tracks:

- state space mappings between systems,
- which observables correspond across systems,
- how fields should be transformed or compared.

Through this, we can define:

- **distances** between systems (e.g., how far apart their distributions or fields are),
- **path evolution symmetry** (whether two systems follow the same evolution under a mapping),
- **classification-based symmetry** across the ensemble (e.g., mapping classes of states in one system to classes in another).

The ensemble framework, together with mappings and distances, turns FDS into a **laboratory for model comparison, symmetry detection, and approximate equivalences** between different dynamic systems.

---

## 6. Conceptual Diagram

Below is a high-level conceptual diagram of the FDS framework, emphasizing the role of **Affected Systems** and **Ensembles**.

> Note: GitHub supports Mermaid diagrams natively.  
> The `click` directives are optional; they can be wired to documentation sections or files.

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
  PathIntegral["Path Integral"]
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
  Symmetry["Symmetry / Distance"]

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
