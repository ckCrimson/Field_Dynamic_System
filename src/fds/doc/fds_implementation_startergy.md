
# Field Dynamic Systems – Implementation Guide (up to Affecting FDS)

This document is a **step-wise guide** for implementing systems in the **Field Dynamic Systems (FDS)** framework, up to and including the **Affecting Field Dynamic System (AffectingFDS)**.

It assumes you already understand the theory (states, fields, multi-step recurrence, affected systems). For derivations and equations, see the theory **[doc/fds_brief_theory](doc/fds_brief_theory)**. 

Here we focus on **what to implement, in what order, and why**. The structure follows the modules in the implementation strategy. 

---

## 0. Big Picture: Layers and Order of Implementation

Implementation is naturally layered:

1. **Static / Core Module**

   * State module (states, spaces, reachability)
   * Field module (field values, fields, transforms, compositions)
2. **Dynamic Module**

   * Single-step dynamics (Kernel, SingleStepField)
   * Multi-step dynamics (MultiStepField)
3. **Dynamic Systems Module**

   * StaticDynamicSystem
   * FieldStaticDynamicSystem
   * FieldDynamicSystem
4. **Affecting Framework**

   * Affecting / correlated states and spaces
   * Grouping of systems that affect each other
   * Affected reachability & multi-step
   * Affecting-specific dynamics
   * AffectingFDS and group evolution

**Recommended implementation order:**

1. Core (State + Field)
2. Dynamic (SingleStep + MultiStep)
3. Dynamic Systems
4. Affecting Framework (up to AffectingFDS & AffectingGroupsEvolution)

Optional pieces are marked `[O]` – they are not needed for every use case but make the framework more expressive.

---

## 1. Static / Core Module

The **static module** defines *what the system is made of* before any evolution: states, spaces, and fields.

### 1.1 State Module

This is the “geometry” of your system.

#### 1.1.1 `State`

* **What it is:** A single configuration of the system.
* **Typical representation:** tuple, vector, small dataclass (e.g., `(x, κ)` for walker; histogram vector for dice).
* **Responsibilities:**

  * Uniquely identify a configuration.
  * Support hashing / equality so it can be used as keys in dictionaries/sets.
* **Design tip:** Make `State` as *lightweight* as possible (no heavy logic, just identity and minimal helpers).

---

#### 1.1.2 `StateSpace`

* **What it is:** A container / interface for all allowed states.
* **Responsibilities:**

  * Represent the *type* of states (e.g., integer lattice, continuous interval, finite combinatorial set).
  * Provide factories / helpers for:

    * membership checks,
    * iterating over local neighbourhoods or predefined sets,
    * optional indexing / ID schemes if needed.
* **Conceptually:** Every dynamic system has exactly one primary `StateSpace`.

---

#### 1.1.3 `[O] StateSpaceMapping`

* **What it is:** A mapping between two state spaces, `StateSpaceA → StateSpaceB`.
* **When you need it:**

  * When relating *two* systems (fine vs coarse, dice vs thermo, etc.).
  * When building mappings used later in the affected framework or ensembles.
* **Responsibilities:**

  * Given a state in `S1`, return the corresponding state(s) or region in `S2`.
  * Optionally provide an inverse-like mapping.

You don’t need this for a single isolated system, but it becomes important for **multi-system relations** and **AffectingFDS**.

---

#### 1.1.4 `[O] StateOperations`

* **What it is:** Binary operations on states: `State × State → State`.
* **Use cases:**

  * Adding positions,
  * Composing labels,
  * Custom algebraic structures on states.
* **You can skip it** if your system doesn’t require explicit state–state operations and you only care about transitions via kernels.

---

#### 1.1.5 `Reachable`

* **What it is:** For a state `s`, provides the set of states reachable in **one step**.
* **Responsibilities:**

  * Given `s`, compute `V[s] = { s' | direct transition s → s' is allowed }`.
* **Use:**

  * Core primitive for both single-step dynamics and multi-step recurrence. 

This is the *local neighbourhood oracle* of the system.

---

#### 1.1.6 `[O] Reaching`

* **What it is:** Inverse notion of `Reachable`.
* **Responsibilities:**

  * Given `s'`, give the set of predecessors `R[s'] = { s | s' ∈ V[s] }`.
* **Use:**

  * In principle, the multi-step algorithm can be written in terms of **reaching states**; in practice, you can compute this on the fly from `Reachable`.
* **Optional:** You can implement it explicitly for clarity or optimization.

---

#### 1.1.7 `MultiStepReaching`

* **What it is:** Builder for the set of states reachable after `ℓ` steps from `s0`.
* **Responsibilities:**

  * Given `Reachable` and initial state `s0`, iteratively compute sets:

    * `L⁰[s0] = {s0}`,
    * `L¹[s0] = V[s0]`,
    * `L²[s0]`, etc.
* **Use:**

  * Drives the *frontier* of the multi-step evolution, telling which states we need to consider at each step.

---

### 1.2 Field Module

This is the “weighting” and “information” layer.

#### 1.2.1 `SingleFieldValue`

* **What it is:** The atomic field value type (real, complex, vector, etc.).
* **Responsibilities:**

  * Store a single value, independent of operations.
* **Design tip:** Keep it generic enough to support both real and complex systems.

---

#### 1.2.2 `SingleFieldValueComposition`

* **What it is:** Binary operation on single field values.
* **Example semantics:**

  * Multiplication of amplitudes or weights,
  * Addition in log-domain,
  * Other algebra depending on the model.
* **Use:** This will later become the *internal composition* in your multi-step recurrence.

---

#### 1.2.3 `SingleFieldValueTransform`

* **What it is:** A transform on single field values, e.g.:

  * Normalization preparation,
  * Non-linear activation,
  * Mapping between representations.
* **Used for:**

  * `Z^p`, `Z^t`, `Z^c`-style transforms at the field-value level.

---

#### 1.2.4 `AdditionSingleFieldValue` (extends `SingleFieldValueComposition`)

* **What it is:** A concrete “addition” composition.
* **Typical use:**

  * External accumulation across contributions from different reaching states.
* **Why separate:** Keeps the core interface general, while providing a default “plus”-like operation.

---

#### 1.2.5 `NormTransform` (extends `SingleFieldValueTransform`)

* **What it is:** Converts a field value to a real non-negative scalar (norm).
* **Used for:**

  * Turning raw field values into weights/probabilities.
* **Examples:**

  * ( |z|^2 ) for complex amplitudes,
  * absolute value or identity for real-positive fields.

---

#### 1.2.6 `FieldValue`

* **What it is:** A *wrapper* over `SingleFieldValue` + its operations.
* **Responsibilities:**

  * Encapsulate:

    * the underlying single value,
    * its addition (`AdditionSingleFieldValue`),
    * its norm (`NormTransform`),
    * possibly other transforms.
* **Benefit:** Makes field-level code independent of the exact scalar type.

---

#### 1.2.7 `[O] FieldFunction`

* **What it is:** A function-like object `State → FieldValue`.
* **Role:**

  * Represents “field as a function on state space”.
* **Optional:** In many implementations, `Field` will embed this behaviour directly.

---

#### 1.2.8 `Field`

* **What it is:** A mapping from `StateSpace` to `FieldValue`, plus structural metadata.
* **Responsibilities:**

  * Store field values on states or transitions.
  * Provide lookup and iteration over supported states.
* **Important:** This is the primary carrier of information that drives the dynamics.

---

#### 1.2.9 `FieldTransform`

* **What it is:** Transform from one `Field` to another over the *same* `StateSpace`.
* **Examples:**

  * Normalization,
  * Filtering,
  * Projection,
  * Applying `Z^p`, `Z^t`, `Z^c` at the field level.

---

#### 1.2.10 `FieldComposition`

* **What it is:** Composition operation between two `Field`s over the same `StateSpace`.
* **Use:**

  * Realize the *internal composition* in the multi-step recurrence at the field level:

    * combine single-step fields with previous-step fields,
    * combine global and local contributions.

---

**Static/Core module complete = State module + Field module.**
Once this is solid, everything else sits on top of it.

---

## 2. Dynamic Module

The dynamic module turns static objects (states & fields) into actual **evolution**.

### 2.1 Single-Step Module

#### 2.1.1 `Kernel`

* **What it is:** The *bare* single-step rule, without global field influence.
* **Interfaces / responsibilities:**

  * Given a state `s` and a candidate `s'` (with `s' ∈ V[s]`), provide a field value that encodes the “raw tendency” to move `s → s'`.
  * Depends on core modules only (State, StateSpace, FieldValue, etc).
* **Interpretation:**

  * Think of it as a local transition rule or kernel ( K(s \to s') ).

---

#### 2.1.2 `[O] HookingBias`

* **What it is:** Optional component to inject biases or corrections based on global fields or other context.
* **Use cases:**

  * Incorporate momentum-like effects,
  * Apply environment-induced biases,
  * Represent hooks into other subsystems.

---

#### 2.1.3 `[O] KernelFieldComposition`

* **What it is:** Composition between the kernel field and a global field.
* **Purpose:**

  * To define how “bare” kernel and environment/global field combine.
* **Typical semantics:**

  * Pointwise multiplication, convolution, or any custom composition defined at the field level.

---

#### 2.1.4 `SingleStepField`

* **What it is:** The *full* single-step field used in evolution.
* **Inputs:**

  * `Kernel`,
  * `Reachable`,
  * optional `HookingBias` and `KernelFieldComposition`,
  * possibly global fields.
* **Responsibilities:**

  * For a given state `s`, and each `s' ∈ V[s]`, provide a `FieldValue` representing the **single-step contribution** towards `s'`.
  * Serve as the `F_t(s' : s)` object that plugs into the multi-step recurrence. 

This is the “one-step physics” of your system.

---

### 2.2 Multi-Step Module

#### 2.2.1 `MultiStepField`

* **What it is:** Multi-step field builder `F^ℓ` from `SingleStepField` + compositions + transforms.
* **Inputs:**

  * `Reachable` / `MultiStepReaching`,
  * `SingleStepField`,
  * `FieldComposition` (internal),
  * `FieldTransform` (for `Z^p`, `Z^t`, `Z^c`).
* **Responsibilities:**

  * Implement the generalized chain-rule recurrence (conceptually):

    * For each step `ℓ` and each new state `s'`:

      * accumulate contributions from all reaching states `s`,
      * apply transforms and compositions,
      * finalize the next-step field.
  * Provide access to:

    * multi-step fields,
    * optionally probability distributions derived from them.

At this point, you can already **evolve fields over time** for a single system.

---

## 3. Dynamic Systems Module

Now we wrap the pieces into **Dynamic System objects** that a user can actually run and analyze.

### 3.1 `StaticDynamicSystem` `[O]`

* **What it is:** A dynamic system defined purely in terms of state transitions *without* explicit fields.
* **Includes:**

  * `State`, `StateSpace`,
  * `Reachable`, `[O] Reaching`, `MultiStepReaching`.
* **Use:**

  * Useful when you want to reason about reachability and paths without committing to a field structure (e.g., for graph exploration).

---

### 3.2 `FieldStaticDynamicSystem` `[O]`

* **What it is:** A static dynamic system *with* fields but without multi-step recurrence.
* **Extends:**

  * `StaticDynamicSystem`,
* **Includes:**

  * `Field` and core field machinery.
* **Use:**

  * For systems where you want fields but either:

    * you manually handle evolution, or
    * you only care about single-step behaviour.

---

### 3.3 `FieldDynamicSystem`

* **What it is:** The full FDS dynamic system.
* **Extends:**

  * `FieldStaticDynamicSystem`.
* **Includes:**

  * `MultiStepField`,
  * an `Operator` used to extract observable evolution (e.g., next observed state, expectation trajectory).
* **Responsibilities:**

  * Provide a clean interface:

    * initialize from an initial state (and maybe initial global fields),
    * step / evolve for `ℓ` steps,
    * query paths, multi-step fields, and distributions.
* **Usage pattern:**

  * Most users will only interact with `FieldDynamicSystem` (or higher-level wrappers built on top of it).

---

## 4. Affecting Framework Module

The **Affecting Framework** lifts FDS from *single-system* to **interacting systems**, where one or more systems **affect** another.

We now go module-by-module.

---

### 4.1 Affecting State Module

This module defines how to represent groups of systems that influence each other.

#### 4.1.1 `AffectingGroupState` `[O]`

* **What it is:** Optional specialized `State` representing the **joint state** of an affecting group.
* **Example:**

  * A tuple of per-system states,
  * plus extra metadata (like shared conserved quantities).
* **Use:**

  * When you want a custom representation of group state beyond the default correlated state.

---

#### 4.1.2 `AffectingGroupStateSpace` `[O]`

* **What it is:** `StateSpace` for `AffectingGroupState`.
* **Extends:**

  * `StateSpace`,
* **Includes:**

  * `AffectingGroupState` as its atomic element type.
* **Use:**

  * When you want explicit control over the space of group states (e.g. restrictions, constraints).

---

#### 4.1.3 `CorrelatedState`

* **What it is:** A default **grouped state** representing the joint configuration of multiple systems.
* **Extends:**

  * `State`.
* **Internal representation (conceptual):**

  * A mapping `sys_id → state_of_that_system`.
* **Role:**

  * First layer to identify groups of affecting systems and their collective state, even if the user does not define a custom `AffectingGroupState`.

---

#### 4.1.4 `CorrelatedStateSpace`

* **What it is:** A default `StateSpace` over `CorrelatedState`.
* **Extends:**

  * `StateSpace`.
* **Use:**

  * As the canonical space of joint states for a collection of systems,
  * particularly for building default affecting groups without custom group-state design.

---

#### 4.1.5 `AffectedSystemsMapping`

* **What it is:** A `StateSpaceMapping` between `CorrelatedStateSpace` and `AffectingGroupStateSpace`.
* **Extends:**

  * `StateSpaceMapping`,
* **Includes:**

  * `CorrelatedStateSpace`.
* **Role:**

  * Bridges the *default* correlated representation and the **user-defined** group-state representation.
  * Tells how to interpret a correlated joint state as an element in the affecting group state space (and possibly vice versa).

---

### 4.2 Affecting Grouping Module

This module answers: *Which systems affect which others?* and *How do we group them?*

#### 4.2.1 `IsAffecting`

* **What it is:** A relation / predicate describing whether two dynamic systems affect each other.
* **Responsibilities:**

  * Given two `FieldDynamicSystem`s, decide if they are in an affecting relationship (and possibly in which direction).
* **Implementation options:**

  * Hard-coded rules,
  * Configuration-based (user declares relationships),
  * Derived from mappings or shared fields.

---

#### 4.2.2 `InterAffectingGroups`

* **What it is:** Grouping logic that forms **inter-affecting groups** of systems.
* **Includes:**

  * `IsAffecting`,
  * `AffectedSystemsMapping`.
* **Responsibilities:**

  * From a list/set of `FieldDynamicSystem`s, build:

    * groups of systems that form mutually or directionally affecting clusters,
    * associated correlated / group states.

This forms the **topological structure** over which affecting dynamics will operate.

---

### 4.3 Affecting Reachable Module

Now we lift the concepts of `Reachable`, `Reaching`, and `MultiStepReaching` to **groups of systems that affect each other**.

#### 4.3.1 `AffectedReachable`

* **What it is:** Group-level single-step reachability.
* **Extends:**

  * `Reachable`.
* **Includes:**

  * `AffectingGroupStateSpace`,
  * `AffectingGroupState`.
* **Responsibilities:**

  * Given a group state, compute the set of **next group states** reachable in one step, respecting:

    * the internal dynamics of each system,
    * the affecting relations between them.

---

#### 4.3.2 `AffectedReaching`

* **What it is:** Group-level inverse reachability.
* **Extends:**

  * `Reaching`.
* **Includes:**

  * `AffectingGroupStateSpace`,
  * `AffectingGroupState`.
* **Role:**

  * Given a group state, compute which previous group states can reach it in one step,
  * Useful for multi-step recurrences at the group level.

---

#### 4.3.3 `AffectingMultiStepReaching`

* **What it is:** Multi-step reachability for affecting groups.
* **Extends:**

  * `MultiStepReaching`.
* **Includes:**

  * `AffectedReachable`,
  * `AffectingGroupState`.
* **Use:**

  * Drives the multi-step evolution of affecting groups, exactly as `MultiStepReaching` does for single systems, but now at the group level.

---

### 4.4 Affecting Dynamics Module

These are the **dynamic counterparts** in the affecting setting.

#### 4.4.1 `AffectingSingleStep`

* **What it is:** Single-step field for an affecting group.
* **Extends:**

  * `SingleStepField`.
* **Role:**

  * Given a current group state (across multiple systems), build the **single-step joint field** that encodes how the whole group moves in one step.

---

#### 4.4.2 `AffectingMultiStepField`

* **What it is:** Multi-step field for affecting groups.
* **Extends:**

  * `MultiStepField`.
* **Role:**

  * Apply the generalized chain-rule recurrence to **group states** using `AffectingSingleStep` and `AffectingMultiStepReaching`.

---

#### 4.4.3 `AffectedSystemsOperator`

* **What it is:** Group-level evolution operator.
* **Extends:**

  * `Operator` (or the operator concept used in base `FieldDynamicSystem`).
* **Responsibilities:**

  * Given an affecting group (and its multi-step field), produce:

    * next observed group state,
    * trajectories for each member system,
    * or other observables defined at the group level.

---

### 4.5 Affecting Field Dynamic System

This is the **full object** for affected systems: the counterpart of `FieldDynamicSystem` for interacting systems.

#### 4.5.1 `AffectingFDS`

* **What it is:** A Field Dynamic System enriched with affecting structure.
* **Extends:**

  * `FieldDynamicSystem` (conceptually FDS),
* **Includes:**

  * `AffectingGroupState`,
  * `AffectingGroupStateSpace`,
  * `AffectedSystemsMapping`,
  * `AffectedReachable`,
  * `AffectedReaching`,
  * `AffectingMultiStepReaching`,
  * `AffectingSingleStep` / `AffectingMultiStepField`,
  * `AffectedSystemsOperator`.
* **Responsibilities:**

  * Provide a high-level interface to:

    * define groups of systems that affect each other,
    * evolve them jointly,
    * query group-level and system-level observables,
    * respect mapping constraints and correlations.

In practice, this is what you instantiate when you want to **simulate or analyze a network of interacting FDSs**.

---

### 4.6 Affecting Group Evolution Module

#### 4.6.1 `AffectingGroupsEvolution`

* **What it is:** A utility / orchestrator for evolving multiple groups of interacting systems.
* **Includes:**

  * `InterAffectingGroups`.
* **Responsibilities:**

  * Given a collection of `FieldDynamicSystem`s:

    * Identify affecting groups,
    * Build `AffectingFDS` instances for each group,
    * Evolve each group according to the desired schedule (synchronous, asynchronous, etc.),
    * Provide results back per system or per group.

This is the “top level” for affecting dynamics when working with *many* systems at once.

---

## 5. Practical “Checklist” for Implementing a New System (up to AffectingFDS)

For a **single system**:

1. Implement `State` and `StateSpace` for your system.
2. Implement `Reachable` (and optionally `Reaching`, `MultiStepReaching`).
3. Implement `SingleFieldValue`, `FieldValue`, `NormTransform`, `SingleFieldValueComposition`, `Field`, `FieldTransform`, `FieldComposition`.
4. Implement `Kernel` for single-step raw transitions.
5. Implement `SingleStepField` from `Kernel` (+ optional `HookingBias`, `KernelFieldComposition`).
6. Implement `MultiStepField` using your transforms and compositions.
7. Wrap it all in `FieldDynamicSystem` with a suitable `Operator`.

To extend into the **Affected Systems Framework**:

8. Decide if you need custom `AffectingGroupState` / `AffectingGroupStateSpace`; otherwise rely on `CorrelatedState` / `CorrelatedStateSpace`.
9. Implement / configure `AffectedSystemsMapping` between the correlated space and the group state space.
10. Implement `IsAffecting` for your system types.
11. Implement `InterAffectingGroups` to group systems.
12. Implement `AffectedReachable`, `[O] AffectedReaching`, and `AffectingMultiStepReaching`.
13. Implement `AffectingSingleStep`, `AffectingMultiStepField`, and `AffectedSystemsOperator`.
14. Implement `AffectingFDS` to tie everything together.
15. Optionally implement `AffectingGroupsEvolution` to handle many systems at once.

