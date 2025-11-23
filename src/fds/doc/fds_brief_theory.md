# Field Dynamic Systems (FDS) – Brief Theory Overview

## Abstract

A **Field Dynamic System (FDS)** is a general mathematical framework for modeling how systems evolve over time using:

- **states** and **state spaces**,  
- **fields** (tensor-like objects with addition, product, and norm),  
- a **generalized chain-rule style recurrence** for multi-step evolution.

Instead of relying only on transition matrices or differential equations, FDS expresses evolution through **field-weighted contributions** over paths, using reachability structure and multi-step fields.

FDS also supports:

- **global fields** that bias evolution,
- **mappings between state spaces**,
- **correlation** between interacting systems (e.g., heat exchange),
- **symmetry** between systems (structural equivalence of dynamics).

A simple **dice system** (states as histograms of face counts) is used as a running mental model.

---

# 1. States and State Spaces

## 1.1 State

A **state** \( s \) is any mathematical object that describes the configuration of a system at an instant.

Examples:

- Dice system: a histogram  (n_1,n_2,\dots,n_6) \) of face counts.  
- Random walker: an integer position \( x \).  
- Thermodynamic toy model: a scalar temperature \( T \).

States should at least support equality and hashing.

---

## 1.2 State Space **S**

The **state space** is the set of all allowed states:

$$ \mathbf{S} = \{ s \mid s \text{ is an allowed state of the system} \}. $$

Examples:

- Dice: all histograms whose entries sum to total number of rolls \(N\).  
- Walker: all integers (or a bounded subset).  
- Temperature: all admissible temperature values.

---

# 2. Reachability

## 2.1 Reachable Set **V**[s_0]

The **reachable set** from a state \( s_0 \) is:

$$ \mathbf{V}[s_0] = \{ s' \mid s_0 \to s' \text{ in one step} \}. $$

Dice example: if \( s_0 \) is a current histogram, then \( \mathbf{V}[s_0] \) consists of histograms where exactly one face count is incremented by 1.

---

## 2.2 Reaching Set **R[s']**

The **reaching set** (inverse reachable) is:

$$ \mathbf{R}[s'] = \{ s \mid s' \in \mathbf{V}[s] \}. $$

This is the set of all predecessor states that can reach \( s' \) in one step.

---

## 2.3 Multi-step Reachability $ L^l[s_0] $

The set of states reachable from \( s_0 \) in exactly \( l \) steps is defined recursively:

$$ \mathbf{L}^0[s_0] = \{ s_0 \}, $$

$$ \mathbf{L}^l[s_0] = \bigcup_{s \in \mathbf{L}^{l-1}[s_0]} \mathbf{V}[s]. $$

This sequence \( \mathbf{L}^0, \mathbf{L}^1, \dots \) describes the **evolution frontier** in terms of reachability alone.

Dice example: \( \mathbf{L}^l[s_0] \) contains all histograms after exactly \( l \) rolls starting from an initial configuration \( s_0 \).

---

# 3. Fields

A **field** is a value from some space \( F \) that comes with:

- an **addition** operation,
- a **product / composition** operation,
- a **norm** or contraction to a real number (used for probabilities or weights).

Examples of \( F \):

- real numbers,  
- complex amplitudes,  
- vectors or tensors,  
- energy-like or potential-like quantities.

---

## 3.1 Field Over a State Space

A field over the state space is a map:

$$ f : \mathbf{S} \rightarrow F, $$

which assigns a field value \( f(s) \) to each state \( s \in \mathbf{S} \).

---

## 3.2 Global Field \( f_g \)

A **global field** is another field on the state space:

$$ f_g : \mathbf{S} \rightarrow F, $$

representing external influences, such as:

- an environment,
- a potential landscape,
- effects of other systems.

It modifies how transitions or states are weighted during evolution.

---

# 4. Single-Step Field

The **single-step field** encodes the weighted contribution of moving from \( s_0 \) to \( s \) in one step, including global influence.

A common form is:

$$ F^{1}(s : s_0) = \frac{F(s : s_0)\,F_g(s)}{\sum_{s_k \in \mathbf{V}[s_0]} F(s_k : s_0)\,F_g(s_k)}. $$

Interpretation:

- \( F(s : s_0) \): intrinsic single-step field (local transition weight).  
- \( F_g(s) \): global field bias at state \( s \).  
- Denominator: normalizes over all reachable neighbours \( s_k \in \mathbf{V}[s_0] \).

Dice example:  
\( F(s : s_0) \) could count the number of micro-configurations leading from histogram \( s_0 \) to \( s \).  
\( F_g(s) \) could bias certain aggregate outcomes (e.g., favoring some distributions).

---

# 5. Multi-Step Field (Generalized Chain Rule)

Rather than explicitly summing over all possible paths (which is exponential in length), FDS uses a **generalized chain-rule style recurrence**.

Conceptually, the multi-step field is:

$$ F^{l}(s_l : s_0) = \sum_{\pi : s_0 \to s_l} \text{Contribution}(\pi), $$

where each path \( \pi = (s_0,s_1,\dots,s_l) \) contributes a value built from:

- single-step field values,
- transforms \( Z^p, Z^t, Z^c \),
- internal composition \( \cdot \),
- external accumulation \( + \).

In practice, we implement this using **reaching sets** instead of enumerating all paths.

---

## 5.1 Multi-Step Recurrence (GitHub-safe form)

For each step \( l \) and each state \( s' \), we update the multi-step field using reaching states \( s \in \mathbf{R}[s'] \).

**Accumulation step:**

$$ F'^{l}(s' : s_0) = F^{l}(s' : s_0) + Z^{t}\!\big(F_t(s' : s)\big) \cdot Z^{p}\!\big(F^{l-1}(s : s_0)\big) $$

where:

- \( F_t(s' : s) \) is the single-step field,
- \( Z^p \) prepares the previous-step contribution,
- \( Z^t \) prepares the single-step contribution,
- \( \cdot \) is internal composition,
- \( + \) accumulates contributions.

**Finalize step:**

$$ F^{l}(s' : s_0) = Z^{c}\!\big(F'^{l}(s' : s_0)\big). $$

Here \( Z^c \) can normalize, threshold, or otherwise clean up the accumulated field.

This recurrence **is** the multi-step evolution rule in FDS.

---

# 6. Operator

An **operator** extracts an observable next state (or distribution) from the field:

$$ \Theta : \mathbf{S} \rightarrow \mathbf{S}. $$

Examples:

- **Probabilistic operator:** sample the next state according to the norm of \( F^{l}(\cdot : s_0) \).  
- **Deterministic operator:** choose the state \( s' \) with maximum weight or some other criterion.  
- **Expectation-based operator:** compute expected values of observables with respect to the induced distribution.

Dice example:  
Given a multi-step field over histograms, \( \Theta \) might return the expected sample mean or the most likely histogram.

---

# 7. Relations Between State Spaces

In many situations, two systems live in **different state spaces** but are related through a mapping:

$$ M : \mathbf{S}_A \to \mathbf{S}_B. $$

This allows us to:

- map states from one system to another,
- push fields forward along \( M \),
- compare dynamics across models (fine vs coarse, micro vs macro).

A typical field mapping follows:

$$ F_B(M(s)) = T\!\big(F_A(s)\big), $$

where \( T \) is a suitable transform (e.g., aggregation, normalization).

---

# 8. Correlation Between Systems

Two systems are **correlated** when the field of one depends on the state of the other, causing coupled evolution.

## 8.1 Thermodynamic Heat Exchange Example

Consider two systems:

- System \( A \) with temperature-like state \( T_A \),
- System \( B \) with temperature-like state \( T_B \).

Define:

$$ F_A(s_A \mid s_B) = F_A^{\text{internal}}(s_A)\,G(T_B - T_A), $$

$$ F_B(s_B \mid s_A) = F_B^{\text{internal}}(s_B)\,G(T_A - T_B), $$

where \( G \) is a coupling function (often proportional to the temperature difference).

As the systems evolve, they exchange “heat” and gradually approach equilibrium.

This is **correlation**: the evolution of \( A \) and \( B \) is no longer independent; their fields explicitly depend on each other’s states.

---

# 9. Symmetry Between Systems

Two systems are **symmetric** if under some mapping their dynamic behavior matches in a precise sense, even if they do not directly interact.

## 9.1 Path Evolution Symmetry

Given a mapping \( M : \mathbf{S}_A \to \mathbf{S}_B \) and a transform \( T \) on fields, we say there is path evolution symmetry if:

$$ F_A^{l}(s_A : s_{0A}) = T\!\big(F_B^{l}(M(s_A) : M(s_{0A}))\big). $$

That is, the multi-step fields correspond under the mapping \(M\) and transform \(T\).

---

## 9.2 Class-Based Symmetry

Often we group states into **equivalence classes** based on conserved quantities or aggregate features.  

If there is a mapping between these classes in \( \mathbf{S}_A \) and \( \mathbf{S}_B \) such that class-level fields correspond, we have **class-based symmetry**.

---

## 9.3 Expectation Evolution Symmetry (EeS)

Even if full path or class-level symmetry fails, **expectation evolution symmetry** may hold: expectations of certain observables match under mapping.

Dice example:  
The full dice histogram system may be complex, but its evolution of the **mean** may match that of a much simpler deterministic system.

---

# 10. Summary Table

| Concept | Meaning |
|--------|---------|
| State \(s\) | a configuration of the system |
| State space \(\mathbf{S}\) | set of all allowed states |
| \(\mathbf{V}[s_0]\) | states reachable from \(s_0\) in one step |
| \(\mathbf{R}[s']\) | predecessors that can reach \(s'\) |
| \(\mathbf{L}^l[s_0]\) | states reachable from \(s_0\) in \(l\) steps |
| Field \(F\) | tensor-like value with algebra |
| Field over state space | map \(f:\mathbf{S}\to F\) |
| Global field \(f_g\) | external influence on evolution |
| Single-step field \(F^{1}\) | normalized one-step contribution |
| Multi-step field \(F^{l}\) | built via generalized chain rule |
| Operator \(\Theta\) | extracts an observable next state |
| Mapping \(M\) | relates different state spaces |
| Correlation | coupled evolution due to field dependence |
| Symmetry | structural equivalence of dynamics under mapping |
