# Field Dynamic Systems (FDS) – Brief Theory Overview

## Abstract

A **Field Dynamic System (FDS)** is a general mathematical framework for modeling how systems evolve over time using **states**, **state spaces**, and **fields** (tensors with defined algebra).  
Instead of classical transition matrices or differential equations, FDS describes evolution using **field-weighted contributions**, **generalized chain-rule recurrence**, and **multi-step path accumulation**.

FDS also supports:

- **global fields** influencing evolution,  
- **mappings between state spaces**,  
- **correlation** between interacting systems (e.g., heat exchange),  
- **symmetry** between systems (structural equivalence of dynamics).

This document gives a concise theory overview using a simple **dice system** as an intuitive example.

---

# 1. States and State Spaces

## 1.1 State

A **state** \( s \) is any mathematical object describing the configuration of a system.

Examples:

- *Dice system:* state = histogram like \((n_1,n_2,\dots,n_6)\).  
- *Random walker:* state = integer position \(x\).  
- *Thermodynamic model:* state = temperature \(T\).

States must support hashing and equality.

---

## 1.2 State Space \( \mathbf{S} \)

The **state space** is:

$$
\mathbf{S} = \{\, s \mid s \text{ is an allowed state of the system} \,\}.
$$

Examples:

- Dice system: all histograms summing to \(N\).  
- Random walker: all integers.  
- Temperature: all allowed temperature values.

---

# 2. Reachability

## 2.1 Reachable Set \( \mathbf{V}[s_0] \)

For a given state \(s_0\), the **reachable set** is:

$$
\mathbf{V}[s_0] = \{\, s' \mid s_0 \to s' \text{ in one step} \,\}.
$$

*Dice example:*  
If \(s_0\) is a histogram, reachable states are histograms that increment one face.

---

## 2.2 Reaching Set \( \mathbf{R}[s'] \)

The **reaching set** (inverse reachable) is:

$$
\mathbf{R}[s'] = \{\, s \mid s' \in \mathbf{V}[s] \,\}.
$$

It lists all predecessors of a state.

---

## 2.3 Multi-step Reachability \( \mathbf{L}^l[s_0] \)

The set of all states reachable from \(s_0\) in exactly \(l\) steps is:

$$
\mathbf{L}^0[s_0] = \{s_0\},
$$

$$
\mathbf{L}^l[s_0] = \bigcup_{s \in \mathbf{L}^{l-1}[s_0]} \mathbf{V}[s].
$$

This drives the multi-step evolution frontier.

*Dice example:*  
\(\mathbf{L}^l\) contains all histograms after \(l\) rolls.

---

# 3. Fields

A **field** is a tensor-like value with:

- an addition rule,  
- a product-like composition,  
- a norm or contraction to real numbers.

Examples:

- real numbers,  
- probability amplitudes,  
- vectors,  
- energy/potential tensors.

---

## 3.1 Field Over a State Space

A field defined over the state space is a map:

$$
f : \mathbf{S} \rightarrow F,
$$

assigning a field value to each state.

---

## 3.2 Global Field \(F_g\)

A **global field** is another field over the same state space:

$$
f_g : \mathbf{S} \to F,
$$

representing external influences such as:

- environment,  
- global potentials,  
- coupled system effects.

---

# 4. Single-Step Field

The single-step field encodes the weighted contribution of moving from \(s_0\) to \(s\) in **one** step.

Definition:

$$
F^{1}(s : s_0)
=
\frac{F(s : s_0)\,F_g(s)}
     {\sum_{s_k \in \mathbf{V}[s_0]}
      F(s_k : s_0)\,F_g(s_k)}.
$$

Interpretation:

- \(F(s : s_0)\) = intrinsic transition weight  
- \(F_g(s)\) = influence from global field  
- denominator = normalization over reachable neighbours

*Dice example:*  
Intrinsic weight = number of ways to increment a face.  
Global field = temperature-like term that biases certain outcomes.

---

# 5. Multi-Step Field (Generalized Chain Rule)

Full evolution is not computed by enumerating all paths (which is exponential).  
Instead we use a **chain-rule-like recurrence**.

The multi-step field \(F^{l}(s_l : s_0)\) is:

$$
F^{l}(s_l : s_0)
=
\sum_{\pi : s_0 \to s_l}
\text{Contribution}(\pi),
$$

where each path’s contribution is built from:

- single-step fields \(F_t\),  
- transforms \(Z^p, Z^t, Z^c\),  
- internal composition \(\cdot\),  
- external accumulation \(+\).

This is implemented using reaching sets.

### General multi-step recurrence

**Accumulation step:**

$$
F'^{l}(s' : s_0)
=
F^{l}(s' : s_0)
+
Z^{t}\bigl(F_{t}(s' : s)\bigr)
\cdot
Z^{p}\bigl(F^{l-1}(s : s_0)\bigr)
$$

for every \(s \in \mathbf{R}[s']\).

**Finalize step:**

$$
F^{l}(s' : s_0)
=
Z^{c}\bigl(F'^{l}(s' : s_0)\bigr).
$$

This gives the exact multi-step evolution without enumerating paths.

---

# 6. Operator

An operator is a rule that extracts an **observable next state** from the field.

$$
\Theta : \mathbf{S} \rightarrow \mathbf{S}.
$$

Examples:

- **Probabilistic:** sample according to norm of \(F^l\).  
- **Deterministic:** choose the state with highest weight.  
- **Expectation-based:** compute expected value of an observable.

*Dice example:*  
Return the expected mean after each roll.

---

# 7. Relations Between State Spaces

Some systems live in different state spaces, but can still be related using **mappings**.

A mapping:

$$
M : \mathbf{S}_A \to \mathbf{S}_B
$$

allows:

- transferring fields,  
- comparing dynamics,  
- coarse-graining or refining,  
- analyzing multi-scale systems.

Field mapping is:

$$
F_B(M(s)) = T(F_A(s)),
$$

for some transform \(T\).

---

# 8. Correlation Between Systems

Two systems are **correlated** if the field of one **directly depends on the state of the other**, causing coupled evolution.

## Thermodynamic Heat Exchange Example

Let:

- system \(A\) have temperature state \(T_A\),  
- system \(B\) have temperature state \(T_B\).

Their fields influence each other:

$$
F_A(s_A \mid s_B)
=
F_A^{\text{internal}}(s_A)\,
G(T_B - T_A),
$$

$$
F_B(s_B \mid s_A)
=
F_B^{\text{internal}}(s_B)\,
G(T_A - T_B).
$$

Where \(G\) is a coupling term (e.g., proportional to temperature difference).

This interaction continues until equilibrium is reached.  
This is **correlation**, because their evolutions are *not independent*.

---

# 9. Symmetry Between Systems

Two systems are **symmetric** if under a mapping they evolve equivalently, even if they do **not** interact.

Types:

## 9.1 Path Evolution Symmetry

$$
F_A^{l}(s_A : s_{0A})
=
T\bigl(
F_B^{l}(M(s_A) : M(s_{0A}))
\bigr).
$$

## 9.2 Class-Based Symmetry

Equivalence classes in both state spaces correspond under mapping.

## 9.3 Expectation Evolution Symmetry (EeS)

Expectations of observables match under mapping.

*Dice example:*  
The dice histogram system may be symmetric to a deterministic mean-evolution system at the expectation level.

---

# 10. Summary Table

| Concept | Meaning |
|--------|---------|
| **State** \(s\) | configuration of the system |
| **State space** \(\mathbf{S}\) | all allowed states |
| **Reachable / Reaching** | neighbourhood structure |
| **Multi-step reachability** | evolution frontier |
| **Field** | tensor with algebra |
| **Field over state space** | mapping \(f:\mathbf{S}\to F\) |
| **Global field** | external influence |
| **Single-step field** | normalized one-step dynamics |
| **Multi-step field** | chain-rule recurrence |
| **Operator** | extracts observed next state |
| **Mappings** | relate different state spaces |
| **Correlation** | interacting systems |
| **Symmetry** | structurally equivalent systems |
