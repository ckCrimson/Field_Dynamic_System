# Field Dynamic Systems (FDS) — Brief Theory Overview

## Abstract

A **Field Dynamic System (FDS)** is a general mathematical framework for modeling system evolution using:

- **states** and **state spaces**,  
- **fields** (tensor-like weights with addition, product, and norm),  
- a **generalized chain-rule recurrence**,  
- **multi-step path accumulation** without enumerating paths.

FDS also provides:

- **global fields** that influence evolution,  
- **mappings between state spaces**,  
- **correlation** between interacting systems (e.g., heat exchange),  
- **symmetry** between systems (structural equivalence).

A simple *dice system* is used throughout for intuition.

---

# 1. States and State Spaces

## 1.1 State

A **state** \(s\) is any mathematical object representing a configuration of a system.

Examples:

- Dice system: histogram \((n_1,n_2,\dots,n_6)\)  
- Random walker: integer position \(x\)  
- Thermodynamic model: temperature value \(T\)

States must support equality and hashing.

---

## 1.2 State Space \(\mathbf{S}\)

The **state space** is:

\[
\mathbf{S} = \{ s \mid s \text{ is an allowed system state} \}
\]

Examples:

- All histograms summing to \(N\) (dice)  
- All integers (walker)  
- All allowed temperatures (thermo model)

---

# 2. Reachability

## 2.1 Reachable Set \(\mathbf{V}[s_0]\)

\[
\mathbf{V}[s_0] = \{\, s' \mid s_0 \to s' \text{ in one step} \,\}
\]

Example (dice): increment one face of the histogram.

---

## 2.2 Reaching Set \(\mathbf{R}[s']\)

\[
\mathbf{R}[s'] = \{\, s \mid s' \in \mathbf{V}[s] \,\}
\]

All predecessors of state \(s'\).

---

## 2.3 Multi-Step Reachability \(\mathbf{L}^l[s_0]\)

\[
\mathbf{L}^0[s_0] = \{s_0\}
\]

\[
\mathbf{L}^l[s_0] = \bigcup_{s \in \mathbf{L}^{l-1}[s_0]} \mathbf{V}[s]
\]

Defines the frontier of reachable states after \(l\) steps.

---

# 3. Fields

A **field** is a tensor-like value equipped with:

- an **addition** rule  
- a **composition/product** rule  
- a **norm** (contraction to a real number)

Examples:

- real numbers  
- complex amplitudes  
- vectors  
- potential tensors  

---

## 3.1 Field Over a State Space

A field over the state space is:

\[
f : \mathbf{S} \rightarrow F
\]

assigning a field value to each state.

---

## 3.2 Global Field \(f_g\)

A **global field** is another field on \(\mathbf{S}\):

\[
f_g : \mathbf{S} \rightarrow F
\]

representing environmental or external influences (e.g., temperature bias).

---

# 4. Single-Step Field

The single-step field combines intrinsic transition weight and global influence.

\[
F^{1}(s : s_0)
=
\frac{
F(s : s_0)\,F_g(s)
}{
\sum_{s_k \in \mathbf{V}[s_0]} F(s_k : s_0)\,F_g(s_k)
}
\]

Interpretation:

- \(F(s:s_0)\): intrinsic transition weight  
- \(F_g(s)\): effect of global field  
- denominator: normalization over neighbours  

Example (dice):  
Intrinsic weight = ways to increment a face.  
Global field = temperature-like bias for outcomes.

---

# 5. Multi-Step Field (Generalized Chain Rule)

Directly summing over all possible paths is exponential.  
Instead, FDS uses a **generalized chain-rule recurrence**.

Conceptually:

\[
F^{l}(s_l : s_0)
=
\sum_{\pi : s_0 \rightarrow s_l}
\text{Contribution}(\pi)
\]

Each path contribution uses:

- single-step field \(F_t\)  
- transforms \(Z^p, Z^t, Z^c\)  
- internal composition \(\cdot\)  
- external accumulation \(+\)

Implemented using reaching sets.

---

## 5.1 General Multi-Step Recurrence (GitHub-Safe)

### Accumulation

\[
F'^{l}(s' : s_0)
=
F^{l}(s' : s_0)
+
Z^{t}\!\bigl(F_{t}(s' : s)\bigr)
\cdot
Z^{p}\!\bigl(F^{l-1}(s : s_0)\bigr)
\]

for every \(s \in \mathbf{R}[s']\).

### Finalize

\[
F^{l}(s' : s_0)
=
Z^{c}\!\bigl(F'^{l}(s' : s_0)\bigr)
\]

This gives the exact multi-step field without enumerating paths.

---

# 6. Operator

An operator extracts an observable state from the multi-step field:

\[
\Theta : \mathbf{S} \rightarrow \mathbf{S}
\]

Examples:

- probabilistic sampling (according to field norm)  
- deterministic max-weight selection  
- expectation-based transition  

Dice example: return the expected mean after each roll.

---

# 7. Mappings Between State Spaces

A mapping between two state spaces:

\[
M : \mathbf{S}_A \rightarrow \mathbf{S}_B
\]

enables:

- translating states  
- transferring fields  
- comparing dynamics  
- coarse-graining / refinement  

Field mapping follows:

\[
F_B(M(s)) = T\!\left(F_A(s)\right)
\]

for some transform \(T\).

---

# 8. Correlation Between Systems

Two systems are **correlated** when each system’s field depends on the other’s state.

### Example: Heat Exchange Between Two Bodies

System \(A\) with temperature \(T_A\):  

\[
F_A(s_A \mid s_B)
=
F_A^{\text{internal}}(s_A)\,
G(T_B - T_A)
\]

System \(B\) with temperature \(T_B\):  

\[
F_B(s_B \mid s_A)
=
F_B^{\text{internal}}(s_B)\,
G(T_A - T_B)
\]

Where \(G\) is a coupling term (e.g., proportional to temperature difference).

The systems evolve until equilibrium.

This is **correlation**: the dynamics are *not independent*.

---

# 9. Symmetry Between Systems

Two systems are **symmetric** if their dynamics match under a mapping.

## 9.1 Path Evolution Symmetry

\[
F_A^{l}(s_A : s_{0A})
=
T\!\left(
F_B^{l}(M(s_A) : M(s_{0A}))
\right)
\]

## 9.2 Class-Based Symmetry

Mapping corresponds equivalence classes between state spaces.

## 9.3 Expectation Evolution Symmetry (EeS)

Expectations of observables evolve identically under mapping.

Example:  
Dice histogram system symmetric to a deterministic mean-evolution model at expectation level.

---

# 10. Summary Table

| Concept | Meaning |
|--------|---------|
| State \(s\) | configuration of the system |
| State space \(\mathbf{S}\) | all allowed states |
| Reachable / Reaching | one-step successors / predecessors |
| Multi-step reachability | reachable after \(l\) steps |
| Field | tensor with addition & product |
| Field over state space | \(f:\mathbf{S}\to F\) |
| Global field | external influence |
| Single-step field | normalized one-step rule |
| Multi-step field | generalized chain-rule recurrence |
| Operator | extracts observed next state |
| Mapping | relates different state spaces |
| Correlation | interacting systems |
| Symmetry | structurally equivalent dynamics |

