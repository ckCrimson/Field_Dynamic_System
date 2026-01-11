# Core State Module

The **Core State Module** is the foundational layer of the Field Dynamic System. It defines "what exists" in the simulation.

It relies on a **Hybrid Architecture** that strictly separates **Storage** (Python Objects) from **Computation** (JAX Arrays) to achieve high-performance simulations.

## 📂 System Architecture

The module is organized into 6 key files based on responsibility:

| File | Responsibility | Key Classes |
| :--- | :--- | :--- |
| **`interfaces.py`** | Base Contracts & Protocols | `State`, `StateSpace`, `IDiscreteStateSpace`, `IContinuousStateSpace`, `IStateOperation` |
| **`State.py`** | Atomic Data Objects | `VectorState` (Points), `AbstractState` (Symbols) |
| **`encoding.py`** | Object $\leftrightarrow$ Array Translation | `VectorEncoding` (Tuples to Arrays), `BitMaskingEncoding` (Symbols to IDs) |
| **`discrete.py`** | Finite Sets Logic | `VectorStateSpace` (JAX Matrix), `IndexedVectorStateSpace` (O(1) Search), `AbstractDiscreteStateSpace` (Sets of Symbols) |
| **`continuous.py`** | Infinite Manifolds | `HypercubeSpace`, `HypersphereSpace`, `CompositeSpace` (CSG) |
| **`transformation.py`** | Space Modifiers | `DiscreteStateTransformation`, `ContinuousStateTransformation`, `VectorStateTransformation` |

---

## 📊 Class Diagram

The following UML diagram illustrates the inheritance hierarchy and relationships between the core components.

```mermaid
classDiagram
    %% ==========================================
    %% 1. The Core Interfaces & Base Classes
    %% ==========================================
    class StateSpace {
        <<Interface>>
        +contains(state)
        +intersection(other)
        +union(other)
    }

    class IDiscreteStateSpace {
        <<Abstract>>
        +encoder: StateEncoder
        +get_matrix()
        +map(operation)
    }

    class IContinuousStateSpace {
        <<Abstract>>
        +project(raw_data)
    }

    class State {
        <<Abstract>>
    }

    class StateEncoder {
        <<Abstract>>
        +shape: tuple
        +encode(data)
        +decode(encoded_data)
    }

    %% ==========================================
    %% 2. The Discrete State Branch
    %% ==========================================
    class VectorStateSpace {
        +allowed_vectors: tuple
        +dim: int
        +get_matrix()
    }
    
    class IndexedVectorStateSpace {
        +indexed_axes: tuple
        +search_by_index()
    }

    class AbstractDiscreteStateSpace {
        +allowed_states: set
    }

    class VectorState {
        +values: tuple
    }

    class AbstractState {
        +name: str
        +properties: dict
    }

    %% ==========================================
    %% 3. The Continuous State Branch
    %% ==========================================
    class ContinuousStateSpace {
        <<Abstract>>
    }
    
    class HypercubeSpace {
        +low: ndarray
        +high: ndarray
    }

    class HypersphereSpace {
        +center: ndarray
        +radius: float
    }
    
    class CompositeSpace {
        +a: StateSpace
        +b: StateSpace
        +op: CSGOp
    }

    %% ==========================================
    %% RELATIONSHIPS
    %% ==========================================
    
    StateSpace <|-- IDiscreteStateSpace
    StateSpace <|-- IContinuousStateSpace
    
    IDiscreteStateSpace <|-- VectorStateSpace
    VectorStateSpace <|-- IndexedVectorStateSpace
    IDiscreteStateSpace <|-- AbstractDiscreteStateSpace
    
    IContinuousStateSpace <|-- ContinuousStateSpace
    ContinuousStateSpace <|-- HypercubeSpace
    ContinuousStateSpace <|-- HypersphereSpace
    ContinuousStateSpace <|-- CompositeSpace

    State <|-- VectorState
    State <|-- AbstractState