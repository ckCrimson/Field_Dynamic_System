# Field Dynamic System (FDS) - JAX Backend

> **A Proof-of-Concept physics engine bridging high-level Object-Oriented design with blisteringly fast, bare-metal Data-Oriented execution via JAX XLA.**

![Quantum Interference Plot](image_c524e6.png)
*Above: A 100-step quantum random walk interference pattern (featuring double-slit environmental constraints and native wave collapse) executed in under 1 second on a standard CPU.*

## 📖 Overview

In scientific computing and complex system simulation, engineers are constantly forced to choose between readability and performance:
* **Object-Oriented Programming (OOP)** is excellent for human developers. It allows for clean domain modeling, flexible abstractions, and readable code. However, CPUs struggle with it due to memory fragmentation and pointer chasing.
* **Data-Oriented Design (DOD)** is what modern hardware craves. Flattened, contiguous arrays processed via SIMD instructions (like JAX or NumPy tensors) are orders of magnitude faster. Yet, writing pure tensor math to manage dynamic logic is incredibly difficult to scale and maintain.

**The Field Dynamic System (FDS)** was built to solve this friction. It is a framework designed to simulate entities interacting with complex spatial fields—from heat diffusion to quantum random walks. 

This specific implementation acts as a **Compiler Bridge**. It allows developers to define their universe using standard, readable Python objects (defining discrete topologies, custom kernels, and environmental masks). Then, using the `compile_bare_metal()` factory, the framework silently strips away the OOP abstractions and compiles the definitions into a pure, stateless mathematical closure. 

Using **JAX and XLA**, the entire field evolution, adjacency matrix multiplication, and entity observation sequence is JIT-compiled into highly optimized C++ / machine code, allowing for massive simulations to run in milliseconds without sacrificing developer experience.
