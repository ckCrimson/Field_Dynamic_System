# Field Dynamic System (FDS) - JAX Backend

> **A Proof-of-Concept physics engine that compiles high-level Object-Oriented simulations into high-performance JAX closures.**

![Quantum Interference Plot](image_c524e6.png)
*Above: A 900-step quantum random walk interference pattern (featuring double-slit environmental constraints and native wave collapse) executed in under 1 second.*

## 📖 Overview

**The Field Dynamic System (FDS)** is a framework designed to simulate entities interacting with complex spatial fields, ranging from simple diffusion to quantum random walks. 

Writing raw JAX arrays and managing index math for dynamic spatial systems is complex and difficult to scale. Conversely, using standard Python objects to define these systems is easy for developers, but far too slow for large-scale simulations.

This project was built to bridge that gap. It allows developers to define a universe using standard Python objects—defining topologies like `Grid8Topology`, custom kernels, and environmental masks. 

Once the environment is defined, the `compile_bare_metal()` factory strips away the OOP abstractions and compiles the rules into a pure, stateless mathematical closure. Powered by **JAX and XLA**, the entire field evolution, sparse matrix multiplication, and entity observation sequence is JIT-compiled. This allows complex simulations to run in milliseconds while keeping the user-facing API clean and declarative.
