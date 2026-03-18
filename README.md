# Field Dynamic System (FDS) Framework

This repository contains the prototype implementation of the Field Dynamic System (FDS) framework.

## Project Status

**Status: Archived / Prototype**

This project was initially started to implement a framework for FDS to handle single-entity simulations and explore the core concepts. While the single-entity simulation was successfully built, architectural bottlenecks and design flaws became apparent, which would make scaling to multi-agent interactions highly complex and difficult to maintain. 

As a result, a separate, more flexible architecture has been designed elsewhere. This repository serves as a learning milestone and a working prototype for the initial design. It is no longer under active development.

## Overview

The framework provides an object-oriented and bare-metal JAX (Data-Oriented Design) execution approach to evolving entities within dynamic fields. It handles:
*   Static and Dynamic system topologies
*   Discrete and Continuous fields
*   State space management
*   Internal clock synchronization

## Demo Scenarios

You can run the demonstration scenarios located in the `examples` directory to see the single-entity simulation in action.

### Running the Demo

To run a basic discrete field simulation:

```bash
python examples/demo_discrete_field.py
```

### Key Components

*   `src/field_dynamic_system/systems/dynamic/field.py`: Contains the core logic for the dynamic field system, including both discrete and continuous implementations.
*   `src/field_dynamic_system/orchestration/`: Manages the execution loops and policies.
*   `src/field_dynamic_system/core/`: Defines the foundational state space models.

## Dependencies

This project uses Poetry for dependency management. Core dependencies include:
*   `jax`
*   `jaxlib`
*   `numpy`

To install dependencies:
```bash
poetry install
```