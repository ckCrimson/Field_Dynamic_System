from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable, Callable, Any, Type
from dataclasses import dataclass
import jax.numpy as jnp
import numpy as np

from src.field_dynamic_system.core.state.interfaces import StateSpace, IStateOperation, IDiscreteStateSpace, IStateSpaceTransformation
from src.field_dynamic_system.core.state.discrete import VectorStateSpace, AbstractDiscreteStateSpace
from src.field_dynamic_system.core.state.continous import ContinuousStateSpace, HypersphereSpace, HypercubeSpace
from src.field_dynamic_system.core.state import VectorState, AbstractState


# --- Level 1: The Interface ---



# --- Level 2: The Abstract Branches ---

@dataclass
class DiscreteStateTransformation(IStateSpaceTransformation):
    """
    Transforms any Discrete Space into another specific Discrete Space type.
    """
    operation: IStateOperation
    target_class: Type[IDiscreteStateSpace]  # User specifies: "I want a VectorStateSpace back"

    def transform(self, space: StateSpace) -> IDiscreteStateSpace:
        if not isinstance(space, IDiscreteStateSpace):
            raise TypeError(f"Transformation input must be DiscreteStateSpace, got {type(space)}")

        # DECISION: JAX Batching vs. Python Loop
        # If we are targeting an Abstract Space, the operation likely returns Strings/Objects.
        # JAX cannot handle strings. We must use a Python loop.
        use_python_loop = issubclass(self.target_class, AbstractDiscreteStateSpace)

        if use_python_loop:
            # Manual Iteration (Safe for Strings)
            if isinstance(space, VectorStateSpace):
                # Iterate over vector objects
                raw_results = [self.operation(v) for v in space.allowed_states]
            else:
                # Abstract -> Abstract
                raw_results = [self.operation(s) for s in space.allowed_states]
        else:
            # JAX Batching (Fast for Math)
            # space.map() usually passes the JAX array/batch
            raw_results = space.map(self.operation)

        # 2. Build the new space
        return self._build_space(raw_results)

    def _build_space(self, raw_data) -> IDiscreteStateSpace:

        # --- CASE A: Target is VectorStateSpace ---
        if issubclass(self.target_class, VectorStateSpace):
            vectors = []
            dim = 0

            if len(raw_data) == 0:
                return self.target_class([], dim=0)

            for item in raw_data:
                # FIX 1: Ensure hashability by forcing Tuple conversion
                if hasattr(item, 'values'):
                    # If it's already a VectorState, extract values as tuple
                    # (Assuming VectorState.values might be an array)
                    vec_tuple = tuple(item.values)
                elif isinstance(item, (np.ndarray, jnp.ndarray)):
                    # Convert JAX/Numpy array to standard float tuple
                    vec_tuple = tuple(item.tolist())
                else:
                    vec_tuple = tuple(item)

                # Create the clean, hashable VectorState
                vec = VectorState(vec_tuple)
                vectors.append(vec)

                if dim == 0: dim = len(vec_tuple)

            return self.target_class(vectors, dim=dim)

        # --- CASE B: Target is AbstractDiscreteStateSpace ---
        elif issubclass(self.target_class, AbstractDiscreteStateSpace):
            states = set()
            for item in raw_data:
                if isinstance(item, AbstractState):
                    states.add(item)
                else:
                    # FIX 2: Provide empty dict for 'properties'
                    states.add(AbstractState(name=str(item), properties={}))

            return self.target_class(states)

class ContinuousStateTransformation(IStateSpaceTransformation, ABC):
    """
    Base for transformations that operate on Manifold Parameters (Shift/Scale).
    """
    pass


# --- Level 3: Concrete Implementations ---

@dataclass
class VectorStateTransformation(DiscreteStateTransformation):
    """
    Transforms a VectorStateSpace by applying a JAX-accelerated function
    to every vector in the set.
    """
    operation: IStateOperation  # The math (e.g., Polar -> Cartesian)

    def transform(self, space: StateSpace) -> VectorStateSpace:
        # Type Check
        if not isinstance(space, VectorStateSpace):
            raise TypeError(f"VectorStateTransformation expects VectorStateSpace, got {type(space)}")

        # 1. High-Performance Batch Map
        # Input: (N, Old_Dim) -> Output: (N, New_Dim)
        new_matrix = space.map(self.operation)

        # 2. Reconstruct Space
        new_dim = new_matrix.shape[-1]

        # We need to convert JAX rows back to VectorState objects for the Set
        # (This iteration is unavoidable but happens only once during setup)
        new_vectors = [VectorState(tuple(v)) for v in new_matrix]

        return VectorStateSpace(new_vectors, dim=new_dim)


@dataclass
class AbstractStateTransformation(DiscreteStateTransformation):
    """
    Transforms an AbstractDiscreteStateSpace (e.g., renaming states).
    """
    operation: Callable[[AbstractState], AbstractState]

    def transform(self, space: StateSpace) -> AbstractDiscreteStateSpace:
        if not isinstance(space, AbstractDiscreteStateSpace):
            raise TypeError(f"AbstractStateTransformation expects AbstractDiscreteStateSpace, got {type(space)}")

        # 1. Apply mapping logic (Standard Python Loop)
        new_states_list = space.map(self.operation)

        # 2. Reconstruct Space
        return AbstractDiscreteStateSpace(set(new_states_list))


@dataclass
class ParameterContinuousTransformation(ContinuousStateTransformation):
    """
    Standard implementation for shifting/scaling Continuous Spaces.
    """
    translation: jnp.ndarray = None  # Shift vector
    scale: float = 1.0  # Scale factor

    def transform(self, space: StateSpace) -> ContinuousStateSpace:
        if not isinstance(space, ContinuousStateSpace):
            raise TypeError("Expected ContinuousStateSpace")

        # Logic for Hypersphere
        if isinstance(space, HypersphereSpace):
            new_center = space.center
            new_radius = space.radius * self.scale

            if self.translation is not None:
                new_center = new_center + self.translation

            return HypersphereSpace(new_center, new_radius, space._encoding)

        # Logic for Hypercube
        elif isinstance(space, HypercubeSpace):
            new_low = space.low * self.scale
            new_high = space.high * self.scale

            if self.translation is not None:
                new_low = new_low + self.translation
                new_high = new_high + self.translation

            return HypercubeSpace(new_low, new_high, space._encoding)

        raise NotImplementedError(f"Transformation not supported for {type(space)}")