from typing import Callable, Type, Any, List, Union
import jax.numpy as jnp
import numpy as np

from .interfaces import IDiscreteStateSpace, State
from .discrete import VectorStateSpace, AbstractDiscreteStateSpace, VectorState


class DiscreteStateTransformation:
    """Base class for transformations."""

    def __init__(self,
                 operation: Callable[[State], Any],
                 target_space_class: Type[IDiscreteStateSpace]):
        self.operation = operation
        self.target_space_class = target_space_class

    def transform(self, space: IDiscreteStateSpace) -> IDiscreteStateSpace:
        # Fallback implementation
        raw_results = [self.operation(s) for s in space.states]
        return self.target_space_class(raw_results)


class AbstractStateTransformation(DiscreteStateTransformation):
    """
    Specialized for Abstract States (Strings/Dicts).
    """

    def transform(self, space: IDiscreteStateSpace) -> IDiscreteStateSpace:
        # Use the generic map (Python Loop)
        raw_results = space.map(self.operation)
        return self.target_space_class(raw_results)


class VectorStateTransformation(DiscreteStateTransformation):
    """
    High-Performance Transformation for Vectors.
    Leverages JAX vmap + Raw Factory for 1000x speedups.
    """

    def __init__(self,
                 operation: Callable[[Any], Any],
                 target_space_class: Type[IDiscreteStateSpace] = VectorStateSpace):
        # Pass correctly to parent
        super().__init__(operation, target_space_class)

    def transform(self, space: VectorStateSpace) -> VectorStateSpace:
        """
        Executes transformation entirely in JAX/Numpy memory where possible.
        """
        # 1. Run the Math (Fast JAX Map)
        raw_matrix = space.map(self.operation)

        # 2. Check if optimization succeeded (returned Array vs List)
        if isinstance(raw_matrix, list) or raw_matrix.dtype == object:
            return self.target_space_class(raw_matrix)

        # 3. FAST PATH: Create Space from Raw Data
        dim = raw_matrix.shape[-1]

        # Use the factory if available
        if hasattr(self.target_space_class, 'from_raw_data'):
            return self.target_space_class.from_raw_data(
                raw_matrix,
                wrapper=lambda x: VectorState(tuple(x.tolist())),
                dim=dim
            )

        # Fallback
        return self.target_space_class.from_raw_array(np.array(raw_matrix))