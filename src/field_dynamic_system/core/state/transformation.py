from typing import Callable, Any, Type, Sequence, Optional, Union
import jax.numpy as jnp
import numpy as np

# Adjust imports to match your project structure
from src.field_dynamic_system.core.state.interfaces import IDiscreteStateSpace, IStateSpaceTransformation
from src.field_dynamic_system.core.state.discrete import VectorStateSpace, VectorState, AbstractDiscreteStateSpace


# --- 1. BASE CLASS ---
class DiscreteStateTransformation(IStateSpaceTransformation):
    """
    Base class for state transformations.
    """

    def __init__(self,
                 operation: Callable[[Any], Any],
                 target_space_class: Type[IDiscreteStateSpace],
                 raw_operation: Optional[Callable[[Any], Any]] = None):
        """
        Args:
            operation: Function to apply to State Objects.
            target_space_class: The class of the output space.
            raw_operation: Optional function to apply to Raw Data directly.
                           If None, defaults to 'operation'.
        """
        self.operation = operation
        self.target_space_class = target_space_class
        # Backward Compatibility: Default to operation if raw is missing
        self.raw_operation = raw_operation if raw_operation else operation

    def transform(self, space: IDiscreteStateSpace) -> IDiscreteStateSpace:
        # Fallback implementation
        raw_results = [self.operation(s) for s in space.states]
        return self.target_space_class(raw_results)


# --- 2. ABSTRACT TRANSFORMATION (Strings, Tuples, Dicts) ---
class AbstractStateTransformation(DiscreteStateTransformation):
    """
    Specialized for Abstract States (Strings/Dicts).
    """

    def transform(self, space: IDiscreteStateSpace) -> IDiscreteStateSpace:
        # 1. FAST PATH: Check for raw states availability
        # OPTIMIZATION: Try to grab the list directly from the proxy to avoid Numpy copy overhead
        raw_source = None
        if hasattr(space, '_idx_to_state') and hasattr(space._idx_to_state, 'raw_data'):
            raw_source = space._idx_to_state.raw_data
        elif hasattr(space, 'raw_states'):
            raw_source = space.raw_states

        if raw_source is not None:
            # Use the raw_operation on the raw buffer
            raw_output = self.transform_raw(raw_source)

            # Reconstruct using Factory
            # Try to infer wrapper from the space
            wrapper_cls = getattr(space, '_get_wrapper_cls', lambda: None)()
            if not wrapper_cls and space.states:
                wrapper_cls = type(space.states[0])

            if hasattr(self.target_space_class, 'from_raw_data'):
                return self.target_space_class.from_raw_data(
                    raw_data=raw_output,
                    wrapper=wrapper_cls
                )

        # 2. SLOW PATH: Use generic map (Python Loop) on Objects
        raw_results = space.map(self.operation)
        return self.target_space_class(raw_results)

    def transform_raw(self, raw_states: Sequence[Any]) -> Sequence[Any]:
        """
        Pure Data Transformation.
        """
        # --- CRITICAL FIX ---
        # We REMOVED the check for 'isinstance(np.ndarray)'.
        # For Abstract data, 'array[0]' is ambiguous (Row 0 vs Element 0).
        # We MUST force iteration to ensure the operation maps over every item.

        return [self.raw_operation(s) for s in raw_states]


# --- 3. VECTOR TRANSFORMATION (Floats, Ints) ---
class VectorStateTransformation(DiscreteStateTransformation):
    """
    High-Performance Transformation for Vectors.
    Leverages JAX vmap + Raw Factory.
    """

    def __init__(self,
                 operation: Callable[[Any], Any],
                 target_space_class: Type[IDiscreteStateSpace] = VectorStateSpace,
                 raw_operation: Optional[Callable[[Any], Any]] = None):

        # Pass raw_operation correctly to parent
        super().__init__(operation, target_space_class, raw_operation)

    def transform(self, space: VectorStateSpace) -> VectorStateSpace:
        """
        Executes transformation entirely in JAX/Numpy memory where possible.
        """
        # 1. Run the Math (Fast JAX Map via raw_operation)
        if hasattr(space, 'get_matrix'):
            raw_matrix = space.get_matrix()
            raw_output = self.transform_raw(raw_matrix)
        else:
            return super().transform(space)

        # 2. Check if optimization succeeded (returned Array vs List)
        if isinstance(raw_output, list) or (hasattr(raw_output, 'dtype') and raw_output.dtype == object):
            return self.target_space_class(raw_output)

        # 3. FAST PATH: Create Space from Raw Data
        dim = raw_output.shape[-1]

        # Use the factory if available
        if hasattr(self.target_space_class, 'from_raw_data'):
            return self.target_space_class.from_raw_data(
                raw_data=raw_output,
                wrapper=lambda x: VectorState(tuple(x.tolist())),
                dim=dim
            )

        # Fallback
        return self.target_space_class.from_raw_array(np.array(raw_output))

    def transform_raw(self, raw_matrix: Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray, jnp.ndarray]:
        """
        Pure Matrix Transformation.
        Uses self.raw_operation.
        """
        # Expectation: Operation handles broadcasting
        return self.raw_operation(raw_matrix)