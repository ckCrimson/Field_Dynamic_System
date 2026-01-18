from typing import Callable, Dict, Tuple
import jax.numpy as jnp
from jax import jit

from src.field_dynamic_system.core.field.mappings import FieldMapper, DiscreteFieldMapper, ContinuousFieldMapper
from src.field_dynamic_system.core.field.algebra import IFieldAlgebra
from src.field_dynamic_system.core.field.compositions import GeneralizedFieldComposition
from src.field_dynamic_system.core.field.transform import FieldTransform
from src.field_dynamic_system.core.field.data import extract_val

# Kernel Cache to avoid recompiling the math logic
_TRANSFORM_KERNEL_CACHE: Dict[Tuple[str, str], Callable] = {}
_COMPOSITION_KERNEL_CACHE: Dict[Tuple[str, str], Callable] = {}


class FieldSpaceTransform:
    """
    Applies a Unary Transform.
    Optimization: "Collapses" Discrete Fields into materialized buffers to avoid closure recompilation.
    """

    @staticmethod
    def apply(source_mapper: FieldMapper,
              transform_op: FieldTransform,
              target_algebra: IFieldAlgebra) -> FieldMapper:

        # 1. Get Static Kernel
        cache_key = (type(transform_op).__name__, type(target_algebra).__name__)
        if cache_key not in _TRANSFORM_KERNEL_CACHE:
            _TRANSFORM_KERNEL_CACHE[cache_key] = jit(lambda x: extract_val(transform_op.transform(x)))
        static_kernel = _TRANSFORM_KERNEL_CACHE[cache_key]

        # --- PATH A: DISCRETE (COLLAPSE STRATEGY) ---
        if isinstance(source_mapper, DiscreteFieldMapper):
            # 1. Materialize entire source field
            # (Fetching the whole field is extremely fast for arrays)
            n_states = source_mapper.state_space.num_states
            all_indices = jnp.arange(n_states, dtype=jnp.int32)

            # This triggers the source's JIT kernel once
            source_values = source_mapper.get_fields_at(all_indices)

            # 2. Apply Transform Batch-wise
            new_values = static_kernel(source_values)

            # 3. Create "Collapsed" Mapper
            # No background function needed (default is Zero), because we filled the buffer!
            new_mapper = DiscreteFieldMapper(source_mapper.state_space, target_algebra, bg_func=None)
            new_mapper.explicit_buffer = new_values
            # Mark all states as "Explicit" (overridden)
            new_mapper.mask_buffer = jnp.ones((n_states, 1), dtype=bool)

            return new_mapper

        # --- PATH B: CONTINUOUS (LAZY STRATEGY) ---
        else:
            # Must use closures for infinite spaces
            src_bg = source_mapper.background_func

            def composed_bg(states):
                return static_kernel(src_bg(states))

            return ContinuousFieldMapper(source_mapper.state_space, target_algebra, composed_bg)


class FieldSpaceComposition:
    """
    Applies a Binary Composition.
    Optimization: "Collapses" Discrete Fields.
    """

    @staticmethod
    def apply(mapper_a: FieldMapper,
              mapper_b: FieldMapper,
              composition_op: GeneralizedFieldComposition,
              target_algebra: IFieldAlgebra) -> FieldMapper:

        # 1. Get Static Kernel
        cache_key = (type(composition_op).__name__, type(target_algebra).__name__)
        if cache_key not in _COMPOSITION_KERNEL_CACHE:
            _COMPOSITION_KERNEL_CACHE[cache_key] = jit(lambda a, b: extract_val(composition_op.compose(a, b)))
        static_kernel = _COMPOSITION_KERNEL_CACHE[cache_key]

        # --- PATH A: DISCRETE (COLLAPSE STRATEGY) ---
        if isinstance(mapper_a, DiscreteFieldMapper) and isinstance(mapper_b, DiscreteFieldMapper):
            # Assume spaces are compatible (same N)
            n_states = mapper_a.state_space.num_states
            all_indices = jnp.arange(n_states, dtype=jnp.int32)

            # 1. Fetch Both fully
            vals_a = mapper_a.get_fields_at(all_indices)
            vals_b = mapper_b.get_fields_at(all_indices)

            # 2. Compute Composition
            new_values = static_kernel(vals_a, vals_b)

            # 3. Create Collapsed Mapper
            new_mapper = DiscreteFieldMapper(mapper_a.state_space, target_algebra, bg_func=None)
            new_mapper.explicit_buffer = new_values
            new_mapper.mask_buffer = jnp.ones((n_states, 1), dtype=bool)

            return new_mapper

        # --- PATH B: CONTINUOUS (LAZY STRATEGY) ---
        else:
            bg_a = mapper_a.background_func
            bg_b = mapper_b.background_func

            def composed_bg(states):
                return static_kernel(bg_a(states), bg_b(states))

            return ContinuousFieldMapper(mapper_a.state_space, target_algebra, composed_bg)