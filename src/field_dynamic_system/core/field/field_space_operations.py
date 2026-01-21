import jax
import jax.numpy as jnp
from typing import Optional, Callable
from src.field_dynamic_system.core.field.mappings import FieldMapper, DiscreteFieldMapper, ContinuousFieldMapper
from src.field_dynamic_system.core.field.transform import FieldTransform, LinearTransform


class FieldSpaceTransformer:
    @staticmethod
    def apply(mapper: FieldMapper,
              transform: FieldTransform,
              override_bg_func: Optional[Callable] = None) -> FieldMapper:
        """
        Applies transform to a mapper.

        Args:
            mapper: Input FieldMapper
            transform: The mathematical operation
            override_bg_func: (Optional) If provided, this function becomes the
                              background of the NEW mapper, ignoring the old background.
        """
        new_algebra = transform.output_algebra_type()

        if isinstance(mapper, DiscreteFieldMapper):
            return FieldSpaceTransformer._apply_discrete(mapper, transform, new_algebra, override_bg_func)
        elif isinstance(mapper, ContinuousFieldMapper):
            return FieldSpaceTransformer._apply_continuous(mapper, transform, new_algebra, override_bg_func)
        raise TypeError(f"Unknown mapper: {type(mapper)}")

    @staticmethod
    def _apply_discrete(mapper: DiscreteFieldMapper,
                        transform: FieldTransform,
                        new_algebra,
                        override_bg_func=None):

        # 1. Transform Explicit Buffer (Same as before)
        if isinstance(transform, LinearTransform):
            new_explicit = jnp.dot(mapper.explicit_buffer, transform.matrix.T)
        else:
            new_explicit = jax.vmap(transform)(mapper.explicit_buffer)

        # 2. Handle Background Logic
        # Priority: Override > Transform(Old) > None
        if override_bg_func is not None:
            final_bg = override_bg_func
        elif mapper.background_func is not None:
            # If the Discrete mapper HAD a background function (hybrid), we compose it
            # This logic mimics continuous composition for the hybrid case
            old_bg = mapper.background_func

            def composed_bg(s):
                return transform(old_bg(s))  # We can optimize this if linear

            final_bg = composed_bg
        else:
            final_bg = None

        return DiscreteFieldMapper(
            mapper.state_space,
            new_algebra,
            explicit_buffer=new_explicit,
            mask_buffer=mapper.mask_buffer,
            bg_func=final_bg  # Set the new background
        )

    @staticmethod
    def _apply_continuous(mapper: ContinuousFieldMapper,
                          transform: FieldTransform,
                          new_algebra,
                          override_bg_func=None):

        # 1. Determine New Background
        if override_bg_func is not None:
            # CASE A: User provided a specific new background
            new_bg_func = override_bg_func
        else:
            # CASE B: Composition (Standard)
            # New_BG(s) = Transform(Old_BG(s))
            old_bg = mapper.background_func

            def composed_bg(state):
                raw_old = old_bg(state)
                if isinstance(transform, LinearTransform):
                    return jnp.dot(raw_old, transform.matrix.T)
                else:
                    return jax.vmap(transform)(raw_old)

            new_bg_func = composed_bg

        # 2. Transform Sparse Cache (Eagerly)
        new_cache = {}
        for s, val in mapper.sparse_cache.items():
            new_cache[s] = transform(val)

        return ContinuousFieldMapper(
            mapper.state_space,
            new_algebra,
            bg_func=new_bg_func,
            sparse_cache=new_cache
        )