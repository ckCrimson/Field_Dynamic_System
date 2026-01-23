import jax
import jax.numpy as jnp
from typing import Optional, Callable, List, Any
from src.field_dynamic_system.core.field.mappings import FieldMapper, DiscreteFieldMapper, ContinuousFieldMapper
from src.field_dynamic_system.core.field.transform import FieldTransform, LinearTransform
from src.field_dynamic_system.core.field.algebra import IFieldAlgebra
from src.field_dynamic_system.core.field.compositions import FieldComposition, ClosedFieldComposition, \
    AdditionComposition


class FieldSpaceTransformer:
    @staticmethod
    def apply(mapper: FieldMapper,
              transform: FieldTransform,
              output_algebra: IFieldAlgebra,  # <--- REQUIRED: You must define the output structure
              override_bg_func: Optional[Callable] = None) -> FieldMapper:
        """
        Applies transform to a mapper.

        Args:
            mapper: Input FieldMapper
            transform: The mathematical operation (Pure Math)
            output_algebra: The FieldAlgebra instance that describes the output space.
            override_bg_func: Optional background function override.
        """

        # We use the passed algebra directly
        new_algebra = output_algebra

        if isinstance(mapper, DiscreteFieldMapper):
            return FieldSpaceTransformer._apply_discrete(mapper, transform, new_algebra, override_bg_func)
        elif isinstance(mapper, ContinuousFieldMapper):
            return FieldSpaceTransformer._apply_continuous(mapper, transform, new_algebra, override_bg_func)
        raise TypeError(f"Unknown mapper: {type(mapper)}")

        # =========================================================
        # 🚀 NEW: RAW KERNEL (For Multi-Step Loops)
        # =========================================================
    @staticmethod
    def apply_raw(explicit_buffer: jnp.ndarray,
                      transform: FieldTransform) -> jnp.ndarray:
            """
            Raw Transform Kernel.

            Input:  JAX Array (N, Dim_In)
            Output: JAX Array (N, Dim_Out)

            Use this inside the simulation loop. It has ZERO Python overhead.
            """
            # OPTIMIZATION: Linear Transforms are just Matrix Multiplication.
            # It is significantly faster to do one Dot Product than vmap(func).
            if isinstance(transform, LinearTransform):
                # Buffer: (N, In)
                # Matrix: (Out, In)
                # Result: (N, In) @ (Out, In).T -> (N, In) @ (In, Out) -> (N, Out)
                return jnp.dot(explicit_buffer, transform.matrix.T)

            # FALLBACK: Non-Linear / Norms
            # We vectorise the transform function over the 0-th axis (batch of states)
            return jax.vmap(transform)(explicit_buffer)

    @staticmethod
    def _apply_discrete(mapper: DiscreteFieldMapper,
                        transform: FieldTransform,
                        new_algebra: IFieldAlgebra,
                        override_bg_func=None):

        # 1. Transform Explicit Buffer
        if isinstance(transform, LinearTransform):
            new_explicit = jnp.dot(mapper.explicit_buffer, transform.matrix.T)
        else:
            new_explicit = jax.vmap(transform)(mapper.explicit_buffer)

        # 2. Handle Background Logic
        if override_bg_func is not None:
            final_bg = override_bg_func
        elif mapper.background_func is not None:
            old_bg = mapper.background_func

            def composed_bg(s):
                return transform(old_bg(s))

            final_bg = composed_bg
        else:
            final_bg = None

        return DiscreteFieldMapper(
            mapper.state_space,
            new_algebra,
            explicit_buffer=new_explicit,
            mask_buffer=mapper.mask_buffer,
            bg_func=final_bg
        )

    @staticmethod
    def _apply_continuous(mapper: ContinuousFieldMapper,
                          transform: FieldTransform,
                          new_algebra: IFieldAlgebra,
                          override_bg_func=None):

        # 1. Determine New Background
        if override_bg_func is not None:
            new_bg_func = override_bg_func
        else:
            old_bg = mapper.background_func

            def composed_bg(state):
                raw_old = old_bg(state)
                if isinstance(transform, LinearTransform):
                    return jnp.dot(raw_old, transform.matrix.T)
                else:
                    return jax.vmap(transform)(raw_old)

            new_bg_func = composed_bg

        # 2. Transform Sparse Cache
        new_cache = {}
        for s, val in mapper.sparse_cache.items():
            new_cache[s] = transform(val)

        return ContinuousFieldMapper(
            mapper.state_space,
            new_algebra,
            bg_func=new_bg_func,
            sparse_cache=new_cache
        )



class FieldSpaceComposer:
    """
    The Engine for Binary Field Operations: F3 = F1 alpha F2.
    """

    @staticmethod
    def compose(mapper_a: FieldMapper,
                mapper_b: FieldMapper,
                composition_op: FieldComposition,
                output_algebra: IFieldAlgebra,
                override_bg_func: Optional[Callable] = None) -> FieldMapper:

        is_a_disc = isinstance(mapper_a, DiscreteFieldMapper)
        is_b_disc = isinstance(mapper_b, DiscreteFieldMapper)

        if is_a_disc and is_b_disc:
            return FieldSpaceComposer._compose_discrete_discrete(
                mapper_a, mapper_b, composition_op, output_algebra, override_bg_func
            )
        elif not is_a_disc and not is_b_disc:
            return FieldSpaceComposer._compose_continuous_continuous(
                mapper_a, mapper_b, composition_op, output_algebra, override_bg_func
            )
        else:
            return FieldSpaceComposer._compose_hybrid(
                mapper_a, mapper_b, composition_op, output_algebra, override_bg_func
            )

    # =========================================================
    # RAW KERNELS
    # =========================================================
    @staticmethod
    def compose_raw(buffer_a: jnp.ndarray, buffer_b: jnp.ndarray, op: FieldComposition) -> jnp.ndarray:
        return op.compose(buffer_a, buffer_b)

    @staticmethod
    def compose_unaligned_raw(buffer_a: jnp.ndarray, indices_a: jnp.ndarray,
                              buffer_b: jnp.ndarray, indices_b: jnp.ndarray,
                              total_union_size: int,
                              op: FieldComposition) -> jnp.ndarray:
        dim = buffer_a.shape[1]
        global_buffer = jnp.zeros((total_union_size, dim), dtype=buffer_a.dtype)
        global_buffer = global_buffer.at[indices_a].set(buffer_a)

        if isinstance(op, AdditionComposition):
            global_buffer = global_buffer.at[indices_b].add(buffer_b)
        else:
            current = global_buffer[indices_b]
            global_buffer = global_buffer.at[indices_b].set(op.compose(current, buffer_b))
        return global_buffer

    # =========================================================
    # HELPER: Fetch Implicit Values (Vectorized)
    # =========================================================
    @staticmethod
    def _get_implicit_values(mapper: DiscreteFieldMapper, states: List[Any]) -> jnp.ndarray:
        """
        Returns the 'Missing' values for a list of states.
        Correctly handles Batch calls to background_func.
        """
        dim = mapper.explicit_buffer.shape[1]
        if not states:
            return jnp.zeros((0, dim), dtype=mapper.explicit_buffer.dtype)

        if mapper.background_func is None:
            return mapper.algebra.get_zero((len(states),))

        # Call BG Func with the FULL LIST (Batch Mode)
        # This fixes the "len(s)=2" bug when passing single strings
        try:
            res = mapper.background_func(states)
            # Ensure shape is (N, D)
            if res.ndim == 1:
                res = res.reshape(len(states), -1)
            return res
        except Exception:
            # Fallback for poorly behaved bg_funcs (one-by-one)
            # (Only use if batch fails)
            vals = [mapper.background_func(s) for s in states]
            res = jnp.array(vals)
            if res.ndim == 1:
                res = res.reshape(len(states), -1)
            elif res.ndim == 3:
                res = res.reshape(len(states), -1)
            return res

    # =========================================================
    # DISCRETE + DISCRETE
    # =========================================================
    @staticmethod
    def _compose_discrete_discrete(a: DiscreteFieldMapper,
                                   b: DiscreteFieldMapper,
                                   op: FieldComposition,
                                   out_alg: IFieldAlgebra,
                                   final_bg=None):

        # 1. Background Logic
        bg_a = a.background_func
        bg_b = b.background_func

        if final_bg:
            new_bg = final_bg
        elif bg_a is None and bg_b is None:
            new_bg = None
        else:
            def composed_bg(s):
                val_a = bg_a(s) if bg_a else a.algebra.get_zero((1,))
                val_b = bg_b(s) if bg_b else b.algebra.get_zero((1,))
                return op.compose(val_a, val_b)

            new_bg = composed_bg

        # 2. State Alignment

        # FAST PATH (Identity)
        if a.state_space is b.state_space:
            new_buffer = FieldSpaceComposer.compose_raw(a.explicit_buffer, b.explicit_buffer, op)
            new_mask = a.mask_buffer | b.mask_buffer
            return DiscreteFieldMapper(a.state_space, out_alg, new_buffer, new_mask, bg_func=new_bg)

        # SLOW PATH (Symbolic Join with Reordering Safety)
        states_a = a.state_space.states
        states_b = b.state_space.states

        # A. Create Union Space FIRST (Establishes Canonical Order)
        # We assume states are hashable.
        union_set = set(states_a).union(set(states_b))
        union_list_raw = list(union_set)

        try:
            new_space = a.state_space.create_subset(union_list_raw)
        except:
            new_space = type(a.state_space)(union_list_raw)

        # B. Get the AUTHORITATIVE state order from the new space
        # (AbstractDiscreteStateSpace often sorts/reorders states)
        target_states = new_space.states

        # C. Build Maps for Source Data
        idx_map_a = {s: i for i, s in enumerate(states_a)}
        idx_map_b = {s: i for i, s in enumerate(states_b)}

        # D. Partition target_states into lists for batch processing
        # We need to preserve the Target Order in the final assembly.
        # But we can't easily vectorise "Some from A, Some from B" mixed arbitrarily.
        # Instead, we create a full-size buffer and scatter into it.

        dim = a.explicit_buffer.shape[1]
        final_buffer = jnp.zeros((len(target_states), dim), dtype=a.explicit_buffer.dtype)

        # Indices in Target Buffer
        target_indices_common = []
        target_indices_only_a = []
        target_indices_only_b = []

        # Source Indices / Objects
        src_indices_a_common = []
        src_indices_b_common = []

        src_indices_a_only = []
        objs_only_a_implicit_b = []  # Objects needed for B's implicit fetch

        src_indices_b_only = []
        objs_only_b_implicit_a = []  # Objects needed for A's implicit fetch

        # Classify every state in the Target Order
        for i, s in enumerate(target_states):
            in_a = s in idx_map_a
            in_b = s in idx_map_b

            if in_a and in_b:
                target_indices_common.append(i)
                src_indices_a_common.append(idx_map_a[s])
                src_indices_b_common.append(idx_map_b[s])
            elif in_a:
                target_indices_only_a.append(i)
                src_indices_a_only.append(idx_map_a[s])
                objs_only_a_implicit_b.append(s)
            elif in_b:
                target_indices_only_b.append(i)
                src_indices_b_only.append(idx_map_b[s])
                objs_only_b_implicit_a.append(s)

        # E. Execute Batched Operations & Scatter

        # 1. Intersection (A op B)
        if target_indices_common:
            raw_a = a.explicit_buffer[jnp.array(src_indices_a_common)]
            raw_b = b.explicit_buffer[jnp.array(src_indices_b_common)]
            vals = op.compose(raw_a, raw_b)
            final_buffer = final_buffer.at[jnp.array(target_indices_common)].set(vals)

        # 2. Only A (A op Implicit B)
        if target_indices_only_a:
            raw_a = a.explicit_buffer[jnp.array(src_indices_a_only)]
            raw_b_implicit = FieldSpaceComposer._get_implicit_values(b, objs_only_a_implicit_b)
            vals = op.compose(raw_a, raw_b_implicit)
            final_buffer = final_buffer.at[jnp.array(target_indices_only_a)].set(vals)

        # 3. Only B (Implicit A op B)
        if target_indices_only_b:
            raw_b = b.explicit_buffer[jnp.array(src_indices_b_only)]
            raw_a_implicit = FieldSpaceComposer._get_implicit_values(a, objs_only_b_implicit_a)
            vals = op.compose(raw_a_implicit, raw_b)
            final_buffer = final_buffer.at[jnp.array(target_indices_only_b)].set(vals)

        # F. Final Mask (All valid)
        final_mask = jnp.ones(len(target_states), dtype=bool)

        return DiscreteFieldMapper(
            new_space, out_alg,
            explicit_buffer=final_buffer,
            mask_buffer=final_mask,
            bg_func=new_bg
        )

    # =========================================================
    # CONTINUOUS & HYBRID (Unchanged)
    # =========================================================
    @staticmethod
    def _compose_continuous_continuous(a, b, op, out_alg, final_bg=None):
        if final_bg:
            new_bg_func = final_bg
        else:
            def composed_bg(state):
                return op.compose(a.background_func(state), b.background_func(state))

            new_bg_func = composed_bg

        keys = set(a.sparse_cache.keys()) | set(b.sparse_cache.keys())
        new_cache = {}
        for k in keys:
            val_a = a.sparse_cache.get(k, a.get_fields_at(k)[0].value)
            val_b = b.sparse_cache.get(k, b.get_fields_at(k)[0].value)
            new_cache[k] = op.compose(val_a, val_b)

        return ContinuousFieldMapper(a.state_space, out_alg, bg_func=new_bg_func, sparse_cache=new_cache)

    @staticmethod
    def _compose_hybrid(a, b, op, out_alg, final_bg=None):
        cont_mapper = a if isinstance(a, ContinuousFieldMapper) else b
        if final_bg:
            new_bg = final_bg
        else:
            def composed_bg(state):
                return op.compose(a.get_fields_at(state)[0].value, b.get_fields_at(state)[0].value)

            new_bg = composed_bg
        return ContinuousFieldMapper(cont_mapper.state_space, out_alg, bg_func=new_bg)