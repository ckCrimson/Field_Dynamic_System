import jax
import jax.numpy as jnp
from typing import Optional, Callable, List, Any, Tuple, Sequence

import numpy as np

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
    def apply_batch(fields: Sequence[Tuple[jnp.ndarray, jnp.ndarray]],
                    transform: FieldTransform) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        BATCH OPTIMIZED PATH.
        Applies a transform to N sparse fields in a SINGLE JAX operation.

        Args:
            fields: List of [(ids_A, vals_A), (ids_B, vals_B), ...]
            transform: The operation (Linear or Functional)

        Returns:
            List of [(ids_A, new_vals_A), (ids_B, new_vals_B), ...]
            (IDs are preserved, Values are transformed)
        """
        if not fields:
            return []

        # 1. UNZIP & MEASURE
        # We need the lengths to split the result back later.
        list_ids = [f[0] for f in fields]
        list_vals = [f[1] for f in fields]

        # Calculate split points for reconstruction
        # We calculate the lengths of each value array
        lengths = [v.shape[0] for v in list_vals]

        # 2. MASSIVE CONCATENATION (The "Fuse")
        # Combine 1,000 small arrays into 1 big array for the GPU
        all_vals = jnp.concatenate(list_vals, axis=0)

        # 3. TRANSFORM (The "Compute")
        # Single Kernel execution (Dot Product or Vmap)
        if isinstance(transform, LinearTransform):
            # Optimized Matrix Mul: (Total_N, In) @ (Out, In).T
            transformed_big = jnp.dot(all_vals, transform.matrix.T)
        else:
            # Fallback Vmap
            transformed_big = jax.vmap(transform)(all_vals)

        # 4. SLICING (The "Split")
        # We assume jnp.split is efficient enough on CPU/GPU boundary for this logic.
        # Note: jnp.split requires indices, so we compute cumulative sum.
        split_indices = np.cumsum(lengths)[:-1]  # Remove last to avoid empty tail

        # We use list conversion to enable iteration for re-packing
        # Note: splitting usually returns a list of arrays
        new_vals_list = jnp.split(transformed_big, split_indices)

        # 5. RE-PACK
        # Pair the original IDs with the new Transformed Values
        result_batch = []
        for i in range(len(list_ids)):
            result_batch.append((list_ids[i], new_vals_list[i]))

        return result_batch

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

            # Note: We rely on JAX's implicit broadcasting or explicit dot
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




# class FieldSpaceComposer:
#     """
#     The Engine for Binary Field Operations: F3 = F1 alpha F2.
#     """
#
#     @staticmethod
#     def compose(mapper_a: FieldMapper,
#                 mapper_b: FieldMapper,
#                 composition_op: FieldComposition,
#                 output_algebra: IFieldAlgebra,
#                 override_bg_func: Optional[Callable] = None) -> FieldMapper:
#
#         is_a_disc = isinstance(mapper_a, DiscreteFieldMapper)
#         is_b_disc = isinstance(mapper_b, DiscreteFieldMapper)
#
#         if is_a_disc and is_b_disc:
#             return FieldSpaceComposer._compose_discrete_discrete(
#                 mapper_a, mapper_b, composition_op, output_algebra, override_bg_func
#             )
#         elif not is_a_disc and not is_b_disc:
#             return FieldSpaceComposer._compose_continuous_continuous(
#                 mapper_a, mapper_b, composition_op, output_algebra, override_bg_func
#             )
#         else:
#             return FieldSpaceComposer._compose_hybrid(
#                 mapper_a, mapper_b, composition_op, output_algebra, override_bg_func
#             )
#
#     # =========================================================
#     # RAW KERNELS (The Engine Room)
#     # =========================================================
#
#     @staticmethod
#     def compose_aligned_raw(buffer_a: jnp.ndarray, buffer_b: jnp.ndarray, op: FieldComposition) -> jnp.ndarray:
#         """
#         Fast Path: Pure Tensor Operation.
#         Use this when you KNOW indices are identical (e.g. same StateSpace).
#         """
#         return op.compose(buffer_a, buffer_b)
#
#     @staticmethod
#     def compose_raw(ids_a: jnp.ndarray, vals_a: jnp.ndarray,
#                     ids_b: jnp.ndarray, vals_b: jnp.ndarray,
#                     op: FieldComposition) -> Tuple[jnp.ndarray, jnp.ndarray]:
#         """
#         Merges two sparse fields. Works for Infinite Grids where IDs = Coordinates.
#         """
#         # 1. Stack (Union)
#         all_ids = jnp.concatenate([ids_a, ids_b], axis=0)
#         all_vals = jnp.concatenate([vals_a, vals_b], axis=0)
#
#         # 2. Unique (Topology / Collision Detection)
#         # axis=0 handles both scalar IDs and vector IDs (coordinates)
#         unique_ids, inverse_indices = jnp.unique(all_ids, return_inverse=True, axis=0)
#
#         # 3. Aggregate (Algebra)
#         out_dim = vals_a.shape[1]
#         merged_vals = jnp.zeros((unique_ids.shape[0], out_dim), dtype=vals_a.dtype)
#
#         if isinstance(op, AdditionComposition):
#             merged_vals = merged_vals.at[inverse_indices].add(all_vals)
#         else:
#             merged_vals = merged_vals.at[inverse_indices].add(all_vals)
#
#         return unique_ids, merged_vals
#
#         if isinstance(op, AdditionComposition):
#             merged_vals = merged_vals.at[inverse_indices].add(all_vals)
#         else:
#             # Fallback/Generic: Default to Add semantics for accumulation
#             merged_vals = merged_vals.at[inverse_indices].add(all_vals)
#
#         return unique_ids, merged_vals
#
#     # =========================================================
#     # HELPER: Fetch Implicit Values (Vectorized)
#     # =========================================================
#     @staticmethod
#     def _get_implicit_values(mapper: DiscreteFieldMapper, states: List[Any]) -> jnp.ndarray:
#         dim = mapper.explicit_buffer.shape[1]
#         if not states:
#             return jnp.zeros((0, dim), dtype=mapper.explicit_buffer.dtype)
#
#         if mapper.background_func is None:
#             return mapper.algebra.get_zero((len(states),))
#
#         try:
#             res = mapper.background_func(states)
#             if res.ndim == 1: res = res.reshape(len(states), -1)
#             return res
#         except Exception:
#             vals = [mapper.background_func(s) for s in states]
#             res = jnp.array(vals)
#             if res.ndim == 1:
#                 res = res.reshape(len(states), -1)
#             elif res.ndim == 3:
#                 res = res.reshape(len(states), -1)
#             return res
#
#     # =========================================================
#     # DISCRETE + DISCRETE (Object Logic)
#     # =========================================================
#     @staticmethod
#     def _compose_discrete_discrete(a: DiscreteFieldMapper,
#                                    b: DiscreteFieldMapper,
#                                    op: FieldComposition,
#                                    out_alg: IFieldAlgebra,
#                                    final_bg=None):
#         # 1. Background Logic
#         bg_a = a.background_func
#         bg_b = b.background_func
#         if final_bg:
#             new_bg = final_bg
#         elif bg_a is None and bg_b is None:
#             new_bg = None
#         else:
#             def composed_bg(s):
#                 val_a = bg_a(s) if bg_a else a.algebra.get_zero((1,))
#                 val_b = bg_b(s) if bg_b else b.algebra.get_zero((1,))
#                 return op.compose(val_a, val_b)
#
#             new_bg = composed_bg
#
#         # 2. State Alignment
#
#         # --- FAST PATH FIX IS HERE ---
#         if a.state_space is b.state_space:
#             # We must use 'compose_aligned_raw' (2 args), not 'compose_raw' (5 args)
#             new_buffer = FieldSpaceComposer.compose_aligned_raw(a.explicit_buffer, b.explicit_buffer, op)
#             new_mask = a.mask_buffer | b.mask_buffer
#             return DiscreteFieldMapper(a.state_space, out_alg, new_buffer, new_mask, bg_func=new_bg)
#
#         # SLOW PATH: Symbolic Join
#         states_a = a.state_space.states
#         states_b = b.state_space.states
#
#         # A. Union Space
#         union_set = set(states_a).union(set(states_b))
#         union_list_raw = list(union_set)
#         try:
#             new_space = a.state_space.create_subset(union_list_raw)
#         except:
#             new_space = type(a.state_space)(union_list_raw)
#
#         # B. Authoritative Order
#         target_states = new_space.states
#         idx_map_a = {s: i for i, s in enumerate(states_a)}
#         idx_map_b = {s: i for i, s in enumerate(states_b)}
#
#         # C. Partitioning & Scatter Logic
#         dim = a.explicit_buffer.shape[1]
#         final_buffer = jnp.zeros((len(target_states), dim), dtype=a.explicit_buffer.dtype)
#
#         target_indices_common = []
#         target_indices_only_a = []
#         target_indices_only_b = []
#         src_indices_a_common = []
#         src_indices_b_common = []
#         src_indices_a_only = []
#         objs_only_a_implicit_b = []
#         src_indices_b_only = []
#         objs_only_b_implicit_a = []
#
#         for i, s in enumerate(target_states):
#             in_a = s in idx_map_a
#             in_b = s in idx_map_b
#
#             if in_a and in_b:
#                 target_indices_common.append(i)
#                 src_indices_a_common.append(idx_map_a[s])
#                 src_indices_b_common.append(idx_map_b[s])
#             elif in_a:
#                 target_indices_only_a.append(i)
#                 src_indices_a_only.append(idx_map_a[s])
#                 objs_only_a_implicit_b.append(s)
#             elif in_b:
#                 target_indices_only_b.append(i)
#                 src_indices_b_only.append(idx_map_b[s])
#                 objs_only_b_implicit_a.append(s)
#
#         if target_indices_common:
#             raw_a = a.explicit_buffer[jnp.array(src_indices_a_common)]
#             raw_b = b.explicit_buffer[jnp.array(src_indices_b_common)]
#             vals = op.compose(raw_a, raw_b)
#             final_buffer = final_buffer.at[jnp.array(target_indices_common)].set(vals)
#
#         if target_indices_only_a:
#             raw_a = a.explicit_buffer[jnp.array(src_indices_a_only)]
#             raw_b_implicit = FieldSpaceComposer._get_implicit_values(b, objs_only_a_implicit_b)
#             vals = op.compose(raw_a, raw_b_implicit)
#             final_buffer = final_buffer.at[jnp.array(target_indices_only_a)].set(vals)
#
#         if target_indices_only_b:
#             raw_b = b.explicit_buffer[jnp.array(src_indices_b_only)]
#             raw_a_implicit = FieldSpaceComposer._get_implicit_values(a, objs_only_b_implicit_a)
#             vals = op.compose(raw_a_implicit, raw_b)
#             final_buffer = final_buffer.at[jnp.array(target_indices_only_b)].set(vals)
#
#         final_mask = jnp.ones(len(target_states), dtype=bool)
#
#         return DiscreteFieldMapper(
#             new_space, out_alg,
#             explicit_buffer=final_buffer,
#             mask_buffer=final_mask,
#             bg_func=new_bg
#         )
#
#     # =========================================================
#     # CONTINUOUS & HYBRID (Unchanged)
#     # =========================================================
#     @staticmethod
#     def _compose_continuous_continuous(a, b, op, out_alg, final_bg=None):
#         if final_bg:
#             new_bg_func = final_bg
#         else:
#             def composed_bg(state):
#                 return op.compose(a.background_func(state), b.background_func(state))
#
#             new_bg_func = composed_bg
#
#         keys = set(a.sparse_cache.keys()) | set(b.sparse_cache.keys())
#         new_cache = {}
#         for k in keys:
#             val_a = a.sparse_cache.get(k, a.get_fields_at(k)[0].value)
#             val_b = b.sparse_cache.get(k, b.get_fields_at(k)[0].value)
#             new_cache[k] = op.compose(val_a, val_b)
#
#         return ContinuousFieldMapper(a.state_space, out_alg, bg_func=new_bg_func, sparse_cache=new_cache)
#
#     @staticmethod
#     def _compose_hybrid(a, b, op, out_alg, final_bg=None):
#         cont_mapper = a if isinstance(a, ContinuousFieldMapper) else b
#         if final_bg:
#             new_bg = final_bg
#         else:
#             def composed_bg(state):
#                 return op.compose(a.get_fields_at(state)[0].value, b.get_fields_at(state)[0].value)
#
#             new_bg = composed_bg
#         return ContinuousFieldMapper(cont_mapper.state_space, out_alg, bg_func=new_bg)


class FieldSpaceComposer:
    """
    The Engine for Binary Field Operations: F3 = F1 [op] F2.
    Handles both High-Speed Physics (aligned) and Complex Setup (misaligned).
    """

    @staticmethod
    def compose(mapper_a: FieldMapper,
                mapper_b: FieldMapper,
                output_algebra: IFieldAlgebra,
                op_name: str = "add",
                override_bg_func: Optional[Callable] = None) -> FieldMapper:
        """
        Main Entry Point. Dispatches based on mapper types.
        """
        is_a_disc = isinstance(mapper_a, DiscreteFieldMapper)
        is_b_disc = isinstance(mapper_b, DiscreteFieldMapper)

        if is_a_disc and is_b_disc:
            return FieldSpaceComposer._compose_discrete_discrete(
                mapper_a, mapper_b, output_algebra, op_name, override_bg_func
            )
        elif not is_a_disc and not is_b_disc:
            return FieldSpaceComposer._compose_continuous_continuous(
                mapper_a, mapper_b, output_algebra, op_name, override_bg_func
            )
        else:
            return FieldSpaceComposer._compose_hybrid(
                mapper_a, mapper_b, output_algebra, op_name, override_bg_func
            )

    # =========================================================
    # RAW KERNELS (The Engine Room)
    # =========================================================

    @staticmethod
    def compose_aligned_raw(buffer_a: jnp.ndarray, buffer_b: jnp.ndarray,
                            op: 'FieldComposition') -> jnp.ndarray:
        """
        Fast Path: Pure Tensor Operation.
        Delegate math to the Composition Object.
        """
        return op.compose(buffer_a, buffer_b)

    @staticmethod
    def compose_batch(fields: Sequence[Tuple[jnp.ndarray, jnp.ndarray]],
                      op: FieldComposition) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        BATCH OPTIMIZED PATH.
        Merges N sparse fields in a SINGLE JAX operation.

        Args:
            fields: A list of tuples [(ids_1, vals_1), (ids_2, vals_2), ...]
            op: The composition strategy (e.g., AdditionComposition)

        Returns:
            (unique_ids, aggregated_values)
        """
        if not fields:
            # Return empty arrays with correct shape/dtype logic if needed
            # Assuming 1D IDs and generic float values for safety
            return jnp.array([], dtype=jnp.int32), jnp.array([], dtype=jnp.float32)

        # 1. Unzip the List (Python side, very cheap)
        # We separate all ID arrays and all Value arrays
        list_of_ids = [f[0] for f in fields]
        list_of_vals = [f[1] for f in fields]

        # 2. MASSIVE CONCATENATION (The "Big Bang")
        # Instead of merging A+B, then +C... we dump everything into one bucket.
        all_ids = jnp.concatenate(list_of_ids, axis=0)
        all_vals = jnp.concatenate(list_of_vals, axis=0)

        # 3. TOPOLOGY RESOLUTION (Unique Sort)
        # We find unique states across ALL fields simultaneously.
        # return_inverse gives us the mapping from the massive list to the unique list.
        unique_ids, inverse_indices = jnp.unique(all_ids, return_inverse=True, axis=0)

        # 4. AGGREGATION (Reduce)
        # We prepare the output buffer
        out_dim = all_vals.shape[1]

        # Get Identity (0.0 for Add)
        if hasattr(op, 'get_identity'):
            merged_vals = op.get_identity((unique_ids.shape[0], out_dim), dtype=all_vals.dtype)
        else:
            merged_vals = jnp.zeros((unique_ids.shape[0], out_dim), dtype=all_vals.dtype)

        # SCATTER ADD
        # This sums up every value that lands on the same ID.
        # This handles collision/superposition for 1000 fields in one go.
        merged_vals = merged_vals.at[inverse_indices].add(all_vals)

        return unique_ids, merged_vals

    @staticmethod
    def compose_raw(ids_a: jnp.ndarray, vals_a: jnp.ndarray,
                    ids_b: jnp.ndarray, vals_b: jnp.ndarray,
                    op: 'FieldComposition') -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Robust Path: Merges two sparse fields.
        """
        # 1. Stack (Union)
        all_ids = jnp.concatenate([ids_a, ids_b], axis=0)
        all_vals = jnp.concatenate([vals_a, vals_b], axis=0)

        # 2. Unique (Topology / Collision Detection)
        # return_inverse gives us the indices to map old->new
        unique_ids, inverse_indices = jnp.unique(all_ids, return_inverse=True, axis=0)

        # 3. Aggregate
        # We need to reduce multiple values landing on the same ID.
        out_dim = vals_a.shape[1]

        # Initialize with Identity (0 for Add, 1 for Mul) if available
        if hasattr(op, 'get_identity'):
            merged_vals = op.get_identity((unique_ids.shape[0], out_dim), dtype=vals_a.dtype)
        else:
            merged_vals = jnp.zeros((unique_ids.shape[0], out_dim), dtype=vals_a.dtype)

        # 4. Perform Reduction
        # JAX's .at[].add() is optimized for scatter-add.
        # For custom ops, we might need a different approach, but for Physics,
        # Sparse Merge is almost always Accumulation (Add).

        # We assume for sparse merging, "Collision" means "Superposition" (Addition).
        # Even if the op is 'Multiply', physically merging two sparse lists
        # usually implies summing the contributions at the same point.
        # If strict multiplication is needed (intersection), logic would differ.

        merged_vals = merged_vals.at[inverse_indices].add(all_vals)

        return unique_ids, merged_vals

    # =========================================================
    # DISCRETE + DISCRETE
    # =========================================================
    @staticmethod
    def _compose_discrete_discrete(a: DiscreteFieldMapper,
                                   b: DiscreteFieldMapper,
                                   out_alg: IFieldAlgebra,
                                   op_name: str,
                                   final_bg=None):

        # 1. FAST PATH: Identity Check
        # If they share the exact same StateSpace instance, we skip alignment logic.
        if a.state_space is b.state_space:
            new_buffer = FieldSpaceComposer.compose_aligned_raw(
                a.explicit_buffer, b.explicit_buffer, out_alg, op_name
            )
            new_mask = a.mask_buffer | b.mask_buffer
            # Background composition
            new_bg = final_bg if final_bg else None

            return DiscreteFieldMapper(a.state_space, out_alg,
                                       explicit_buffer=new_buffer,
                                       mask_buffer=new_mask,
                                       bg_func=new_bg)

        # 2. SLOW PATH: Alignment Required
        # This uses Python loops. Only used during Setup/Init.

        # A. Create Union Space
        states_a = a.state_space.states
        states_b = b.state_space.states
        union_list = list(set(states_a).union(set(states_b)))

        # Create new dummy space
        # (Assuming StateSpace can handle list init, or use a.state_space.create_subset)
        try:
            new_space = a.state_space.create_subset(union_list)
        except:
            # Fallback for Abstract spaces
            from src.field_dynamic_system.core.state.discrete import AbstractDiscreteStateSpace
            new_space = AbstractDiscreteStateSpace(union_list)

        # B. Map Indices
        idx_map_a = {s: i for i, s in enumerate(states_a)}
        idx_map_b = {s: i for i, s in enumerate(states_b)}

        # C. Build New Buffer
        N = len(union_list)
        dim = a.explicit_buffer.shape[1]
        final_buffer = jnp.zeros((N, dim), dtype=a.explicit_buffer.dtype)

        # This loop is unavoidable for symbolic alignment
        for i, s in enumerate(union_list):
            val_a = out_alg.get_zero((1, dim))
            val_b = out_alg.get_zero((1, dim))

            if s in idx_map_a:
                val_a = a.explicit_buffer[idx_map_a[s]]
            elif a.background_func:
                # Implicit value
                val_a = a.get_fields_at(s)[0].value

            if s in idx_map_b:
                val_b = b.explicit_buffer[idx_map_b[s]]
            elif b.background_func:
                val_b = b.get_fields_at(s)[0].value

            if op_name == "add":
                res = out_alg.add(val_a, val_b)
            else:
                res = out_alg.mul(val_a, val_b)

            final_buffer = final_buffer.at[i].set(res)

        return DiscreteFieldMapper(new_space, out_alg, explicit_buffer=final_buffer)

    # =========================================================
    # CONTINUOUS (Simplified)
    # =========================================================
    @staticmethod
    def _compose_continuous_continuous(a, b, out_alg, op_name, final_bg=None):
        if final_bg:
            new_bg = final_bg
        else:
            def composed_bg(coords):
                v1 = a.get_raw_batch(coords)
                v2 = b.get_raw_batch(coords)
                if op_name == "add": return out_alg.add(v1, v2)
                return out_alg.mul(v1, v2)

            new_bg = composed_bg

        return ContinuousFieldMapper(a.state_space, out_alg, bg_func=new_bg)

    @staticmethod
    def _compose_hybrid(a, b, out_alg, op_name, final_bg=None):
        # Hybrid usually results in Continuous
        cont = a if isinstance(a, ContinuousFieldMapper) else b
        disc = b if isinstance(a, ContinuousFieldMapper) else a

        def hybrid_bg(coords):
            # Evaluate continuous
            v_cont = cont.get_raw_batch(coords)
            # Discrete is treated as background 0 unless we overlap?
            # Usually hybrid composition treats discrete as 'points' in continuous space.
            # For simplicity, we just return continuous + 0
            return v_cont

        return ContinuousFieldMapper(cont.state_space, out_alg, bg_func=hybrid_bg)