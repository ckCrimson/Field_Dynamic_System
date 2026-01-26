import jax
import jax.numpy as jnp
from jax.experimental import sparse
from typing import Optional, Any

# Interfaces
from src.field_dynamic_system.generator.generator_interfaces import IDiscreteFieldGenerator
from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper, IFieldMapper
from src.field_dynamic_system.core.field.field_space_operations import FieldSpaceTransformer


class DiscreteFieldGenerator(IDiscreteFieldGenerator):
    """
    The Concrete Physics Engine.
    Pipeline: F(t) -> Z^j (Extrinsic) -> Z^i (Intrinsic) -> [Flow: Chain(K, F_g)] -> F(t+1)
    """

    def __init__(self, topology, kernel, intrinsic_transform, extrinsic_transform,
                 intrinsic_composer, chain_composer, global_field_mapper=None):

        super().__init__(topology, kernel, intrinsic_transform, extrinsic_transform,
                         intrinsic_composer, chain_composer, global_field_mapper)

        # --- FIX: Initialize ALL Cache Attributes ---
        self._is_compiled = False
        self._compiled_jump_matrix = None  # M (Sparse Matrix for Linear Path)
        self._jump_matrix_values = None  # J (Raw Values for Generic Path)
        self._jump_matrix_indices = None  # [Source, Target] indices
        self._use_linear_flow = False  # Optimization Flag

    def precompute_matrix(self):
        """
        Builds the Flux Matrix (J).
        """
        # 1. TOPOLOGY
        adj = self.topology.adjacency_matrix
        raw_indices = adj.indices
        targets = raw_indices[:, 0]
        sources = raw_indices[:, 1]

        # Store for Generic Path
        self._jump_matrix_indices = jnp.stack([sources, targets], axis=1)

        # 2. KERNEL
        raw_k = self.kernel.compute_raw_batch(self._jump_matrix_indices, None)

        # 3. GLOBAL FIELD FUSE
        if self.F_g is not None:
            # Sync Size
            max_target = targets.max()
            if self.F_g.raw_buffer.shape[0] <= max_target:
                self.F_g.sync_size(int(max_target) + 1)

            vals_g = self.F_g.raw_buffer[targets]
            jump_vals = self.alpha_i.compose(raw_k, vals_g)
        else:
            jump_vals = raw_k

        # Cache the Raw Values (Required for Generic Path check)
        self._jump_matrix_values = jump_vals

        # 4. OPTIMIZATION CHECK
        ac_is_linear = getattr(self.alpha_c, 'is_linear_interaction', False)

        if ac_is_linear:
            self._use_linear_flow = True
            n_states = adj.shape[0]
            M_data = jump_vals.flatten()

            # Matrix A @ x indices: [Row=Target, Col=Source]
            self._compiled_jump_matrix = sparse.BCOO(
                (M_data, raw_indices),
                shape=(n_states, n_states)
            )
        else:
            self._use_linear_flow = False
            self._compiled_jump_matrix = None

        self._is_compiled = True

    def generate_multi_step(self,
                            current_mapper: IFieldMapper,
                            steps: int,
                            global_mapper: Optional[IFieldMapper] = None) -> IFieldMapper:

        # Ensure Compilation
        if not self._is_compiled:
            self.precompute_matrix()

        # 1. SETUP
        N = self.topology.adjacency_matrix.shape[0]
        current_mapper.sync_size(N)
        f_init = current_mapper.raw_buffer

        # 2. LOOP BODY
        def step_fn(carry_field, _):
            # A. Extrinsic (Time/Step Prep)
            f_step = FieldSpaceTransformer.apply_raw(carry_field, self.Z_j)

            # B. Intrinsic (Node Prep)
            f_star = FieldSpaceTransformer.apply_raw(f_step, self.Z_i)

            # C. Flow
            if self._use_linear_flow:
                # Fast Path
                f_next = self._compiled_jump_matrix @ f_star
            else:
                # Generic Path
                src_ids = self._jump_matrix_indices[:, 0]
                tgt_ids = self._jump_matrix_indices[:, 1]

                f_star_src = f_star[src_ids]
                flows = self.alpha_c.compose(f_star_src, self._jump_matrix_values)

                f_next = jnp.zeros_like(carry_field)
                f_next = f_next.at[tgt_ids].add(flows)

            return f_next, None

        # 3. EXECUTE
        final_field, _ = jax.lax.scan(step_fn, f_init, length=steps)

        # 4. RESULT
        result_mapper = DiscreteFieldMapper(current_mapper.state_space, current_mapper.algebra)
        result_mapper.sync_size(final_field.shape[0])
        result_mapper.apply_vector(final_field)

        return result_mapper