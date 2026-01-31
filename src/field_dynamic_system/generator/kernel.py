from typing import Any
import jax.numpy as jnp
from jax import vmap
from abc import ABC, abstractmethod

from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper


# ==========================================
# 1. THE ROOT CONTRACT (Indices Based)
# ==========================================
class AbstractTransitionKernel(ABC):
    """
    The minimal contract.
    Must accept 'edge_indices' because Topology-Based kernels (Uniform/Normalized)
    need access to the Graph Structure (node degrees), not just values.
    """

    @abstractmethod
    def compute_raw_batch(self,
                          edge_indices: jnp.ndarray,
                          context_mapper: 'DiscreteFieldMapper' ) -> jnp.ndarray:
        """
        Args:
            edge_indices: Shape (N_Edges, 2) -> [Source_ID, Target_ID]
            context_mapper: Access to the State Space and values.
        Returns:
            Weights: Shape (N_Edges, 1)
        """
        pass


# ==========================================
# 2. THE CONVENIENCE LAYER (Any Type Support)
# ==========================================
class ElementwiseKernel(AbstractTransitionKernel):
    """
    A kernel that handles the data fetching for you.
    Supports both JAX Arrays (Fast) and Arbitrary Python Objects (Generic).
    """

    def compute_raw_batch(self, edge_indices, context_mapper):
        # 1. RESOLVE INDICES TO STATES
        src_ids = edge_indices[:, 0]
        tgt_ids = edge_indices[:, 1]

        space = context_mapper.state_space

        # CHECK 1: Is the Space Vectorized? (Coordinates)
        if hasattr(space, 'get_matrix'):
            # --- FAST PATH (JAX) ---
            # The space is a matrix (e.g., Grid, Embedding). We use JAX Gather.
            all_states = jnp.array(space.get_matrix())
            src_batch = all_states[src_ids]
            tgt_batch = all_states[tgt_ids]

            # Vectorize the user logic
            vectorized_func = vmap(self.compute_transition_value)
            return vectorized_func(src_batch, tgt_batch)

        # CHECK 2: Is the Space Generic? (Python Objects)
        else:
            # --- GENERIC PATH (Python Loop) ---
            # The space is a list of objects (e.g., Strings, Custom Classes).
            # We must fetch manually.

            # Assuming space exposes a way to get state by index (usually a list)
            # If space.states is a list:
            all_states = list(space.states)

            # Gather (List Comprehension)
            # We convert JAX indices to Python ints for list indexing
            src_ids_py = src_ids.tolist()
            tgt_ids_py = tgt_ids.tolist()

            results = []
            for s_idx, t_idx in zip(src_ids_py, tgt_ids_py):
                state_a = all_states[s_idx]
                state_b = all_states[t_idx]

                # Call user logic on raw objects
                val = self.compute_transition_value(state_a, state_b)
                results.append(val)

            # Return as JAX array (The System requires numeric weights)
            return jnp.array(results).reshape(-1, 1)

    @abstractmethod
    def compute_transition_value(self, state_0: Any, state_out: Any) -> Any:
        """
        Define logic for ONE pair.
        Can receive Tuples, Strings, Objects, or Floats.
        Must return a float/scalar.
        """
        pass


# ==========================================
# 3. TOPOLOGY KERNELS (Require Indices)
# ==========================================

class UnbiasedKernel(AbstractTransitionKernel):
    """
    Assigns uniform probability 1.0.
    Does not care about state values.
    """

    def __init__(self, prob: float = 1.0):
        self.prob = prob

    def compute_raw_batch(self, edge_indices, context_mapper) -> jnp.ndarray:
        N = edge_indices.shape[0]
        return jnp.full((N, 1), self.prob, dtype=jnp.float32)


class UniformKernel(AbstractTransitionKernel):
    """
    Calculates 1/Degree based on topology structure.
    """

    def compute_raw_batch(self, edge_indices, context_mapper):
        sources = edge_indices[:, 0]

        # 1. Count Degrees
        # Note: We use bincount for efficiency on integer IDs
        max_id = jnp.max(sources) + 1
        degrees = jnp.bincount(sources, minlength=max_id)

        # 2. Map back to edges
        edge_degrees = degrees[sources]

        # 3. Prob = 1/Degree
        safe_degrees = jnp.where(edge_degrees == 0, 1.0, edge_degrees)
        probs = 1.0 / safe_degrees

        return probs.reshape(-1, 1)


# ==========================================
# 4. COORDINATE KERNEL (Fast JAX Only)
# ==========================================
class CoordinateKernel(AbstractTransitionKernel):
    """
    Specialized for Spatial Physics. Assumes states are numeric vectors.
    """

    def compute_raw_batch(self, edge_indices, context_mapper) -> jnp.ndarray:
        # Optimization: Direct Matrix Gather
        all_coords = jnp.array(context_mapper.state_space.get_matrix())

        src_ids = edge_indices[:, 0]
        tgt_ids = edge_indices[:, 1]

        source_states = all_coords[src_ids]
        target_states = all_coords[tgt_ids]

        return vmap(self.compute)(source_states, target_states)

    @abstractmethod
    def compute(self, source_state, target_state):
        pass