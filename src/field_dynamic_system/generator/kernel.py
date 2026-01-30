
import jax.numpy as jnp
from jax import vmap
from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import vmap
from abc import ABC, abstractmethod

from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper


# --- LEVEL 1: THE ROOT CONTRACT (For Performance Experts) ---
class AbstractTransitionKernel(ABC):
    """
    The minimal contract required by the generator.
    If you know how to write vectorized JAX code (matrix multiplication, etc.),
    inherit from this directly.
    """

    @abstractmethod
    def compute_raw_batch(self,
                          source_batch: jnp.ndarray,
                          target_batch: jnp.ndarray,
                          context_mapper: 'DiscreteFieldMapper') -> jnp.ndarray:
        """
        Must return shape (Batch_Size, Field_Dim).
        The user is responsible for ensuring this is efficient.
        """
        pass


# --- LEVEL 2: THE CONVENIENCE LAYER (For Physics Logic) ---
class ElementwiseKernel(AbstractTransitionKernel):
    """
    A helper class for users who want to define physics for a single pair
    and let the system handle the vectorization (vmap).
    """

    def compute_raw_batch(self, source_batch, target_batch, context_mapper):
        # We automatically vectorize the user's single-item logic
        # in_axes=(0, 0) means "iterate over the first dimension of both inputs"
        vectorized_func = vmap(self.compute_transition_value, in_axes=(0, 0))

        # We assume the user logic doesn't need the context_mapper inside the loop
        # (Pass necessary constants via __init__ if needed)
        return vectorized_func(source_batch, target_batch)

    @abstractmethod
    def compute_transition_value(self, state_0: jnp.ndarray, state_out: jnp.ndarray) -> jnp.ndarray:
        """
        Define logic for ONE pair.
        System calls this 1,000,000 times in parallel.
        """
        pass
# --- Example Implementation: Uniform Diffusion ---
class UniformKernel(AbstractTransitionKernel):
    def compute_raw_batch(self, edge_indices, context_mapper):
        # edge_indices shape: (N_edges, 2) -> [Source_ID, Target_ID]
        sources = edge_indices[:, 0]

        # 1. Calculate Degree (Count occurrences of each source)
        # We need to know how many times each source appears to normalize.
        # jnp.unique with return_counts is good, but for raw speed/simplicity
        # in this specific topology (degree is constant 2 everywhere except boundaries),
        # we can compute it properly.

        # Count outgoing edges for each source index
        # We assume indices are dense or mapped to 0..N-1

        # Get counts for every source ID involved in edges
        unique_src, counts = jnp.unique(sources, return_counts=True)

        # 2. Map counts back to edges
        # We need an array of size N_edges where each entry is the degree of that edge's source.
        # Strategy: Use a lookup array if IDs are small integers (which they are here).
        max_id = int(jnp.max(sources)) + 1
        degree_map = jnp.zeros(max_id, dtype=jnp.float32)
        degree_map = degree_map.at[unique_src].set(counts.astype(jnp.float32))

        degrees = degree_map[sources]  # Gather degrees for each edge

        # 3. Probability = 1.0 / Degree
        probs = 1.0 / degrees

        return probs.reshape((-1, 1))


# ==========================================
# 2. THE ABSTRACTION LAYER (Scientist Level)
# ==========================================
class CoordinateKernel(AbstractTransitionKernel):
    """
    The Scientist's Helper.
    Automatically handles the "Gather" operation.
    You just define the physics between two state vectors (Coordinate A -> Coordinate B).
    """

    def compute_raw_batch(self, edge_indices: jnp.ndarray, context_mapper) -> jnp.ndarray:
        # 1. ACCESS HARDWARE MEMORY
        # We assume the space has a matrix representation of states.
        # Shape: (Total_States, Dimensions)
        all_coords = jnp.array(context_mapper.discrete_space.get_matrix())

        # 2. GATHER VECTORS (The Boilerplate)
        src_ids = edge_indices[:, 0]
        tgt_ids = edge_indices[:, 1]

        # JAX instant lookup
        source_states = all_coords[src_ids]
        target_states = all_coords[tgt_ids]

        # 3. VECTORIZE USER LOGIC
        # We use vmap to turn your single-item logic into a batch processor
        vectorized_func = vmap(self.compute, in_axes=(0, 0))

        return vectorized_func(source_states, target_states)

    @abstractmethod
    def compute(self, source_state: jnp.ndarray, target_state: jnp.ndarray) -> jnp.ndarray:
        """
        Define your physics here for a SINGLE pair.

        Args:
            source_state: Vector (D,)
            target_state: Vector (D,)
        Returns:
            Scalar Weight or Vector Field
        """
        pass


# ==========================================
# 3. CONCRETE IMPLEMENTATIONS
# ==========================================

class UnbiasedKernel(AbstractTransitionKernel):
    """
    The Default Kernel.
    Assigns a uniform weight (probability) to every connection.
    Does NOT need to look up coordinates (Physics is identical everywhere).
    """

    def __init__(self, prob: float = 1.0):
        self.prob = prob

    def compute_raw_batch(self, source_batch, target_batch, context_mapper) -> jnp.ndarray:
        # No gather needed. Just return 1.0s.
        # source_batch is used just to know the size (N).
        N = source_batch.shape[0]

        # Return shape (N, 1)
        return jnp.full((N, 1), self.prob, dtype=jnp.float32)


class DistanceDecayKernel(CoordinateKernel):
    """
    Example of a Coordinate-Aware Kernel.
    Weight decays with distance: 1 / (d^2 + epsilon)
    """

    def compute(self, source_state, target_state):
        # Euclidean Distance
        delta = target_state - source_state
        dist_sq = jnp.sum(delta ** 2)

        # Inverse Square Law
        return (1.0 / (dist_sq + 1e-6)).reshape(1)