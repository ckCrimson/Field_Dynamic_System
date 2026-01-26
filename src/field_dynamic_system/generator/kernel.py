from abc import abstractmethod, ABC
from typing import Any
import jax.numpy as jnp


class AbstractTransitionKernel(ABC):
    """
    Defines the Physics of Interaction: K(s_target | s_source).
    This is stateless logic (e.g., 'Inverse Square Law', 'Gaussian').
    """

    # 1. High-Level (Object) - For Single Step Debugging / Continuous

    def compute_transition_value(self, state_initial: Any, state_target: Any) -> Any:
        """ Returns the raw Field Value (Tensor/Float), NOT a Mapper. """
        pass

    # 2. Low-Level (Raw) - For Discrete Batch Processing
    @abstractmethod
    def compute_raw_batch(self,
                          source_ids: jnp.ndarray,
                          target_ids: jnp.ndarray
                          ) -> jnp.ndarray:
        """
        Vectorized calculation for thousands of edges at once.
        Returns tensor of shape (N_edges, Field_Dim).
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