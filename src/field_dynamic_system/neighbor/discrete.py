
import jax.lax
from jax.experimental import sparse

from .interfaces import Topology
from ..core.state.interfaces import IDiscreteStateSpace

from abc import abstractmethod
from typing import Any, Optional, Sequence
import jax.numpy as jnp
import jax



# class DiscreteTopology(Topology):
#     """
#     Base class for Finite State connectivity.
#
#     USER RESPONSIBILITY:
#     - Implement 'compute_neighbors(state) -> List[State]'
#
#     FRAMEWORK RESPONSIBILITY:
#     - Automatically maps these connections to JAX Indices.
#     - Automatically builds the Adjacency Matrix for performance.
#     - Handles Multi-Step Reachability using JAX.
#     """
#
#     def __init__(self, state_space: IDiscreteStateSpace):
#         super().__init__(state_space)
#         self.discrete_space: IDiscreteStateSpace = state_space
#
#         # Cache for the matrix (Lazy Loading)
#         self._adjacency_matrix: Optional[jnp.ndarray] = None
#
#     # --- THE USER METHOD ---
#     @abstractmethod
#     def compute_neighbors(self, state: Any) -> Sequence[Any]:
#         """
#         User implements this logic.
#         Input: A single State object (e.g., VectorState(0,0) or 'Room A').
#         Output: A list/sequence of reachable State objects.
#         """
#         pass
#
#     # --- THE FRAMEWORK LOGIC (Automatic Matrix Building) ---
#     @property
#     def adjacency_matrix(self) -> jnp.ndarray:
#         """
#         Automatically calls the user's 'compute_neighbors' for every state
#         to build the high-performance matrix once.
#         """
#         if self._adjacency_matrix is None:
#             self._adjacency_matrix = self._auto_build_matrix()
#         return self._adjacency_matrix
#
#     def _auto_build_matrix(self) -> jnp.ndarray:
#         """Internal engine that converts User Logic -> JAX Matrix."""
#         print(f"Compiling Topology for {self.discrete_space.num_states} states...")
#
#         N = self.discrete_space.num_states
#         rows = []
#         cols = []
#
#         # 1. Iterate over every state in the universe
#         # We access the raw list of states from the space
#         all_states = self.discrete_space.states
#
#         for i, current_state in enumerate(all_states):
#             # 2. ASK THE USER: "Where can I go from here?"
#             # This is the pure Python logic you wrote
#             neighbors = self.compute_neighbors(current_state)
#
#             # 3. Convert User's Answer to Machine Indices
#             for neighbor in neighbors:
#                 # We use the Space's lookup to find the index
#                 target_idx = self.discrete_space.get_index_of(neighbor)
#
#                 # If the neighbor is valid (exists in the space), record the link
#                 if target_idx != -1:
#                     rows.append(i)
#                     cols.append(target_idx)
#
#         # 4. Build JAX Matrix
#         # (Defaulting to dense for compatibility, BCOO for scale)
#         adj = jnp.zeros((N, N), dtype=jnp.float32)
#         if rows:
#             adj = adj.at[jnp.array(rows), jnp.array(cols)].set(1.0)
#
#         return adj
#
#     # --- IMPLEMENTING ITopology CONTRACT ---
#
#     def successor(self, state: Any) -> IDiscreteStateSpace:
#         """
#         Returns the immediate reachable space.
#         Uses the User's Logic directly (no need for matrix lookup here).
#         """
#         neighbors = self.compute_neighbors(state)
#         return self.discrete_space.create_subset(list(neighbors))
#
#     def multi_step_successor(self, initial_state: Any, steps: int) -> IDiscreteStateSpace:
#         """
#         Uses the Auto-Built Matrix for blazing fast multi-step expansion.
#         (L^l logic)
#         """
#         # 1. Get Start Index
#         idx = self.discrete_space.get_index_of(initial_state)
#         if idx == -1:
#             return self.discrete_space.create_subset([])
#
#         # 2. Setup Wave Vector
#         N = self.discrete_space.num_states
#         x = jnp.zeros(N, dtype=jnp.float32)
#         x = x.at[idx].set(1.0)
#
#         # 3. Get the Matrix (Triggers Auto-Build if first time)
#         A = self.adjacency_matrix
#
#         # 4. Run JAX Kernel
#         # (Standard JAX scan loop)
#         def body_fun(i, current_x):
#             next_x = current_x @ A
#             return (next_x > 0.0).astype(jnp.float32)
#
#         final_x = jax.lax.fori_loop(0, steps, body_fun, x)
#
#         # 5. Result -> State Space
#         reached_indices = jnp.where(final_x > 0)[0]
#         return self._indices_to_space(reached_indices)
#
#     def _indices_to_space(self, indices: jnp.ndarray) -> IDiscreteStateSpace:
#         if indices.size == 0:
#             return self.discrete_space.create_subset([])
#
#         idx_list = indices.tolist()
#         all_states = self.discrete_space.states
#         subset_states = [all_states[i] for i in idx_list]
#         return self.discrete_space.create_subset(subset_states)


class DiscreteTopology(Topology):
    """
    Base class for Finite State connectivity.
    Now optimized with Sparse Matrices (BCOO) for O(N) memory usage.
    """

    def __init__(self, state_space: IDiscreteStateSpace):
        super().__init__(state_space)
        self.discrete_space: IDiscreteStateSpace = state_space
        # Type hint update: Matrix is now a Sparse Object
        self._adjacency_matrix: Optional[sparse.BCOO] = None

    @abstractmethod
    def compute_neighbors(self, state: Any) -> Sequence[Any]:
        """User implements this. Unchanged."""
        pass

    @property
    def adjacency_matrix(self) -> sparse.BCOO:
        """Returns the JAX BCOO Sparse Matrix."""
        if self._adjacency_matrix is None:
            self._adjacency_matrix = self._auto_build_matrix()
        return self._adjacency_matrix

    def _auto_build_matrix(self) -> sparse.BCOO:
        """
        Internal Engine: User Logic -> Sparse Matrix.
        """
        N = self.discrete_space.num_states
        print(f"Compiling Sparse Topology for {N} states...")

        rows = []
        cols = []

        # 1. Iterate states (Python Loop) - This part is unavoidable for generic logic
        all_states = self.discrete_space.states

        for i, current_state in enumerate(all_states):
            neighbors = self.compute_neighbors(current_state)

            for neighbor in neighbors:
                target_idx = self.discrete_space.get_index_of(neighbor)

                # Record valid connection
                if target_idx != -1:
                    rows.append(i)
                    cols.append(target_idx)

        # 2. Build Sparse Matrix (The Switch)
        if not rows:
            # Handle empty graph case safely
            return sparse.BCOO.fromdense(jnp.zeros((N, N), dtype=jnp.float32))

        # JAX Sparse construction
        # Indices must be (Num_Edges, 2)
        indices = jnp.column_stack((jnp.array(rows), jnp.array(cols)))
        values = jnp.ones(len(rows), dtype=jnp.float32)

        # Create BCOO Matrix
        # shape=(N, N) ensures it behaves like the dense matrix mathematically
        mat = sparse.BCOO((values, indices), shape=(N, N))

        # Consolidate duplicates (e.g. if logic returns same neighbor twice)
        mat = mat.sum_duplicates()
        return mat

    # --- IMPLEMENTING ITopology CONTRACT ---

    def successor(self, state: Any) -> IDiscreteStateSpace:
        """Same as before: Uses user logic directly."""
        neighbors = self.compute_neighbors(state)
        return self.discrete_space.create_subset(list(neighbors))

    def multi_step_successor(self, initial_state: Any, steps: int) -> IDiscreteStateSpace:
        """
        Multi-Step using Sparse Matrix Multiplication.
        """
        idx = self.discrete_space.get_index_of(initial_state)
        if idx == -1:
            return self.discrete_space.create_subset([])

        N = self.discrete_space.num_states

        # The 'Wave' vector x remains Dense (1D array)
        # Sparse (NxN) @ Dense (N) -> Dense (N) is very efficient
        x = jnp.zeros(N, dtype=jnp.float32)
        x = x.at[idx].set(1.0)

        # Get the Sparse Matrix
        A = self.adjacency_matrix

        # JAX Compiled Loop
        def body_fun(i, current_x):
            # Sparse Matrix-Vector Multiplication
            # NOTE: sparse.BCOO defines logic for '@' operator with dense vectors
            next_x = A.T @ current_x  # Transpose might be needed depending on row/col definition
            # Let's verify orientation:
            # We defined Rows=Source, Cols=Target.
            # Vector x is (1, N).
            # We want: x_new = x_old @ A
            # Since A is sparse, JAX often prefers (A @ x) column vectors.
            # Let's treat x as a column for the math: x_new = A.T @ x

            # Using 'A.T @ current_x' essentially propagates flow FROM rows TO cols.
            next_val = A.transpose() @ current_x

            return (next_val > 0.0).astype(jnp.float32)

        final_x = jax.lax.fori_loop(0, steps, body_fun, x)

        reached_indices = jnp.where(final_x > 0)[0]
        return self._indices_to_space(reached_indices)

    def _indices_to_space(self, indices: jnp.ndarray) -> IDiscreteStateSpace:
        if indices.size == 0:
            return self.discrete_space.create_subset([])

        idx_list = indices.tolist()
        all_states = self.discrete_space.states
        subset_states = [all_states[i] for i in idx_list]
        return self.discrete_space.create_subset(subset_states)