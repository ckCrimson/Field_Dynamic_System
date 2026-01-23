
import jax.lax
from jax.experimental import sparse

from .interfaces import Topology
from ..core.state.interfaces import IDiscreteStateSpace

from abc import abstractmethod
from typing import Any, List, Tuple, Optional, Callable, Sequence
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

        # =========================================================
        # 1. RAW DATA KERNELS (The Fast Path)
        # =========================================================

    def get_raw_successor(self, state_indices: jnp.ndarray) -> jnp.ndarray:
            """
            Input:  Indices of current states [i, k, ...]
            Output: Indices of ALL immediate neighbors (Unique).
            """
            # Convert indices to a "Wavefront" vector
            N = self.discrete_space.num_states
            x = jnp.zeros(N, dtype=jnp.float32)

            # Mark current states as active (1.0)
            x = x.at[state_indices].set(1.0)

            # Matrix Multiply: (1, N) @ (N, N) -> (1, N)
            # We use A.T because we store Rows=Source, Cols=Target
            # Flow: Source -> Target
            A = self.adjacency_matrix

            # Note: BCOO matmul logic:
            # A @ v -> Standard transform
            # If A[i, j] = 1 implies i->j.
            # If x has 1 at i. x @ A -> result has 1 at j.
            # JAX sparse usually assumes A @ x.
            # Let's check dimensions: (N, N) @ (N, 1) -> (N, 1).
            # So we treat x as column vector.

            next_x = A.T @ x  # Or A.transpose() @ x depending on JAX version

            # Return indices where value > 0
            return jnp.where(next_x > 0)[0]

    def get_raw_predecessor(self, state_indices: jnp.ndarray) -> jnp.ndarray:
            """
            Input: Indices of target states.
            Output: Indices of states that can REACH the targets.
            """
            # Predecessors are just Successors on the Transposed Graph.
            # If A[i, j] = 1 (i->j), then A.T[j, i] = 1 (j<-i).

            N = self.discrete_space.num_states
            x = jnp.zeros(N, dtype=jnp.float32)
            x = x.at[state_indices].set(1.0)

            A = self.adjacency_matrix

            # Flow Backwards: Use A instead of A.T (or vice versa depending on definition)
            # If A.T moves forward (Source->Target), then A moves backward (Target->Source).
            prev_x = A @ x

            return jnp.where(prev_x > 0)[0]

    def get_raw_multi_step_successor(self, initial_indices: jnp.ndarray, steps: int) -> jnp.ndarray:
            """
            Input: Indices of start states.
            Output: Indices of states reachable after 'steps' transitions.
            """
            if steps == 0: return initial_indices

            N = self.discrete_space.num_states
            x = jnp.zeros(N, dtype=jnp.float32)
            x = x.at[initial_indices].set(1.0)

            A = self.adjacency_matrix

            # JIT-compiled loop
            def body_fun(i, current_x):
                # Propagate flow
                next_val = A.T @ current_x
                # Binarize to prevent explosion/overflow, keep it as "Reachable Mask"
                return (next_val > 0.0).astype(jnp.float32)

            final_x = jax.lax.fori_loop(0, steps, body_fun, x)

            return jnp.where(final_x > 0)[0]

        # =========================================================
        # 2. OBJECT LAYER (The User Interface)
        # =========================================================


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



# ==============================================================================
# 1. GRAPH TOPOLOGY (The "Lookup" Engine)
# ==============================================================================
class GraphTopology(DiscreteTopology):
    """
    Connections defined by an explicit list of edges.
    Fastest build time for arbitrary graphs.
    """

    def __init__(self, state_space: IDiscreteStateSpace, edges: List[Tuple[Any, Any]], directed: bool = True):
        # We process edges immediately to build the matrix
        self.raw_edges = edges
        self.directed = directed
        super().__init__(state_space)

    def compute_neighbors(self, state: Any) -> List[Any]:
        # Fallback (slow), usually we use the matrix
        neighbors = []
        for u, v in self.raw_edges:
            if u == state:
                neighbors.append(v)
            elif not self.directed and v == state:
                neighbors.append(u)
        return neighbors

    def _auto_build_matrix(self) -> sparse.BCOO:
        N = self.discrete_space.num_states
        print(f"Building Graph Topology from {len(self.raw_edges)} edges...")

        rows, cols = [], []

        # Optimization: Build index map once
        # (Assumes states are hashable. If not, we need a slower lookup)
        try:
            state_to_idx = {s: i for i, s in enumerate(self.discrete_space.states)}
            can_hash = True
        except TypeError:
            can_hash = False

        for u, v in self.raw_edges:
            if can_hash:
                u_idx = state_to_idx.get(u, -1)
                v_idx = state_to_idx.get(v, -1)
            else:
                u_idx = self.discrete_space.get_index_of(u)
                v_idx = self.discrete_space.get_index_of(v)

            if u_idx != -1 and v_idx != -1:
                rows.append(u_idx)
                cols.append(v_idx)
                if not self.directed:
                    rows.append(v_idx)
                    cols.append(u_idx)

        if not rows: return sparse.BCOO.fromdense(jnp.zeros((N, N), dtype=jnp.float32))

        indices = jnp.column_stack((jnp.array(rows), jnp.array(cols)))
        values = jnp.ones(len(rows), dtype=jnp.float32)
        return sparse.BCOO((values, indices), shape=(N, N)).sum_duplicates()


# ==============================================================================
# 2. DELTA TOPOLOGY (The "Vector Addition" Engine)
# ==============================================================================
class DeltaTopology(DiscreteTopology):
    """
    Connections defined by relative offsets (Deltas).
    State B is reachable from A if B = A + Delta.

    Excellent for Grids, Lattices, and Uniform Structures.
    """

    def __init__(self, state_space: IDiscreteStateSpace, deltas: List[Any]):
        super().__init__(state_space)
        self.deltas = deltas

    def compute_neighbors(self, state: Any) -> List[Any]:
        neighbors = []
        for d in self.deltas:
            try:
                # This now calls VectorState.__add__
                possible_neighbor = state + d

                # CRITICAL: We must check if this new state actually exists
                # in the Universe (StateSpace).
                # If the robot walks off the grid, 'possible_neighbor'
                # is valid math, but invalid for the Space.

                # get_index_of returns -1 if not found.
                if self.discrete_space.get_index_of(possible_neighbor) != -1:
                    neighbors.append(possible_neighbor)

            except TypeError:
                pass
        return neighbors

    def _auto_build_matrix(self) -> sparse.BCOO:
        """
        Optimized 'Shift' Builder.
        """
        N = self.discrete_space.num_states
        print(f"Building Delta Topology with {len(self.deltas)} moves...")

        # 1. Convert all states to a Matrix (N, D)
        # We assume VectorStateSpace for maximum speed
        try:
            # Extract raw values from states
            all_states = jnp.array([s.values for s in self.discrete_space.states])
            is_vector = True
        except:
            is_vector = False
            # Fallback to slow Python loop if states aren't simple vectors
            return super()._auto_build_matrix()

        # 2. Convert Deltas to Matrix (K, D)
        deltas_arr = jnp.array([d.values if hasattr(d, 'values') else d for d in self.deltas])

        # 3. KD-Tree or Broadcast Search
        # For perfect grids, we can use exact matching.
        # Strategy: Broadcast (N, 1, D) + (1, K, D) -> (N, K, D) (Candidate Neighbors)
        # Then find which Candidates exist in the original set.

        # NOTE: For massive spaces, finding "Does vector V exist in Set S" is the bottleneck.
        # We use a hash map on CPU for O(1) lookup.

        # Map: Vector Tuple -> Index
        # Rounding needed to avoid float errors
        state_map = {tuple(map(lambda x: round(float(x), 5), s)): i
                     for i, s in enumerate(all_states)}

        rows, cols = [], []

        # CPU Loop (Faster than GPU search for sparse connections)
        for i in range(N):
            current_vec = all_states[i]

            for k in range(len(deltas_arr)):
                target_vec = current_vec + deltas_arr[k]

                # Lookup
                key = tuple(map(lambda x: round(float(x), 5), target_vec))
                if key in state_map:
                    target_idx = state_map[key]
                    rows.append(i)
                    cols.append(target_idx)

        indices = jnp.column_stack((jnp.array(rows), jnp.array(cols)))
        values = jnp.ones(len(rows), dtype=jnp.float32)
        return sparse.BCOO((values, indices), shape=(N, N)).sum_duplicates()


# ==============================================================================
# 3. METRIC DISCRETE TOPOLOGY (The "Distance" Engine)
# ==============================================================================
class MetricDiscreteTopology(DiscreteTopology):
    """
    Consolidated class for all Distance-based connections.

    - Standard Distance: min_r=0, max_r=R
    - Ring/Shell:        min_r=R1, max_r=R2
    - Exact Edge:        min_r=R-e, max_r=R+e
    """

    def __init__(self, state_space: IDiscreteStateSpace,
                 max_dist: float,
                 min_dist: float = 0.0,
                 distance_fn: Optional[Callable] = None):

        super().__init__(state_space)
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.dist_fn = distance_fn  # Optional custom metric

    def compute_neighbors(self, state: Any) -> List[Any]:
        """
        Calculates neighbors for a SINGLE state on CPU.
        """
        # 1. Get the vector for the query state
        if hasattr(state, 'values'):
            query_vec = jnp.array(state.values)
        else:
            query_vec = jnp.array(state)

        neighbors = []

        # 2. Iterate through all states in the space
        # (For 100k states, this is slow, but for 100 it's instant)
        for candidate in self.discrete_space.states:
            if hasattr(candidate, 'values'):
                cand_vec = jnp.array(candidate.values)
            else:
                cand_vec = jnp.array(candidate)

            # 3. Calculate Distance
            dist = float(jnp.linalg.norm(query_vec - cand_vec))

            # 4. Check Thresholds
            if self.min_dist <= dist <= self.max_dist:
                # If min_dist > 0, we implicitly exclude the state itself (dist=0)
                # unless the user explicitly requested min_dist=0
                if dist == 0.0 and self.min_dist > 0:
                    continue
                neighbors.append(candidate)

        return neighbors

    def _auto_build_matrix(self) -> sparse.BCOO:
        """
        Optimized Pairwise Distance Builder.
        O(N^2) complexity, but fully parallelized on GPU.
        """
        N = self.discrete_space.num_states
        print(f"Building Metric Topology (N={N}, Range=[{self.min_dist}, {self.max_dist}])...")

        # 1. Get State Matrix (N, D)
        try:
            states_matrix = jnp.array([s.values for s in self.discrete_space.states])
        except:
            print("Warning: MetricTopology requires VectorStates for optimization.")
            return super()._auto_build_matrix()

        # 2. Compute Distance Matrix (N, N) - Dense!
        # D_ij = ||x_i - x_j||
        # Using JAX vmap or broadcasting

        # (N, 1, D) - (1, N, D) -> (N, N, D)
        diff = states_matrix[:, None, :] - states_matrix[None, :, :]
        dist_matrix = jnp.linalg.norm(diff, axis=-1)

        # 3. Apply Thresholds (Boolean Mask)
        # min <= dist <= max
        # Also exclude self-loops (dist > 0) usually, unless min_dist=0 includes self.
        mask = (dist_matrix <= self.max_dist) & (dist_matrix >= self.min_dist)

        # Remove diagonal if purely looking for neighbors (optional)
        if self.min_dist > 0:
            pass  # Diagonal already removed because dist(i,i)=0
        else:
            # If min=0, we might want to keep self-loops or remove them based on convention.
            # Let's remove self-loops for "neighbors".
            mask = mask.at[jnp.diag_indices(N)].set(False)

        # 4. Convert Dense Mask -> Sparse Matrix
        # extracting indices where mask is True
        rows, cols = jnp.where(mask)

        if rows.shape[0] == 0:
            return sparse.BCOO.fromdense(jnp.zeros((N, N), dtype=jnp.float32))

        indices = jnp.column_stack((rows, cols))
        values = jnp.ones(rows.shape[0], dtype=jnp.float32)

        return sparse.BCOO((values, indices), shape=(N, N))