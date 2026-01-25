
import jax.lax
import numpy as np
from jax.experimental import sparse

from .interfaces import Topology
from ..core import AbstractState
from ..core.state.interfaces import IDiscreteStateSpace
import scipy.sparse as sp  # <--- CRITICAL: Import Scipy as 'sp' to avoid conflict
from abc import abstractmethod
from typing import Any, List, Tuple, Optional, Callable, Sequence, Dict, Set
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
    def __init__(self, state_space: IDiscreteStateSpace):
        super().__init__(state_space)
        self.discrete_space: IDiscreteStateSpace = state_space

        # --- TRACK A: Legacy Object Layer ---
        # existing code using JAX sparse
        self._adjacency_matrix: Optional[jax.experimental.sparse.BCOO] = None

        # --- TRACK B: Parallel Raw Implementation ---
        self._raw_to_id: Dict[Any, int] = {}
        self._id_to_raw: List[Any] = []

        # 2. Lazy Matrix: Mutable (CPU)
        # ERROR WAS HERE: You had 'sparse.lil_matrix' (which referred to JAX)
        # FIX: Use 'sp.lil_matrix' (referring to Scipy)
        self._raw_cpu_matrix = sp.lil_matrix((0, 0), dtype=np.float32)

        self._raw_jax_matrix: Optional[jax.experimental.sparse.BCOO] = None
        self._raw_explored: Set[int] = set()
        self._raw_dirty = False

    # =========================================================
    # 1. RAW DATA KERNELS (The Parallel Fast Path)
    # =========================================================

    def get_raw_successor(self, state_data_batch: Sequence[Any]) -> Sequence[Any]:
        """
        Input: Batch of Primitive Data.
        Output: Batch of Neighbor Primitive Data.

        Uses a completely separate, lazy registry/matrix.
        Does NOT use self.discrete_space or self.adjacency_matrix.
        """
        # A. Resolve Inputs to Local IDs (Registering if new)
        input_ids = self._raw_register_batch(state_data_batch)

        # B. Ensure Frontier (Lazy Expansion)
        # Check if we have computed neighbors for these IDs
        self._raw_ensure_expanded(input_ids)

        # C. Sync & Execute (JAX)
        if self._raw_dirty or self._raw_jax_matrix is None:
            self._raw_sync_to_device()

        A = self._raw_jax_matrix
        N = A.shape[0]

        # Wavefront Setup
        x = jnp.zeros(N, dtype=jnp.float32)
        idx_array = jnp.array(input_ids, dtype=jnp.int32)
        x = x.at[idx_array].set(1.0)

        # Matrix Multiply
        next_x = A.T @ x

        # Result Extraction
        next_indices = jnp.where(next_x > 0)[0]
        next_indices_list = np.array(next_indices).tolist()

        # D. Map back to Data
        return [self._id_to_raw[i] for i in next_indices_list]

    # --- Internal Helpers for Raw Track ---

    def _raw_register_batch(self, data_batch: Sequence[Any]) -> List[int]:
        """Maps data to Local IDs. Updates registry if new."""
        ids = []
        for data in data_batch:
            if data not in self._raw_to_id:
                new_id = len(self._id_to_raw)
                self._raw_to_id[data] = new_id
                self._id_to_raw.append(data)
            ids.append(self._raw_to_id[data])
        return ids

    def _raw_ensure_expanded(self, ids: List[int]):
        """
        OPTIMIZED: Uses Batch List Collection to avoid Scipy overhead.
        """
        unknown = [i for i in ids if i not in self._raw_explored]
        if not unknown:
            return

        # 1. BATCH COLLECTION (Pure Python Lists are fast)
        new_rows = []
        new_cols = []

        # Track max_id locally to avoid resizing in loop
        current_max_id = self._raw_cpu_matrix.shape[0] - 1

        for src_id in unknown:
            data = self._id_to_raw[src_id]
            neighbors = self.compute_neighbors(data)

            if not neighbors:
                self._raw_explored.add(src_id)
                continue

            dst_ids = self._raw_register_batch(neighbors)

            # Optimization: Just append to lists, don't touch Matrix yet
            count = len(dst_ids)
            new_rows.extend([src_id] * count)
            new_cols.extend(dst_ids)

            # Track size
            local_max = max(dst_ids) if dst_ids else src_id
            local_max = max(local_max, src_id)
            if local_max > current_max_id:
                current_max_id = local_max

            self._raw_explored.add(src_id)

        if not new_rows:
            return

        # 2. BULK RESIZE (Once)
        if current_max_id >= self._raw_cpu_matrix.shape[0]:
            # Grow strategy
            new_size = max(current_max_id + 1, int(self._raw_cpu_matrix.shape[0] * 1.5))
            self._raw_cpu_matrix.resize((new_size, new_size))

        # 3. BULK WRITE (Once)
        # This is vectorized and extremely fast compared to loop
        self._raw_cpu_matrix[new_rows, new_cols] = 1.0

        self._raw_dirty = True

    def _raw_sync_to_device(self):
        """Converts CPU LIL -> JAX BCOO."""
        if self._raw_cpu_matrix.shape == (0, 0):
            # Safe default
            self._raw_jax_matrix = jax.experimental.sparse.BCOO.fromdense(jnp.zeros((0, 0)))
        else:
            coo = self._raw_cpu_matrix.tocoo()
            indices = jnp.array(np.vstack((coo.row, coo.col)).T)
            values = jnp.array(coo.data)
            shape = coo.shape

            self._raw_jax_matrix = jax.experimental.sparse.BCOO(
                (values, indices), shape=shape
            )
        self._raw_dirty = False

    def expand_frontier(self, initial_states: Sequence[Any], depth: int):
        """
        SCOUT: Aggressively discovers the topology 'depth' steps ahead.
        This ensures the Matrix contains all possible reachable states
        before you run the heavy JAX simulation.
        """
        current_frontier = initial_states

        # We assume the user wants to expand from these states
        # First, ensure the start points are registered
        ids = self._raw_register_batch(current_frontier)

        for i in range(depth):
            # 1. Expand the current layer
            # This calls your _raw_ensure_expanded logic (which we optimized with batching)
            # It updates _raw_cpu_matrix and _raw_explored efficiently.
            self._raw_ensure_expanded(ids)

            # 2. Get the next layer of IDs (Pure connectivity check)
            # We look at the Scipy matrix rows we just built to find the next IDs.
            # This is much faster than re-computing neighbors.

            next_ids = []
            for src_id in ids:
                # Efficient Scipy lookup: Get all non-zero columns for this row
                # rows is [src_id], data is [1.0, 1.0...]
                _, cols = self._raw_cpu_matrix[src_id].nonzero()
                next_ids.extend(cols)

            # Deduplicate for the next step
            if not next_ids:
                break

            ids = list(set(next_ids))

    # =========================================================
    # 2. OBJECT LAYER (The User Interface - UNTOUCHED)
    # =========================================================

    @property
    def adjacency_matrix(self) -> jax.experimental.sparse.BCOO:
        """Returns the JAX BCOO Sparse Matrix."""
        if self._adjacency_matrix is None:
            self._adjacency_matrix = self._auto_build_matrix()
        return self._adjacency_matrix

    def _auto_build_matrix(self) -> jax.experimental.sparse.BCOO:
        """
        Internal Engine: User Logic -> Sparse Matrix.
        """
        N = self.discrete_space.num_states
        print(f"Compiling Sparse Topology for {N} states...")

        rows = []
        cols = []

        # 1. Iterate states (Python Loop)
        all_states = self.discrete_space.states

        for i, current_state in enumerate(all_states):
            neighbors = self.compute_neighbors(current_state)

            for neighbor in neighbors:
                target_idx = self.discrete_space.get_index_of(neighbor)

                # Record valid connection
                if target_idx != -1:
                    rows.append(i)
                    cols.append(target_idx)

        # 2. Build Sparse Matrix
        if not rows:
            return jax.experimental.sparse.BCOO.fromdense(jnp.zeros((N, N), dtype=jnp.float32))

        indices = jnp.column_stack((jnp.array(rows), jnp.array(cols)))
        values = jnp.ones(len(rows), dtype=jnp.float32)

        mat = jax.experimental.sparse.BCOO((values, indices), shape=(N, N))
        mat = mat.sum_duplicates()
        return mat

    # --- IMPLEMENTING ITopology CONTRACT (UNTOUCHED) ---


    @abstractmethod
    def compute_neighbors(self, state_val: Any) -> Sequence[Any]:
        """
        User implements this.
        RECOMMENDATION: This should return PRIMITIVE values (int, str, tuple),
        not State Objects, to ensure the Raw Track remains fast.
        """
        pass

    def create_state_object(self, state_val: Any) -> Any:
        """
        Factory Method: Converts a raw state value into a State Object.

        Default: Wraps the value in an AbstractState.
        Override: If you use custom State classes (e.g. CellState(x, y, type)).
        """
        # We perform a lazy import to avoid circular dependency issues at module level
        return AbstractState(state_val)

    def successor(self, state: Any) -> IDiscreteStateSpace:
        """
        Returns a StateSpace subset containing the neighbors.
        BRIDGES THE GAP: Raw Values -> State Objects.
        """
        # 1. Compute Raw Neighbors (Fast Logic)
        # e.g. Returns ["10", "11"]
        raw_neighbors = self.compute_neighbors(state)

        # 2. Convert to Objects (Factory Logic)
        # e.g. Returns [AbstractState("10"), AbstractState("11")]
        object_neighbors = [self.create_state_object(val) for val in raw_neighbors]

        # 3. Create Subset (Object Logic)
        return self.discrete_space.create_subset(object_neighbors)

    def multi_step_successor(self, initial_state: Any, steps: int) -> IDiscreteStateSpace:
        # Same as before...
        # ... logic omitted for brevity (uses matrix multiply) ...
        # The return uses _indices_to_space which effectively grabs existing objects
        # from the space, so no new factory call is needed there.
        idx = self.discrete_space.get_index_of(initial_state)
        if idx == -1:
            return self.discrete_space.create_subset([])

        N = self.discrete_space.num_states
        x = jnp.zeros(N, dtype=jnp.float32)
        x = x.at[idx].set(1.0)

        A = self.adjacency_matrix

        def body_fun(i, current_x):
            next_x = A.T @ current_x
            return (next_x > 0.0).astype(jnp.float32)

        final_x = jax.lax.fori_loop(0, steps, body_fun, x)
        reached_indices = jnp.where(final_x > 0)[0]

        return self._indices_to_space(reached_indices)

    def _indices_to_space(self, indices: jnp.ndarray) -> IDiscreteStateSpace:
        """
        Helper: Converts JAX indices (from matrix math) back into a StateSpace subset.
        """
        # 1. Handle Empty Result
        if indices.size == 0:
            return self.discrete_space.create_subset([])

        # 2. Convert JAX Array -> Python List
        idx_list = indices.tolist()

        # 3. Lookup State Objects
        # We access the master list of states from the space
        all_states = self.discrete_space.states
        subset_states = [all_states[i] for i in idx_list]

        # 4. Return new Subset
        return self.discrete_space.create_subset(subset_states)



# ==============================================================================
# 1. GRAPH TOPOLOGY (The "Lookup" Engine)
# ==============================================================================
class GraphTopology(DiscreteTopology):
    """
    Connections defined by an explicit list of edges.
    OPTIMIZED: Pre-fills the Raw Matrix for instant access.
    """

    def __init__(self, state_space: IDiscreteStateSpace, edges: List[Tuple[Any, Any]], directed: bool = True):
        super().__init__(state_space)
        self.directed = directed
        self.raw_edges = edges

        # --- NEW: Pre-load the Raw Track ---
        # We register all edges immediately so JAX can use them instantly.
        self._preload_graph()

    def _preload_graph(self):
        print(f"Pre-loading {len(self.raw_edges)} edges into Raw Topology...")

        # 1. Collect all unique nodes from edges
        # (We don't care about state_space, we trust the edges)
        all_nodes = set()
        for u, v in self.raw_edges:
            all_nodes.add(u)
            all_nodes.add(v)

        # 2. Bulk Register to get IDs
        # This populates _raw_to_id and _id_to_raw
        self._raw_register_batch(list(all_nodes))

        # 3. Build Matrix Indices
        rows = []
        cols = []

        for u, v in self.raw_edges:
            u_id = self._raw_to_id[u]
            v_id = self._raw_to_id[v]

            rows.append(u_id)
            cols.append(v_id)
            if not self.directed:
                rows.append(v_id)
                cols.append(u_id)

        # 4. Resize and Fill CPU Matrix
        max_id = len(self._id_to_raw)
        self._raw_cpu_matrix.resize((max_id, max_id))
        self._raw_cpu_matrix[rows, cols] = 1.0

        # Mark all as explored so we don't try to re-compute them
        self._raw_explored.update(self._raw_to_id.values())
        self._raw_dirty = True

    def compute_neighbors(self, state_val: Any) -> Sequence[Any]:
        # Fallback for dynamic/new nodes not in the initial edge list
        # (Rare for GraphTopology, but required by interface)
        neighbors = []
        for u, v in self.raw_edges:
            if u == state_val:
                neighbors.append(v)
            elif not self.directed and v == state_val:
                neighbors.append(u)
        return neighbors


# ==============================================================================
# 2. DELTA TOPOLOGY (The "Vector Addition" Engine)
# ==============================================================================
class DeltaTopology(DiscreteTopology):
    """
    Infinite Grid / Lattice Engine.
    Math-based neighbors. No boundaries.
    """

    def __init__(self, state_space: IDiscreteStateSpace, deltas: Sequence[Any]):
        super().__init__(state_space)
        self.deltas = deltas

    def compute_neighbors(self, state_val: Any) -> Sequence[Any]:
        """
        Pure Math. Returns neighbors regardless of whether they 'exist' yet.
        This enables Infinite Expansion.
        """
        neighbors = []

        # Handle Vectors (Tuple/Array) vs Scalars
        # We assume state_val is a primitive (e.g. (0,0) or 10)

        try:
            # OPTIMIZED: Vectorized addition if possible
            # If state_val is a tuple, we iterate.
            if isinstance(state_val, (tuple, list)):
                base = np.array(state_val)
                for d in self.deltas:
                    # Result is a new coordinate
                    res = base + np.array(d)
                    # Convert back to tuple for hashing
                    neighbors.append(tuple(res.tolist()))
            else:
                # Scalar addition
                for d in self.deltas:
                    neighbors.append(state_val + d)

        except Exception as e:
            # Fallback for complex objects
            print(f"Topology Math Error: {e}")
            pass

        return neighbors
# ==============================================================================
# 3. METRIC DISCRETE TOPOLOGY (The "Distance" Engine)
# ==============================================================================
class MetricDiscreteTopology(DiscreteTopology):
    """
    Dynamic Radius Search.
    Finds neighbors among 'Known States' that are close.
    """

    def __init__(self, state_space: IDiscreteStateSpace, max_dist: float, distance_fn=None):
        super().__init__(state_space)
        self.max_dist = max_dist
        self.dist_fn = distance_fn

    def compute_neighbors(self, state_val: Any) -> Sequence[Any]:
        """
        Scans ONLY the raw frontier to find connections.
        Note: This is O(N) where N is 'Explored States'.
        For massive spaces, DeltaTopology is preferred.
        """
        neighbors = []

        # 1. Convert Query to Array
        q = np.array(state_val)

        # 2. Compare against ALL known raw states (The "Memory")
        # Optimization: In a real system, use a KD-Tree here.
        # For now, we linear scan the registry.

        known_states = self._id_to_raw

        for candidate in known_states:
            if candidate == state_val: continue

            # Simple Euclidean
            c = np.array(candidate)
            dist = np.linalg.norm(q - c)

            if dist <= self.max_dist:
                neighbors.append(candidate)

        return neighbors