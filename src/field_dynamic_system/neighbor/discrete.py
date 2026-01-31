import jax
import jax.numpy as jnp
from jax.experimental import sparse
import numpy as np
import scipy.sparse as sp
from typing import Any, Sequence, Optional, Dict, List, Set, Union, Tuple
from abc import ABC, abstractmethod

from src.field_dynamic_system.core import VectorState
from src.field_dynamic_system.core.state import IDiscreteStateSpace
from src.field_dynamic_system.neighbor.interfaces import Topology


class DiscreteTopology(Topology, ABC):
    def __init__(self, state_space: IDiscreteStateSpace):
        super().__init__(state_space)
        self.discrete_space = state_space
        self._adjacency_matrix = None

        # HYBRID STORAGE: Fast Path uses Arrays, Slow Path uses Lists
        self._raw_to_id = {}
        self._id_to_raw = []

        # FAST PATH CACHE (Raw Integer Storage)
        self._fast_coords: Optional[np.ndarray] = None
        self._fast_path_active = False

        self._raw_cpu_matrix = sp.lil_matrix((0, 0), dtype=np.float32)
        self._raw_jax_matrix = None
        self._raw_explored = set()
        self._raw_dirty = False

    @abstractmethod
    def compute_neighbors(self, state_val: Any) -> Sequence[Any]:
        pass

        # Inside DiscreteTopology class...

    def get_adjacency_matrix(self, kernel=None, context_mapper=None):
            """
            Alias to satisfy the GenericMarkovianGenerator interface.
            It simply returns the cached matrix you already built.
            """
            # We ignore 'kernel' because DiscreteTopology builds its own structure
            # (based on grids/graphs) independent of dynamic physics.
            return self.adjacency_matrix

    @property
    def adjacency_matrix(self) -> sparse.BCOO:
        if self._adjacency_matrix is None:
            self._adjacency_matrix = self._build_adjacency_matrix()
        return self._adjacency_matrix

    def _build_adjacency_matrix(self) -> sparse.BCOO:
        if hasattr(self, 'deltas') and hasattr(self.discrete_space, 'get_matrix'):
            try:
                return self._build_fast_vector_matrix()
            except Exception:
                pass
        return self._build_standard_matrix()

    def _build_fast_vector_matrix(self) -> sparse.BCOO:
        # Use Cached Coords if available (Discovery Path)
        if self._fast_coords is not None:
            raw_coords = self._fast_coords
        else:
            # Fallback to Space (Static Path)
            if not hasattr(self.discrete_space, 'get_matrix'): raise ValueError("No matrix data")
            raw_coords = np.array(self.discrete_space.get_matrix(), dtype=np.int64)

        N, D = raw_coords.shape
        if N > 10000: print(f"⚡ DiscreteTopology: FAST BUILD ({N:,} states)...")

        if N == 0: return sparse.BCOO.fromdense(jnp.zeros((0, 0), dtype=jnp.float32))

        # Vectorized Neighbor Search
        bounds_max = int(np.max(raw_coords)) + 2
        strides = np.array([bounds_max ** i for i in range(D)], dtype=np.int64)
        encoded_states = raw_coords.dot(strides)
        sort_perm = np.argsort(encoded_states)
        sorted_encoded = encoded_states[sort_perm]

        sources, targets = [], []
        deltas = np.array(self.deltas, dtype=np.int64)

        for delta in deltas:
            delta_val = np.dot(delta, strides)
            potential = sorted_encoded + delta_val
            pos = np.searchsorted(sorted_encoded, potential)
            pos = np.clip(pos, 0, N - 1)
            mask = (sorted_encoded[pos] == potential)
            sources.append(sort_perm[np.where(mask)[0]])
            targets.append(sort_perm[pos[mask]])

        if not sources: return sparse.BCOO.fromdense(jnp.zeros((N, N), dtype=jnp.float32))

        # Build Matrix
        all_src = np.concatenate(sources)
        all_tgt = np.concatenate(targets)
        indices = jnp.column_stack((jnp.array(all_tgt), jnp.array(all_src)))
        values = jnp.ones(len(all_src), dtype=jnp.float32)

        # Injected Weights Check
        if hasattr(self, 'weight'):
            values = jnp.full((len(all_src),), self.weight, dtype=jnp.float32)

        return sparse.BCOO((values, indices), shape=(N, N))

    def _build_standard_matrix(self) -> sparse.BCOO:
        states = list(self.discrete_space.states)
        N = len(states)
        state_map = {s: i for i, s in enumerate(states)}
        if hasattr(self.discrete_space, 'state_to_index'): state_map = self.discrete_space.state_to_index
        sources, targets = [], []
        for i, s in enumerate(states):
            for n in self.compute_neighbors(s):
                idx = -1
                if n in state_map:
                    idx = state_map[n]
                elif VectorState:
                    try:
                        idx = state_map.get(VectorState(n), -1)
                    except:
                        pass
                if idx != -1: sources.append(i); targets.append(idx)
        if not sources: return sparse.BCOO.fromdense(jnp.zeros((N, N), dtype=jnp.float32))
        indices = jnp.column_stack((jnp.array(targets), jnp.array(sources)))
        values = jnp.ones(len(sources), dtype=jnp.float32)
        return sparse.BCOO((values, indices), shape=(N, N))

    def get_raw_successor(self, state_data_batch: Sequence[Any]) -> Sequence[Any]:
        if self._fast_path_active: return []
        ids = self._raw_register_batch(state_data_batch)
        self._raw_ensure_expanded(ids)
        next_indices = set()
        if self._raw_cpu_matrix is not None:
            rows = self._raw_cpu_matrix.rows
            for src_id in ids:
                if src_id < len(rows): next_indices.update(rows[src_id])
        return [self._id_to_raw[i] for i in next_indices]

    # --- RESTORED MISSING METHOD ---
    def get_raw_multi_step_successor(self, initial_states: Sequence[Any], steps: int) -> Sequence[Any]:
        current_batch = initial_states
        for _ in range(steps):
            current_batch = self.get_raw_successor(current_batch)
            if not current_batch: break
        return current_batch

    def expand_frontier(self, initial_states: Sequence[Any], depth: int):
        ids = self._raw_register_batch(initial_states)
        for _ in range(depth):
            self._raw_ensure_expanded(ids)
            next_ids = []
            if self._raw_cpu_matrix is None: break
            rows = self._raw_cpu_matrix.rows
            for src_id in ids:
                if src_id < len(rows): next_ids.extend(rows[src_id])
            if not next_ids: break
            ids = list(set(next_ids))

    def export_discovery(self) -> Tuple[List[Any], Optional[sparse.BCOO]]:
        if self._fast_path_active and self._fast_coords is not None:
            self._adjacency_matrix = self._build_fast_vector_matrix()
            return self._fast_coords.tolist(), self._adjacency_matrix
        self._raw_sync_to_device()
        return self._id_to_raw, self._raw_jax_matrix

    def _raw_sync_to_device(self):
        if self._raw_dirty or self._raw_jax_matrix is None:
            if self._raw_cpu_matrix is None or self._raw_cpu_matrix.shape == (0, 0):
                self._raw_jax_matrix = sparse.BCOO.fromdense(jnp.zeros((0, 0), dtype=jnp.float32))
            else:
                coo = self._raw_cpu_matrix.transpose().tocoo()
                indices = jnp.array(np.vstack((coo.row, coo.col)).T)
                values = jnp.array(coo.data)
                self._raw_jax_matrix = sparse.BCOO((values, indices), shape=coo.shape)
            self._raw_dirty = False

    def _raw_register_batch(self, data_batch):
        ids = []
        if hasattr(data_batch, 'tolist'): data_batch = data_batch.tolist()
        for data in data_batch:
            if hasattr(data, 'item') and hasattr(data, 'shape') and data.shape == (): data = data.item()
            if data not in self._raw_to_id:
                new_id = len(self._id_to_raw)
                self._raw_to_id[data] = new_id
                self._id_to_raw.append(data)
            ids.append(self._raw_to_id[data])
        return ids

    def _raw_ensure_expanded(self, ids):
        if self._fast_path_active: return
        unknown = [i for i in ids if i not in self._raw_explored]
        if not unknown: return
        new_rows, new_cols = [], []
        if self._raw_cpu_matrix is None:
            self._raw_cpu_matrix = sp.lil_matrix((0, 0), dtype=np.float32)
        current_max_id = self._raw_cpu_matrix.shape[0] - 1
        for src_id in unknown:
            data = self._id_to_raw[src_id]
            neighbors = self.compute_neighbors(data)
            if not neighbors:
                self._raw_explored.add(src_id);
                continue
            dst_ids = self._raw_register_batch(neighbors)
            count = len(dst_ids)
            new_rows.extend([src_id] * count)
            new_cols.extend(dst_ids)
            local_max = max(dst_ids) if dst_ids else src_id
            local_max = max(local_max, src_id)
            if local_max > current_max_id: current_max_id = local_max
            self._raw_explored.add(src_id)
        if not new_rows: return
        if current_max_id >= self._raw_cpu_matrix.shape[0]:
            new_size = max(current_max_id + 1, int(self._raw_cpu_matrix.shape[0] * 1.5))
            self._raw_cpu_matrix.resize((new_size, new_size))
        self._raw_cpu_matrix[new_rows, new_cols] = 1.0
        self._raw_dirty = True

    def successor(self, state):
        return self.discrete_space.create_subset(list(self.compute_neighbors(state)))

    def multi_step_successor(self, initial_state, steps):
        idx = self.discrete_space.get_index_of(initial_state)
        if idx == -1: return self.discrete_space.create_subset([])
        N = self.discrete_space.num_states
        x = jnp.zeros(N, dtype=jnp.float32).at[idx].set(1.0)
        A = self.adjacency_matrix

        def body(i, val): return (A @ val > 0.0).astype(jnp.float32)

        final_x = jax.lax.fori_loop(0, steps, body, x)
        return self._indices_to_space(jnp.where(final_x > 0)[0])

    def _indices_to_space(self, indices):
        all_s = list(self.discrete_space.states)
        return self.discrete_space.create_subset([all_s[i] for i in np.array(indices)])


class GraphTopology(DiscreteTopology):
    def __init__(self, state_space, adjacency_matrix=None, edges=None):
        super().__init__(state_space)
        if isinstance(adjacency_matrix, list) and edges is None: edges = adjacency_matrix; adjacency_matrix = None
        N = state_space.num_states
        if edges is not None:
            if len(edges) > 0 and not isinstance(edges[0][0], (int, np.integer)):
                new_edges = []
                for u, v in edges:
                    u_idx = state_space.get_index_of(u)
                    v_idx = state_space.get_index_of(v)
                    if u_idx != -1 and v_idx != -1: new_edges.append((u_idx, v_idx))
                edges = new_edges
        if adjacency_matrix is not None:
            if sp.issparse(adjacency_matrix):
                self._raw_cpu_matrix = adjacency_matrix.tolil().astype(np.float32)
            else:
                self._raw_cpu_matrix = sp.lil_matrix(adjacency_matrix, dtype=np.float32)
        elif edges is not None:
            self._raw_cpu_matrix = sp.lil_matrix((N, N), dtype=np.float32)
            if edges:
                srcs, tgts = zip(*edges)
                self._raw_cpu_matrix[srcs, tgts] = 1.0
        else:
            raise ValueError("GraphTopology requires 'adjacency_matrix' or 'edges'.")
        coo = self._raw_cpu_matrix.transpose().tocoo()
        indices = jnp.array(np.vstack((coo.row, coo.col)).T)
        values = jnp.array(coo.data)
        self._adjacency_matrix = sparse.BCOO((values, indices), shape=coo.shape)
        self._id_to_raw = list(state_space.states)
        self._raw_to_id = {s: i for i, s in enumerate(self._id_to_raw)}
        self._raw_explored = set(range(N))

    def compute_neighbors(self, state_val):
        idx = self.discrete_space.get_index_of(state_val) if hasattr(self.discrete_space,
                                                                     'get_index_of') else self._raw_to_id.get(state_val,
                                                                                                              -1)
        if idx == -1: return []
        return [self._id_to_raw[i] for i in self._raw_cpu_matrix.rows[idx]]


class VectorGridTopology(DiscreteTopology):
    def __init__(self, state_space, deltas):
        super().__init__(state_space)
        self.deltas = np.array(deltas, dtype=np.int64)

    def compute_neighbors(self, state_val):
        raw = state_val.values if hasattr(state_val, 'values') else state_val
        return [tuple((np.array(raw) + d).tolist()) for d in self.deltas]

    def _build_adjacency_matrix(self) -> sparse.BCOO:
        return self._build_fast_vector_matrix()

    def expand_frontier(self, initial_states: Sequence[Any], depth: int):
        self._fast_path_active = True
        initial_arr = np.array(initial_states, dtype=np.int64)
        if initial_arr.ndim == 1: initial_arr = initial_arr.reshape(1, -1)
        D = initial_arr.shape[1]
        dtype_view = np.dtype((np.void, D * 8))
        current_frontier = np.ascontiguousarray(initial_arr).view(dtype_view).ravel()
        visited_view = current_frontier.copy()
        self._raw_cpu_matrix = None

        for _ in range(depth):
            if current_frontier.size == 0: break
            current_coords = current_frontier.view(np.int64).reshape(-1, D)
            candidates = current_coords[:, None, :] + self.deltas[None, :, :]
            candidates_flat = candidates.reshape(-1, D)
            candidates_view = np.ascontiguousarray(candidates_flat).view(dtype_view).ravel()
            unique_candidates = np.unique(candidates_view)
            new_frontier = np.setdiff1d(unique_candidates, visited_view, assume_unique=True)
            if new_frontier.size == 0: break
            visited_view = np.concatenate((visited_view, new_frontier))
            visited_view.sort()
            current_frontier = new_frontier

        self._fast_coords = visited_view.view(np.int64).reshape(-1, D)


class DeltaTopology(DiscreteTopology):
    def __init__(self, state_space, deltas):
        super().__init__(state_space)
        clean_deltas = []
        for d in deltas:
            if hasattr(d, 'values'):
                clean_deltas.append(d.values)
            else:
                clean_deltas.append(d)
        self.deltas = np.array(clean_deltas)

    def compute_neighbors(self, state_val):
        neighbors = []
        raw = state_val.values if hasattr(state_val, 'values') else state_val
        base = np.array(raw) if isinstance(raw, (tuple, list, np.ndarray)) else raw
        for d in self.deltas:
            candidate = tuple((base + d).tolist()) if isinstance(raw, (tuple, list, np.ndarray)) else raw + d
            if self.discrete_space:
                is_in = self.discrete_space.contains(candidate)
                if hasattr(is_in, 'size') and is_in.size > 1:
                    is_in = is_in.all()
                elif hasattr(is_in, 'item'):
                    is_in = is_in.item()
                if not is_in and VectorState:
                    is_in = self.discrete_space.contains(VectorState(candidate))
                    if hasattr(is_in, 'size') and is_in.size > 1:
                        is_in = is_in.all()
                    elif hasattr(is_in, 'item'):
                        is_in = is_in.item()
                if is_in: neighbors.append(candidate)
            else:
                neighbors.append(candidate)
        return neighbors


class MetricDiscreteTopology(DiscreteTopology):
    def __init__(self, state_space, max_dist, distance_fn=None):
        super().__init__(state_space)
        self.max_dist = max_dist

    def compute_neighbors(self, state_val):
        neighbors = []
        candidates = self.discrete_space.states if self.discrete_space else self._id_to_raw
        q = np.array(state_val.values) if hasattr(state_val, 'values') else np.array(state_val)
        for c in candidates:
            if c == state_val: continue
            c_val = np.array(c.values) if hasattr(c, 'values') else np.array(c)
            if np.linalg.norm(q - c_val) <= self.max_dist: neighbors.append(c)
        return neighbors


# --- ADDED THIS CLASS BACK ---
class WeightedVectorGridTopology(VectorGridTopology):
    def __init__(self, state_space, deltas, weight=1.0):
        super().__init__(state_space, deltas)
        self.weight = weight