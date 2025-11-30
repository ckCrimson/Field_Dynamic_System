import copy
import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Generic, TypeVar, Set, Callable, List, Optional, Iterable, Tuple

import numpy as np

from fds import State, StatSpace
from fds.core.fds_state import Reachable
from fds.core.fds_state.state_space import DiscreteFiniteStatSpace

S = TypeVar('S', bound=State)

Reach1 = Callable[[int], np.ndarray]




def rebuild_like(proto_out_space, states: Iterable[S], current: Optional[S] = None):
    """
    Build a NEW state space of the same concrete kind as `proto_out_space`,
    populated with `states`, with `current` if provided.
    No mutation of the prototype; no deepcopy.
    """
    # DiscreteFiniteStatSpace is your concrete class we use everywhere

    if isinstance(proto_out_space, DiscreteFiniteStatSpace):
        # preserve ordering rule and dim if present
        return DiscreteFiniteStatSpace(
            states=states,
            current=current,
            key=getattr(proto_out_space, "_key", None),
            dim=getattr(proto_out_space, "_dim", None),
        )

    # If in future you add more concrete spaces, branch here.
    raise TypeError(f"Unsupported prototype space type: {type(proto_out_space).__name__}")


def get_reaching_ids(
    initial_id: int,
    space,                 # DiscreteFiniteStatSpace-like (size(), get_state_by_id())
    l: int,
    step_ids: Reach1,      # fast one-step expander on IDs
) -> np.ndarray:
    """
    Return ids reachable from `initial_id` in EXACTLY `l` steps.
    Allocation-light; memoizes one-step expansions; early exits on empty frontier.
    """
    if l < 0:
        raise ValueError("l must be >= 0")
    if l == 0:
        return np.array([initial_id], dtype=np.int32)

    N = space.size()
    cache: List[Optional[np.ndarray]] = [None] * N

    def expand_once(sid: int) -> np.ndarray:
        arr = cache[sid]
        if arr is None:
            nxt = step_ids(sid)
            arr = np.unique(nxt.astype(np.int32, copy=False))  # unique/sorted for fast unions
            cache[sid] = arr
        return arr

    frontier = np.array([initial_id], dtype=np.int32)
    for _ in range(l):
        if frontier.size == 0:
            break
        combined = np.concatenate([expand_once(sid) for sid in frontier]) if frontier.size else frontier
        frontier = np.unique(combined)
    return frontier

def make_step_ids_from_reachable(space, reachable, *, use_allowed: bool = False):
    """
    Robust adapter: converts the reachable states to ids using either:
    - space.get_id(s)  (fast path when states match universe keys)
    - space.id_of(s.value)  (IntegerLine fallback by value)
    - or coords-based fallback if available
    """
    def to_id_safe(s):
        # 1) direct lookup if equal/hash-equal
        try:
            return space.get_id(s)
        except Exception:
            pass
        # 2) IntegerLine: map by integer value
        if hasattr(space, "id_of") and hasattr(s, "value"):
            return space.id_of(s.value)
        # 3) coords-based fallback (if your State exposes coords and your space provides a lookup)
        if hasattr(s, "coords") and len(getattr(s, "coords")) == 1 and hasattr(space, "id_of"):
            return space.id_of(int(s.coords[0]))
        # If you have other spaces, add their canonicalization here.
        raise KeyError(f"State {s!r} not in reference_space universe")

    def step_ids(sid: int) -> np.ndarray:
        src = space.get_state_by_id(sid)
        dst_space = reachable.get_reachable(src)  # uses policy internally if present
        dst_states = list(dst_space.get_all_states())
        if not dst_states:
            return np.empty(0, dtype=np.int32)
        ids = np.fromiter((to_id_safe(s) for s in dst_states), dtype=np.int32, count=len(dst_states))
        return np.unique(ids)

    return step_ids

class MultiStepReaching(Generic[S]):
    """
    Computes the state space of states reachable in exactly `l` steps
    by iteratively using a single-step Reachable.
    """
    def __init__(self, reachable: Reachable[S]):
        self.reachable = reachable

    def get_reaching_primary(self, initial_state: S, reference_space: StatSpace[S],l: int) -> StatSpace[S]:
        """
        Returns a StatSpace containing all states reachable from `initial_state` in exactly l steps.
        """
        prototype_space  = copy.deepcopy(reference_space)
        current_states: Set[S] = {initial_state}
        for _ in range(l):
            next_states: Set[S] = set()
            for s in current_states:
                # single-step reachable returns a StatSpace
                reachable_space = self.reachable.get_reachable(s)
                next_states = next_states.union(reachable_space.get_all_states())
            current_states = next_states

        # Build resulting state space: choose one of the reachable states as current (or keep initial
        prototype_space.set_state(initial_state)
        prototype_space.build_from_states(current_states)
        return prototype_space

    def get_reaching(
            self,  # keep your method shape
            initial_state: S,
            l: int,
            reference_space,
            get_reaching_from_allowed: bool = False,
    ):
        """
        NEW: ID-first multi-step reaching; no deepcopy.
        Builds the output with `rebuild_like(reference_space, ...)`.
        """
        # map once
        initial_id = reference_space.get_id(initial_state)

        # build fast one-step expander from your current reachable object
        step_ids = make_step_ids_from_reachable(reference_space, self.reachable,
                                                use_allowed=get_reaching_from_allowed)

        # run optimized engine
        ids = get_reaching_ids(initial_id, reference_space, l, step_ids)

        # materialize states ONCE at the end
        dst_states = [reference_space.get_state_by_id(i) for i in ids]

        # choose stable current
        cur = initial_state if initial_id in set(ids.tolist()) else (dst_states[0] if dst_states else initial_state)

        # rebuild (no deepcopy) of the same kind as the reference
        return rebuild_like(reference_space, dst_states, current=cur)

    # src/fds/dynamics/multi_step/get_multi_step_reaching.py

    # ---- helper: pure worker (must be top-level for pickling) ----
    def _expand_chunk_ids(
            ids_chunk: np.ndarray,
            use_csr: bool,
            indptr: Optional[np.ndarray],
            indices: Optional[np.ndarray],
            # When use_csr=False:
            step_ids_is_none: bool,
            # Picklable callable (or None when CSR is used)
            step_ids: Optional[Callable[[int], np.ndarray]],
    ) -> np.ndarray:
        """Expand one chunk of frontier ids into a single np.int32 array of neighbor ids."""
        out_arrays: List[np.ndarray] = []
        if use_csr:
            # CSR: neighbors = indices[indptr[sid]:indptr[sid+1]]
            for sid in ids_chunk:
                a, b = int(indptr[sid]), int(indptr[sid + 1])
                if b > a:
                    out_arrays.append(indices[a:b])
        else:
            # Function: neighbors = step_ids(sid)  (may or may not be unique)
            if step_ids_is_none or step_ids is None:
                return np.empty(0, dtype=np.int32)
            for sid in ids_chunk:
                nb = step_ids(int(sid))
                if nb.size:
                    out_arrays.append(nb.astype(np.int32, copy=False))
        if not out_arrays:
            return np.empty(0, dtype=np.int32)
        return np.concatenate(out_arrays, axis=0)

    def get_multi_step_reaching(
            self,
            initial_state: S,
            reference_space,  # DiscreteFiniteStateSpace-like
            l: int,
            *,
            csr: Optional[Tuple[np.ndarray, np.ndarray]] = None,  # (indptr, indices)
            step_ids: Optional[Callable[[int], np.ndarray]] = None,  # top-level callable returning np.ndarray[int32]
            parallel: bool = False,
            use_processes: bool = True,
            workers: Optional[int] = None,
            chunk_min_size: int = 2048,
            parallel_threshold: int = 20_000,
            exactly_l_steps: bool = True,  # if False → ≤ l steps via visited mask
    ):
        """
        Multi-step reachability via ID-first frontier expansion.
        - Fixes universe mismatch: uses ids_view() only if universes match; else maps states→ids.
        - Deterministic (concat → unique), no deepcopy (rebuild_like at end).
        """

        if l < 0:
            raise ValueError("l must be >= 0")

        # -------- helper: rebuild without deepcopy (adjust import path if needed) --------
        def rebuild_like(proto_out_space, states: Iterable[S], current: Optional[S] = None):
            if isinstance(proto_out_space, DiscreteFiniteStatSpace):
                return DiscreteFiniteStatSpace(
                    states=states,
                    current=current,
                    key=getattr(proto_out_space, "_key", None),
                    dim=getattr(proto_out_space, "_dim", None),
                )
            raise TypeError(f"Unsupported prototype type: {type(proto_out_space).__name__}")

        # -------- universe & trivial case --------
        src_id = reference_space.get_id(initial_state)
        N = reference_space.size()
        if l == 0:
            return rebuild_like(reference_space, [initial_state], current=initial_state)

        # -------- choose expander: CSR / step_ids / internal adapter --------
        use_csr = csr is not None
        indptr = indices = None
        if use_csr:
            indptr, indices = csr
            if indptr.dtype not in (np.int64, np.int32):
                indptr = indptr.astype(np.int64, copy=False)
            if indices.dtype != np.int32:
                indices = indices.astype(np.int32, copy=False)
            if indptr.shape[0] != N + 1:
                raise ValueError("CSR indptr must have length N+1")

        # Internal adapter from self.reachable if neither CSR nor step_ids provided
        if (not use_csr) and (step_ids is None):
            # capture universe identity if available
            ref_uid = getattr(reference_space, "universe_id", None)

            def _step_ids_from_reachable(sid: int) -> np.ndarray:
                src = reference_space.get_state_by_id(int(sid))
                dst_space = self.reachable.get_reachable(src)

                # Use ids_view only if it's the SAME universe
                same_universe = (
                        ref_uid is not None
                        and getattr(dst_space, "universe_id", None) == ref_uid
                )
                if same_universe and hasattr(dst_space, "ids_view") and callable(dst_space.ids_view):
                    ids = dst_space.ids_view()
                    return ids.astype(np.int32, copy=False) if ids.dtype != np.int32 else ids

                # Otherwise, robust mapping states -> reference ids
                states = list(dst_space.get_all_states()) if hasattr(dst_space, "get_all_states") else []
                if not states:
                    return np.empty(0, dtype=np.int32)
                try:
                    ids = np.fromiter(
                        (reference_space.get_id(s) for s in states),
                        dtype=np.int32, count=len(states)
                    )
                except Exception:
                    if hasattr(reference_space, "id_of"):
                        def to_id(s):
                            if hasattr(s, "value"):
                                return reference_space.id_of(int(s.value))
                            if hasattr(s, "coords"):
                                return reference_space.id_of(int(s.coords[0]))
                            raise KeyError(f"Cannot map {s!r} to id")

                        ids = np.fromiter((to_id(s) for s in states), dtype=np.int32, count=len(states))
                    else:
                        raise
                return np.unique(ids)

            step_ids = _step_ids_from_reachable
            # Closure is not picklable → disable process-based parallel
            if parallel and use_processes:
                parallel = False

        # -------- ≤ l semantics (visited mask) --------
        visited = None
        if not exactly_l_steps:
            visited = np.zeros(N, dtype=bool)
            visited[src_id] = True

        # -------- single-process memoization (avoid cross-worker contention) --------
        cache: List[Optional[np.ndarray]] = [None] * N if not parallel else []

        def expand_once_single(sid: int) -> np.ndarray:
            arr = cache[sid] if cache else None
            if arr is not None:
                return arr
            if use_csr:
                a, b = int(indptr[sid]), int(indptr[sid + 1])
                nb = indices[a:b]
            else:
                nb = step_ids(int(sid))
                if nb.dtype != np.int32:
                    nb = nb.astype(np.int32, copy=False)
            nb = np.unique(nb) if nb.size else nb
            if cache:
                cache[sid] = nb
            return nb

        # -------- frontier loop --------
        frontier = np.array([src_id], dtype=np.int32)

        for _ in range(l):
            if frontier.size == 0:
                break

            if (not parallel) or (frontier.size < parallel_threshold) or (workers == 1):
                # single-process
                exps = [expand_once_single(int(sid)) for sid in frontier]
                combined = np.concatenate(exps) if exps else np.empty(0, dtype=np.int32)
            else:
                # parallel map-reduce
                k = workers
                if k is None:
                    try:
                        import os
                        k = os.cpu_count() or 4
                    except Exception:
                        k = 4
                n = frontier.size
                target_tasks = max(k * 4, 1)
                chunk_size = max(chunk_min_size, math.ceil(n / target_tasks))
                chunks = [frontier[i:min(i + chunk_size, n)] for i in range(0, n, chunk_size)]

                exec_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
                combined_arrays: List[np.ndarray] = []
                with exec_cls(max_workers=k) as pool:
                    futures = [
                        pool.submit(
                            _expand_chunk_ids,
                            chunk, use_csr, indptr, indices,
                            step_ids is None, step_ids
                        )
                        for chunk in chunks
                    ]
                    for fut in as_completed(futures):
                        arr = fut.result()
                        if arr.size:
                            combined_arrays.append(arr)
                combined = np.concatenate(combined_arrays) if combined_arrays else np.empty(0, dtype=np.int32)

            if visited is not None and combined.size:
                combined = combined[~visited[combined]]

            next_frontier = np.unique(combined)

            if visited is not None and next_frontier.size:
                visited[next_frontier] = True

            if next_frontier.size == 0:
                frontier = next_frontier
                break
            if next_frontier.size == frontier.size and np.array_equal(next_frontier, frontier):
                break

            frontier = next_frontier

        # -------- build result once (no deepcopy) --------
        out_ids = frontier
        out_states = [reference_space.get_state_by_id(i) for i in out_ids]
        cur = initial_state if src_id in out_ids.tolist() else (out_states[0] if out_states else initial_state)
        return rebuild_like(reference_space, out_states, current=cur)

    # You said you already have this; included for completeness.
    def rebuild_like(proto_out_space, states: Iterable[S], current: Optional[S] = None):
        """
        Build a NEW state space of the same 'kind' as proto_out_space, without deepcopy.
        """
        if isinstance(proto_out_space, DiscreteFiniteStatSpace):
            return DiscreteFiniteStatSpace(
                states=states,
                current=current,
                key=getattr(proto_out_space, "_key", None),
                dim=getattr(proto_out_space, "_dim", None),
            )
        raise TypeError(f"Unsupported prototype type: {type(proto_out_space).__name__}")


def _expand_chunk_ids(
    ids_chunk: np.ndarray,
    use_csr: bool,
    indptr: Optional[np.ndarray],
    indices: Optional[np.ndarray],
    # When use_csr=False:
    step_ids_is_none: bool,
    # Picklable callable (or None when CSR is used)
    step_ids: Optional[Callable[[int], np.ndarray]],
) -> np.ndarray:
    """Expand one chunk of frontier ids into a single np.int32 array of neighbor ids."""
    out_arrays: List[np.ndarray] = []
    if use_csr:
        # CSR: neighbors = indices[indptr[sid]:indptr[sid+1]]
        for sid in ids_chunk:
            a, b = int(indptr[sid]), int(indptr[sid + 1])
            if b > a:
                out_arrays.append(indices[a:b])
    else:
        # Function: neighbors = step_ids(sid)  (may or may not be unique)
        if step_ids_is_none or step_ids is None:
            return np.empty(0, dtype=np.int32)
        for sid in ids_chunk:
            nb = step_ids(int(sid))
            if nb.size:
                out_arrays.append(nb.astype(np.int32, copy=False))
    if not out_arrays:
        return np.empty(0, dtype=np.int32)
    return np.concatenate(out_arrays, axis=0)