import copy
from typing import TypeVar, Generic, Optional, Iterable, Callable, List, Dict, Any

from fds import State, StatSpace
from fds.core.fds_state import StateSpaceMapping

S = TypeVar("S", bound=State)

class IdentityMapping(StateSpaceMapping[S, S], Generic[S]):
    """No-op mapping: returns the same states in the output space type."""

    def __init__(self) -> None:
        super().__init__(mapping_reverse=None)

    def get_mapping(self, state_in: S, stateSpaceOut: StatSpace[S]) -> StatSpace[S]:
        # Build into a *fresh* space to avoid mutating the prototype.
        out = copy.deepcopy(stateSpaceOut)
        out.build_from_states({state_in}, current=state_in)
        return out

    # Optional: more efficient bulk mapping
    def get_mapping_over_state_space(
        self, space_in: StatSpace[S], space_out: StatSpace[S],preserve_current: bool = True
    ) -> StatSpace[S]:
        out = copy.deepcopy(space_out)
        states = space_in.get_all_states()
        cur = space_in.get_state()
        out.build_from_states(states, current=cur)
        return out



class gister:
    """
    Matrix-style register of pairwise mappings between channel/state-space ids.

    - Rows/columns correspond to `channel_id` (string or hashable label).
    - Cell (i, j) holds the mapping M_ij from id_i -> id_j, or None if unset.
    - No auto-inverses or composition; this is a direct lookup table.

    Optimized for O(1) set/get via an index map and a resizable 2D list.

    Convenience: build directly from a dict keyed by `(src_id, dst_id)` pairs via
    `MappingRegister.from_pair_dict({ ("a","b"): mapping_ab, ... })`.
    """

    def __init__(
        self,
        ids: Optional[Iterable[str]] = None,
        *,
        fill_diagonal_identity: bool = True,
        identity_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._ids: List[str] = []
        self._index: Dict[str, int] = {}
        self._M: List[List[Optional[Any]]] = []  # 2D matrix of mappings
        self._fill_diag = fill_diagonal_identity
        self._id_factory = identity_factory or IdentityMapping
        if ids:
            for cid in ids:
                self.add_id(cid)

    # --------------------- factories / bulk build --------------------- #
    @classmethod
    def from_pair_dict(
        cls,
        pair_map: Dict[tuple[str, str], Any],
        *,
        fill_diagonal_identity: bool = True,
        identity_factory: Optional[Callable[[], Any]] = None,
    ) -> "MappingRegister":
        """Build a register from `{(src, dst): mapping, ...}`.
        Automatically collects all ids from keys and fills the matrix.
        """
        # collect unique ids from the pair keys (stable order of first appearance)
        ids: List[str] = []
        seen: set[str] = set()
        for (a, b) in pair_map.keys():
            if a not in seen:
                ids.append(a); seen.add(a)
            if b not in seen:
                ids.append(b); seen.add(b)
        reg = cls(ids=ids,
                  fill_diagonal_identity=fill_diagonal_identity,
                  identity_factory=identity_factory)
        for (src, dst), mapping in pair_map.items():
            reg.set(src, dst, mapping)
        return reg

    def update_from_pair_dict(self, pair_map: Dict[tuple[str, str], Any]) -> None:
        """Add/overwrite entries from a `{(src, dst): mapping}` dict."""
        for (src, dst), mapping in pair_map.items():
            self.set(src, dst, mapping)

    # --------------------- basic administration --------------------- #
    @property
    def ids(self) -> List[str]:
        return list(self._ids)

    def add_id(self, cid: str) -> None:
        if cid in self._index:
            return
        n = len(self._ids)
        self._ids.append(cid)
        self._index[cid] = n
        # extend existing rows
        for row in self._M:
            row.append(None)
        # add new row
        new_row: List[Optional[Any]] = [None] * (n + 1)
        self._M.append(new_row)
        # optional identity on diagonal
        if self._fill_diag:
            self._M[n][n] = self._id_factory()  # type: ignore[call-arg]

    def ensure_ids(self, ids: Iterable[str]) -> None:
        for cid in ids:
            self.add_id(cid)

    def _idx(self, cid: str) -> int:
        try:
            return self._index[cid]
        except KeyError as e:
            raise KeyError(f"Unknown channel_id '{cid}'. Known: {self._ids}") from e

    # --------------------------- set/get --------------------------- #
    def set(self, src: str, dst: str, mapping: Any) -> None:
        if src not in self._index:
            self.add_id(src)
        if dst not in self._index:
            self.add_id(dst)
        i = self._idx(src)
        j = self._idx(dst)
        self._M[i][j] = mapping

    def get(self, src: str, dst: str) -> Any:
        i = self._idx(src)
        j = self._idx(dst)
        m = self._M[i][j]
        if m is None:
            raise KeyError(f"No mapping registered from '{src}' to '{dst}'.")
        return m

    def has(self, src: str, dst: str) -> bool:
        if src not in self._index or dst not in self._index:
            return False
        return self._M[self._index[src]][self._index[dst]] is not None

    # ---------------------- validation / export --------------------- #
    def validate_complete(self, *, require_off_diagonal: bool = True) -> List[tuple[str, str]]:
        """
        Return list of missing (src, dst) pairs. If `require_off_diagonal=True`,
        also require every off-diagonal entry to be present.
        The diagonal is considered valid if identity is filled or not required.
        """
        missing: List[tuple[str, str]] = []
        n = len(self._ids)
        for i in range(n):
            for j in range(n):
                if i == j:
                    if self._fill_diag and self._M[i][j] is None:
                        missing.append((self._ids[i], self._ids[j]))
                    continue
                if require_off_diagonal and self._M[i][j] is None:
                    missing.append((self._ids[i], self._ids[j]))
        return missing

    def to_matrix(self) -> List[List[Optional[Any]]]:
        """Deep-ish copy (rows copied) for inspection or serialization."""
        return [row[:] for row in self._M]

    def __repr__(self) -> str:
        n = len(self._ids)
        header = [" "] + self._ids
        lines = ["\t".join(header)]
        for i, rid in enumerate(self._ids):
            row = [rid]
            for j in range(n):
                row.append("•" if self._M[i][j] is not None else "–")
            lines.append("\t".join(row))
        return "MappingRegister(\n" + "\n".join(lines) + "\n)"
