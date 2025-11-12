# ======================================
# Grouping engine: InterAffectingGroups
# ======================================
from typing import Generic, Dict, Optional, Any, List, Set, TypeVar

from fds import State
from fds.affecting.affecting_systems_framework.is_affecting import IsAffecting, Connectivity
from fds.dynamic_systems.field_dynamic_system import FieldDynamicSystem
T = TypeVar("T", bound=State)
class InterAffectingGroups(Generic[T]):
    """
    Partition systems into **disjoint groups** of inter-affecting systems using an
    `IsAffecting` policy.

    Connectivity modes:
      - "weak": undirected edge if affects(a,b) OR affects(b,a); groups = connected components
      - "mutual": undirected edge only if BOTH affects(a,b) AND affects(b,a); groups = connected comps
      - "strong": directed graph with edges a->b if affects(a,b); groups = strongly connected components

    Caches group computations; call `refresh()` when systems/states/params change.
    """

    def __init__(
        self,
        systems: Dict[str, FieldDynamicSystem],
        rule: IsAffecting[T],
        *,
        connectivity: Connectivity = "weak",
        include_singletons: bool = True,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._systems: Dict[str, FieldDynamicSystem] = systems
        self._rule = rule
        self._conn: Connectivity = connectivity
        self._include_singletons = include_singletons
        self._params = params or {}
        # caches
        self._groups_list_cache: Optional[List[Set[str]]] = None
        self._groups_dict_cache: Optional[Dict[int, Dict[str, FieldDynamicSystem]]] = None

    def set_systems(self,systems: Dict[str, FieldDynamicSystem] ):
        self._systems = systems
    # ----------------------- internal graph builders ----------------------- #
    def _build_undirected_adj(self, mutual: bool) -> Dict[str, Set[str]]:
        names = list(self._systems.keys())
        adj: Dict[str, Set[str]] = {n: set() for n in names}
        for i, ni in enumerate(names):
            si = self._systems[ni]
            for nj in names[i + 1 :]:
                sj = self._systems[nj]
                ab = bool(self._rule.affects(si, sj, **self._params))
                ba = bool(self._rule.affects(sj, si, **self._params))
                linked = (ab and ba) if mutual else (ab or ba)
                if linked:
                    adj[ni].add(nj)
                    adj[nj].add(ni)
        return adj

    def _build_directed_adj(self) -> Dict[str, Set[str]]:
        names = list(self._systems.keys())
        out: Dict[str, Set[str]] = {n: set() for n in names}
        for i, ni in enumerate(names):
            si = self._systems[ni]
            for nj in names:
                if nj == ni:
                    continue
                sj = self._systems[nj]
                if self._rule.affects(si, sj, **self._params):
                    out[ni].add(nj)
        return out

    # ----------------------- component extractors ----------------------- #
    @staticmethod
    def _components_undirected(adj: Dict[str, Set[str]], include_singletons: bool) -> List[Set[str]]:
        visited: Set[str] = set()
        groups: List[Set[str]] = []
        for start in adj.keys():
            if start in visited:
                continue
            if not include_singletons and not adj[start]:
                visited.add(start)
                continue
            comp: Set[str] = set()
            stack = [start]
            visited.add(start)
            while stack:
                v = stack.pop()
                comp.add(v)
                for w in adj[v]:
                    if w not in visited:
                        visited.add(w)
                        stack.append(w)
            if include_singletons or len(comp) > 1:
                groups.append(comp)
        return groups

    @staticmethod
    def _scc_kosaraju(out: Dict[str, Set[str]]) -> List[Set[str]]:
        # Build reverse graph
        rev: Dict[str, Set[str]] = {n: set() for n in out}
        for u, nbrs in out.items():
            for v in nbrs:
                rev[v].add(u)
        # 1st pass: order by finish times
        visited: Set[str] = set()
        order: List[str] = []

        def dfs1(u: str) -> None:
            visited.add(u)
            for v in out[u]:
                if v not in visited:
                    dfs1(v)
            order.append(u)

        for n in out.keys():
            if n not in visited:
                dfs1(n)

        # 2nd pass: reverse graph
        comps: List[Set[str]] = []
        visited.clear()

        def dfs2(u: str, comp: Set[str]) -> None:
            visited.add(u)
            comp.add(u)
            for v in rev[u]:
                if v not in visited:
                    dfs2(v, comp)

        for u in reversed(order):
            if u not in visited:
                comp: Set[str] = set()
                dfs2(u, comp)
                comps.append(comp)
        return comps

    # ----------------------------- compute groups ----------------------------- #
    def _compute_groups(self) -> List[Set[str]]:
        if self._conn == "strong":
            out = self._build_directed_adj()
            comps = self._scc_kosaraju(out)
            if not self._include_singletons:
                comps = [c for c in comps if len(c) > 1]
            return comps
        else:
            mutual = (self._conn == "mutual")
            adj = self._build_undirected_adj(mutual)
            return self._components_undirected(adj, self._include_singletons)

    # ----------------------------- public API ----------------------------- #
    def get_affected_groups_list(self) -> List[Set[str]]:
        if self._groups_list_cache is None:
            self._groups_list_cache = self._compute_groups()
        return self._groups_list_cache

    def get_affected_groups_dict(self) -> Dict[int, Dict[str, FieldDynamicSystem]]:
        if self._groups_dict_cache is None:
            groups = self.get_affected_groups_list()
            out: Dict[int, Dict[str, FieldDynamicSystem]] = {}
            for gid, g in enumerate(groups):
                out[gid] = {name: self._systems[name] for name in sorted(g)}
            self._groups_dict_cache = out
        return self._groups_dict_cache

    def as_name_to_group(self) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        for gid, group in self.get_affected_groups_dict().items():
            for name in group.keys():
                mapping[name] = gid
        return mapping

    def refresh(self) -> None:
        self._groups_list_cache = None
        self._groups_dict_cache = None

    def refresh_with_new_systems(self, systems: Dict[str, FieldDynamicSystem]) -> None:
        self._systems = systems
        self.refresh()