from typing import Any, Mapping, Optional, Iterable, Set, List, Dict, Tuple, Iterator

import itertools

from fds import StatSpace
from fds.affecting.affecting_systems_framework.correlated_state import CorrelatedState
from fds.dynamic_systems.field_dynamic_system import FieldDynamicSystem

# ==== Project types (use your real imports) ====
# from your_project.state import State
# from your_project.space import StatSpace
# import numpy as np

SName = str
State = Any  # replace with your concrete State base if available


class _FiniteFromStates(StatSpace[State]):
    """Wrap a raw set[State] as a finite StatSpace, preserving current state semantics."""
    def __init__(self, states: Set[State], current: Optional[State] = None):
        init = current if current is not None else next(iter(states))
        self._states: Set[State] = set(states)
        super().__init__(initial_state=init, set_of_states=self._states)

    def get_all_states(self) -> Set[State]:
        return self._states

    def dimension(self) -> int:
        # fall back to 1 unless your States compute a vector length
        return 1

    def build_from_states(self, states: Set[State], current: Optional[State] = None) -> "_FiniteFromStates":
        self._states = set(states)
        if current is not None:
            self.set_state(current)
        return self

    def contains(self, s: State) -> bool:
        return s in self._states


class CorrelatedStateSpace(StatSpace["CorrelatedState"]):
    """
    Cartesian‑product state space over labeled factor spaces.

    ✔ Calls `StatSpace.__init__` (parent ctor) with `initial_state` and the
      (optionally materialized) set of states.
    ✔ Implements `dimension()` as the SUM of factor `dimension()` values.
    ✔ Works with your provided `CorrelatedState` (mapping/tuple backed).
    """

    # -------------------- construction --------------------
    def __init__(
        self,
        spaces: Mapping[SName, StatSpace],
        *,
        order: Optional[Iterable[SName]] = None,
        materialize: bool = True,
        initial_state: Optional["CorrelatedState"] = None,
        set_of_states : Set[CorrelatedState] = None

    ) -> None:
        if not spaces:
            raise ValueError("CorrelatedStateSpace requires at least one component space")

        # Determine deterministic label order
        self._labels: List[SName] = list(order) if order is not None else list(spaces.keys())
        self._spaces: List[StatSpace] = [spaces[name] for name in self._labels]
        self._name_to_idx: Dict[SName, int] = {n: i for i, n in enumerate(self._labels)}

        # Derive initial correlated state from component current states (if not provided)
        if initial_state is None:
            init_map = {n: sp.get_state() for n, sp in zip(self._labels, self._spaces)}
            initial_state = CorrelatedState(init_map, order=self._labels)

        # Optionally pre‑compute the full Cartesian product
        initial_states: Set[CorrelatedState]
        if materialize:
            initial_states = self._product_set()
        else:
            initial_states = set()  # parent ctor will skip build_from_states if empty


        if set_of_states is not None:
            set_of_states.union(initial_states)
            super().__init__(initial_state=initial_state, set_of_states=set_of_states)
        # IMPORTANT: call parent constructor (your requirement)
        else:
            super().__init__(initial_state=initial_state, set_of_states=initial_states)

        # Track whether the set has been materialized
        self._materialized: bool = materialize
        self._states: Set[CorrelatedState] = initial_states

    # -------------------- StatSpace API --------------------
    def get_all_states(self) -> Set["CorrelatedState"]:
        # If we initialized lazily, materialize on first access
        if not self._materialized:
            self._states = self._product_set()
            self._materialized = True
        return self._states

    def build(self,
              spaces: Optional[Mapping[SName, StatSpace]] = None,
              *,
              order: Optional[Iterable[SName]] = None,
              materialize: Optional[bool] = None) -> None:
        if spaces is not None:
            self._labels = list(order) if order is not None else list(spaces.keys())
            self._spaces = [spaces[name] for name in self._labels]
            self._name_to_idx = {n: i for i, n in enumerate(self._labels)}
        if materialize is not None:
            self._materialized = bool(materialize)
        # rebuild if materializing
        if self._materialized:
            self._states = self._product_set()
        else:
            self._states = set()

    def build_from_states(self, states: Set["CorrelatedState"], current: Optional["CorrelatedState"] = None) -> "CorrelatedStateSpace":
        # Called by parent __init__ if a non‑empty set is passed
        self._states = set(states)
        self._materialized = True
        if current is not None:
            self.set_state(current)
        return self

    def dimension(self) -> int:
        """Total dimension = sum of factor dimensions."""
        total = 0
        for sp in self._spaces:
            d = sp.dimension()  # may raise if a factor doesn't define dimension
            total += int(d)
        return total

    # We keep `StatSpace.contains()` around, but this is a faster logical check.
    def contains(self, s: "CorrelatedState") -> bool:  # type: ignore[override]
        cs = self.pack(s)
        for comp, sp in zip(cs._states, self._spaces):
            if not sp.contains(comp):
                return False
        return True

    # -------------------- helpers --------------------
    def labels(self) -> Tuple[SName, ...]:
        return tuple(self._labels)

    def spaces(self) -> Tuple[StatSpace, ...]:
        return tuple(self._spaces)

    def factor(self, name: SName) -> StatSpace:
        return self._spaces[self._name_to_idx[name]]

    def shape(self) -> Tuple[int, ...]:
        """Finite cardinality for each factor (will raise if a factor isn't finite)."""
        return tuple(len(sp.get_all_states()) for sp in self._spaces)

    def size(self) -> int:
        prod = 1
        for d in self.shape():
            prod *= d
        return prod

    def iter_states(self) -> Iterator["CorrelatedState"]:
        """Lazy Cartesian product iteration without materializing a set."""
        iters = [iter(sp.get_all_states()) for sp in self._spaces]
        for combo in itertools.product(*iters):
            comps = {n: combo[i] for i, n in enumerate(self._labels)}
            yield CorrelatedState(comps, order=self._labels)

    # -------- representation helpers --------
    def pack(self, data: "Mapping[SName, State] | Iterable[State] | CorrelatedState") -> "CorrelatedState":
        if isinstance(data, CorrelatedState):
            return data
        if isinstance(data, Mapping):
            comps = {name: data[name] for name in self._labels}
            return CorrelatedState(comps, order=self._labels)
        seq = list(data)
        if len(seq) != len(self._labels):
            raise ValueError("Iterable length does not match number of factors")
        comps = {name: seq[i] for i, name in enumerate(self._labels)}
        return CorrelatedState(comps, order=self._labels)

    def unpack(self, cs: "CorrelatedState") -> Mapping[SName, State]:
        return {n: cs[n] for n in self._labels}

    # -------------------- internals --------------------
    def _product_set(self) -> Set["CorrelatedState"]:
        sets_per_factor: List[Set[State]] = [sp.get_all_states() for sp in self._spaces]
        result: Set[CorrelatedState] = set()
        for combo in itertools.product(*sets_per_factor):
            comps = {n: combo[i] for i, n in enumerate(self._labels)}
            result.add(CorrelatedState(comps, order=self._labels))
        return result
        # ========= Convenience constructors (classmethods) ========= #

    @classmethod
    def build_from_set_of_states(
        cls,
        states: Set["CorrelatedState"],
        initial_state: Optional["CorrelatedState"] = None,
        *,
        order: Optional[Iterable[SName]] = None,
    ) -> "CorrelatedStateSpace":
        """
        Build a correlated space whose contents are exactly `states`.
        If `initial_state` is provided and not in `states`, it is added.

        Factor spaces are inferred per label by collecting component-states
        across the input set and wrapping each collection with `_FiniteFromStates`.
        """
        if not states:
            raise ValueError("build_from_set_of_states: `states` cannot be empty")

        # Use label order from `order`, else infer from any sample CorrelatedState
        sample = next(iter(states))
        lbls: List[SName] = list(order) if order is not None else list(sample.get_order())

        # ensure initial present
        if initial_state is not None and initial_state not in states:
            states = set(states)
            states.add(initial_state)

        # Collect per-label component states
        per_label: Dict[SName, Set[State]] = {n: set() for n in lbls}
        for cs in states:
            cs = cs if isinstance(cs, CorrelatedState) else cls.pack(cls, cs)  # defensive
            for n in lbls:
                per_label[n].add(cs[n])

        # Build factor spaces as finite wrappers over the collected sets
        factor_spaces: Dict[SName, StatSpace] = {}
        for n in lbls:
            # pick a deterministic current component
            cur_comp = next(iter(per_label[n]))
            factor_spaces[n] = _FiniteFromStates(per_label[n], current=cur_comp)

        # Choose initial correlated state
        if initial_state is None:
            init_map = {n: next(iter(per_label[n])) for n in lbls}
            initial_state = CorrelatedState(init_map, order=lbls)

        # Materialize exactly the provided set
        return cls(
            spaces=factor_spaces,
            order=lbls,
            materialize=True,
            initial_state=initial_state,
            # parent ctor will be given the full set via build_from_states below
        ).build_from_states(states, current=initial_state)

    @classmethod
    def build_from_system_initials(
        cls,
        systems: Dict[str, "FieldDynamicSystem"],
        *,
        order: Optional[Iterable[SName]] = None,
        materialize: bool = True,
    ) -> "CorrelatedStateSpace":
        """
        Build a correlated space containing exactly ONE correlated state:
        the tuple of each system's initial state.

        Factor spaces are the systems' own state spaces *restricted* to the
        singleton {initial_state} (via `_FiniteFromStates`).
        """
        if not systems:
            raise ValueError("build_from_system_initials: `systems` cannot be empty")

        lbls: List[SName] = list(order) if order is not None else list(systems.keys())

        # Build per-factor singleton spaces and initial correlated state
        factor_spaces: Dict[SName, StatSpace] = {}
        init_map: Dict[SName, State] = {}
        for name in lbls:
            sys = systems[name]
            s0 = sys.state_space.get_state() if hasattr(sys, "state_space") else sys.initial_state
            init_map[name] = s0
            factor_spaces[name] = _FiniteFromStates({s0}, current=s0)

        cs0 = CorrelatedState(init_map, order=lbls)

        space = cls(
            spaces=factor_spaces,
            order=lbls,
            materialize=materialize,
            initial_state=cs0,
        )
        if materialize:
            # pin it to exactly the singleton set
            space.build_from_states({cs0}, current=cs0)
        return space

    @classmethod
    def build_from_dict_of_states(
        cls,
        system_state: Dict[str, State],
        *,
        order: Optional[Iterable[SName]] = None,
        materialize: bool = True,
    ) -> "CorrelatedStateSpace":
        """
        Build a correlated space containing exactly ONE correlated state formed
        from the provided `system_state` mapping (name -> component state).

        Factor spaces are singleton finite spaces around each provided component.
        """
        if not system_state:
            raise ValueError("build_from_dict_of_states: `system_state` cannot be empty")

        lbls: List[SName] = list(order) if order is not None else list(system_state.keys())

        factor_spaces: Dict[SName, StatSpace] = {}
        for name in lbls:
            comp = system_state[name]
            factor_spaces[name] = _FiniteFromStates({comp}, current=comp)

        cs0 = CorrelatedState(system_state, order=lbls)

        space = cls(
            spaces=factor_spaces,
            order=lbls,
            materialize=materialize,
            initial_state=cs0,
        )
        if materialize:
            space.build_from_states({cs0}, current=cs0)
        return space

    @classmethod
    def build_state_space_from_inputs(
        cls,
        states: Optional[Set["CorrelatedState"]] = None,
        initial_state: Optional["CorrelatedState"] = None,
        systems: Optional[Dict[str, "FieldDynamicSystem"]] = None,
        system_state: Optional[Dict[str, State]] = None,
        *,
        order: Optional[Iterable[SName]] = None,
        materialize: bool = True,
    ) -> "CorrelatedStateSpace":
        """
        Unified convenience: decides which constructor to use based on which
        argument is provided (in priority order): `states`, `systems`, `system_state`.

        - If `states` is provided, delegates to `build_from_set_of_states`.
        - Else if `systems` is provided, delegates to `build_from_system_initials`.
        - Else if `system_state` is provided, delegates to `build_from_dict_of_states`.
        """
        if states is not None:
            return cls.build_from_set_of_states(states, initial_state, order=order)
        if systems is not None:
            return cls.build_from_system_initials(systems, order=order, materialize=materialize)
        if system_state is not None:
            return cls.build_from_dict_of_states(system_state, order=order, materialize=materialize)
        raise ValueError("build_state_space_from_inputs: provide one of {states, systems, system_state}")




