from typing import Mapping, Optional, Iterable, Set, List, Dict, Tuple, Iterator
import itertools

from fds.core.fds_state.state_space import StatSpace, DiscreteFiniteStatSpace
from fds.affecting.affecting_systems_framework.correlated_state import CorrelatedState
from fds.dynamic_systems.field_dynamic_system import FieldDynamicSystem
from fds.core.fds_state.state import State  # your new @dataclass State

SName = str


class _FiniteFromStates(DiscreteFiniteStatSpace[State]):
    """
    Wrap a finite set[State] as a DiscreteFiniteStatSpace.

    This gives us:
      - ids_view(), get_id(), get_state_by_id(), size()
      - O(1) membership and efficient iteration
    """

    def __init__(self, states: Set[State], current: Optional[State] = None) -> None:
        if not states:
            raise ValueError("_FiniteFromStates requires a non-empty set of states")
        cur = current if current is not None else next(iter(states))
        # DiscreteFiniteStatSpace takes an Iterable[S] and an optional current
        super().__init__(states=states, current=cur, key=None, dim=None)

class CorrelatedStateSpace(DiscreteFiniteStatSpace[CorrelatedState]):
    """
    Cartesian-product state space over labeled factor spaces, implemented as a
    DiscreteFiniteStatSpace[CorrelatedState].

    - Each element is a CorrelatedState (which itself is a State subclass).
    - Provides ids_view(), get_id(), get_state_by_id(), size(), etc.
    - Keeps factor metadata (labels, factor spaces) for pack/unpack and analysis.
    """

    def __init__(
        self,
        spaces: Mapping[SName, StatSpace],
        *,
        order: Optional[Iterable[SName]] = None,
        states: Optional[Iterable[CorrelatedState]] = None,
        initial_state: Optional[CorrelatedState] = None,
    ) -> None:
        if not spaces:
            raise ValueError("CorrelatedStateSpace requires at least one component space")

        # ----- factor metadata -----
        self._labels: List[SName] = list(order) if order is not None else list(spaces.keys())
        self._spaces: List[StatSpace] = [spaces[name] for name in self._labels]
        self._name_to_idx: Dict[SName, int] = {n: i for i, n in enumerate(self._labels)}

        # ----- determine states -----
        if states is not None:
            # Use exactly the provided correlated states
            state_list = list(states)
            if not state_list:
                raise ValueError("CorrelatedStateSpace: `states` cannot be empty")
        else:
            # Materialize full Cartesian product of factor spaces
            state_list = list(self._product_states())
            if not state_list:
                raise ValueError("CorrelatedStateSpace: Cartesian product is empty")

        # ----- determine initial_state -----
        if initial_state is None:
            # default: deterministic choice – first in list
            initial_state = state_list[0]
        else:
            if initial_state not in state_list:
                # keep semantics: ensure initial is part of the space
                state_list = [initial_state] + state_list

        # ----- dimension: sum of factor dimensions -----
        total_dim = 0
        for sp in self._spaces:
            d = sp.dimension()
            total_dim += int(d)

        # ----- call base ctor (this builds ids, maps, etc.) -----
        super().__init__(states=state_list, current=initial_state, key=None, dim=total_dim)

    # -------------------- StatSpace / metadata API --------------------

    def dimension(self) -> int:
        """Total dimension = sum of factor dimensions."""
        return int(self._dim or 1)

    def labels(self) -> Tuple[SName, ...]:
        return tuple(self._labels)

    def spaces(self) -> Tuple[StatSpace, ...]:
        return tuple(self._spaces)

    def factor(self, name: SName) -> StatSpace:
        return self._spaces[self._name_to_idx[name]]

    def shape(self) -> Tuple[int, ...]:
        """
        Finite cardinality for each factor.
        Note: calls get_all_states() on each factor; make sure those are finite.
        """
        return tuple(len(sp.get_all_states()) for sp in self._spaces)

    def size(self) -> int:
        """
        Total number of correlated states (may be smaller than product of shapes
        if this space was built from an explicit subset).
        """
        return super().size()

    def contains(self, s: CorrelatedState) -> bool:  # type: ignore[override]
        """
        Fast logical check: each component must belong to its factor space.

        NOTE: This is slightly weaker than "s is in our stored set" if you build
        the space from an explicit subset, but it is consistent with the idea
        of the correlated universe. If you want strict membership, you can
        also use `s in self.get_all_states()`.
        """
        cs = self.pack(s)
        for comp, sp in zip(cs._states, self._spaces):  # relies on CorrelatedState internals
            if not sp.contains(comp):
                return False
        return True

    # -------------------- helpers --------------------

    def iter_states(self) -> Iterator[CorrelatedState]:
        """Iterate over all states in this space."""
        # base class get_all_states() returns an iterable over actual states
        yield from self.get_all_states()

    # -------- representation helpers --------
    def pack(self, data: "Mapping[SName, State] | Iterable[State] | CorrelatedState") -> CorrelatedState:
        """
        Normalize inputs into a CorrelatedState with our label order.
        """
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

    def unpack(self, cs: CorrelatedState) -> Mapping[SName, State]:
        return {n: cs[n] for n in self._labels}

    # -------------------- internals --------------------

    def _product_states(self) -> Iterator[CorrelatedState]:
        """
        Cartesian product of factor spaces as CorrelatedState objects.
        Used when `states` is not explicitly provided.
        """
        sets_per_factor: List[Iterable[State]] = [sp.get_all_states() for sp in self._spaces]
        for combo in itertools.product(*sets_per_factor):
            comps = {n: combo[i] for i, n in enumerate(self._labels)}
            yield CorrelatedState(comps, order=self._labels)

    # ========= Convenience constructors (classmethods) ========= #

    @classmethod
    def build_from_set_of_states(
        cls,
        states: Set[CorrelatedState],
        initial_state: Optional[CorrelatedState] = None,
        *,
        order: Optional[Iterable[SName]] = None,
    ) -> "CorrelatedStateSpace":
        """
        Build a correlated space whose contents are exactly `states`.
        Factor spaces are inferred per label from the components.
        """
        if not states:
            raise ValueError("build_from_set_of_states: `states` cannot be empty")

        sample = next(iter(states))
        lbls: List[SName] = list(order) if order is not None else list(sample.get_order())

        # ensure initial present
        if initial_state is not None and initial_state not in states:
            states = set(states)
            states.add(initial_state)

        # collect per-label component states
        per_label: Dict[SName, Set[State]] = {n: set() for n in lbls}
        for cs in states:
            for n in lbls:
                per_label[n].add(cs[n])

        # build factor spaces as finite wrappers
        factor_spaces: Dict[SName, StatSpace] = {}
        for n in lbls:
            cur_comp = next(iter(per_label[n]))
            factor_spaces[n] = _FiniteFromStates(per_label[n], current=cur_comp)

        # choose initial correlated state
        if initial_state is None:
            init_map = {n: next(iter(per_label[n])) for n in lbls}
            initial_state = CorrelatedState(init_map, order=lbls)

        return cls(spaces=factor_spaces, order=lbls, states=states, initial_state=initial_state)

    @classmethod
    def build_from_system_initials(
        cls,
        systems: Dict[str, FieldDynamicSystem],
        *,
        order: Optional[Iterable[SName]] = None,
    ) -> "CorrelatedStateSpace":
        """
        Build a correlated space containing exactly ONE correlated state:
        the tuple of each system's initial state.
        """
        if not systems:
            raise ValueError("build_from_system_initials: `systems` cannot be empty")

        lbls: List[SName] = list(order) if order is not None else list(systems.keys())

        factor_spaces: Dict[SName, StatSpace] = {}
        init_map: Dict[SName, State] = {}
        for name in lbls:
            sys = systems[name]
            s0 = sys.state_space.get_state() if hasattr(sys, "state_space") else sys.initial_state
            init_map[name] = s0
            factor_spaces[name] = _FiniteFromStates({s0}, current=s0)

        cs0 = CorrelatedState(init_map, order=lbls)
        return cls(spaces=factor_spaces, order=lbls, states={cs0}, initial_state=cs0)

    @classmethod
    def build_from_dict_of_states(
        cls,
        system_state: Dict[str, State],
        *,
        order: Optional[Iterable[SName]] = None,
    ) -> "CorrelatedStateSpace":
        """
        Build a correlated space containing exactly ONE correlated state formed
        from the provided mapping (name -> component state).
        """
        if not system_state:
            raise ValueError("build_from_dict_of_states: `system_state` cannot be empty")

        lbls: List[SName] = list(order) if order is not None else list(system_state.keys())

        factor_spaces: Dict[SName, StatSpace] = {}
        for name in lbls:
            comp = system_state[name]
            factor_spaces[name] = _FiniteFromStates({comp}, current=comp)

        cs0 = CorrelatedState(system_state, order=lbls)
        return cls(spaces=factor_spaces, order=lbls, states={cs0}, initial_state=cs0)

    @classmethod
    def build_state_space_from_inputs(
        cls,
        states: Optional[Set[CorrelatedState]] = None,
        initial_state: Optional[CorrelatedState] = None,
        systems: Optional[Dict[str, FieldDynamicSystem]] = None,
        system_state: Optional[Dict[str, State]] = None,
        *,
        order: Optional[Iterable[SName]] = None,
    ) -> "CorrelatedStateSpace":
        """
        Unified convenience: decides which constructor to use based on which
        argument is provided (in priority order): `states`, `systems`, `system_state`.
        """
        if states is not None:
            return cls.build_from_set_of_states(states, initial_state, order=order)
        if systems is not None:
            return cls.build_from_system_initials(systems, order=order)
        if system_state is not None:
            return cls.build_from_dict_of_states(system_state, order=order)
        raise ValueError("build_state_space_from_inputs: provide one of {states, systems, system_state}")

    def __repr__(self):
        ids = self.ids_view()
        return_string = ""
        for id in ids:
            state = self.get_state_by_id(id)
            return_string += f"{id}: {state} \n"
        return return_string

    def type_of_individual_states(self):
        pass
