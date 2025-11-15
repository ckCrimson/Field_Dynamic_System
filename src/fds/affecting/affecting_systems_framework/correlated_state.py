
import numpy as np

from fds import State



from typing import Mapping, Iterable, Optional, Iterator, Tuple


# assuming your State is imported from fds.core.fds_state.state
# @dataclass(slots=True, frozen=True, eq=True)
# class State: ...

class CorrelatedState(State):
    """
    A composite State consisting of named component States.

    Internally:
      - Base `State.state` is a tuple of (name, component_state.state) pairs,
        in a deterministic order (self._order).
      - We also cache:
          _order: list[str]
          _states: tuple[State, ...]
          _name_to_idx: dict[str, int]
    This keeps it:
      - hashable & comparable via base State,
      - convenient to work with as a correlated container.
    """

    __slots__ = ("_order", "_states", "_name_to_idx", "_vector_cache")

    def __init__(self, components: Mapping[str, State], order: Optional[Iterable[str]] = None):
        # 1) determine order
        order_list = list(order) if order is not None else sorted(components.keys())
        states_tuple: Tuple[State, ...] = tuple(components[name] for name in order_list)

        # 2) build the underlying hashable payload for base State
        #    this encodes both names and inner .state values
        payload = tuple((name, st.state) for name, st in zip(order_list, states_tuple))

        # 3) set base dataclass field and our own slots using object.__setattr__
        object.__setattr__(self, "state", payload)
        object.__setattr__(self, "_order", order_list)
        object.__setattr__(self, "_states", states_tuple)
        object.__setattr__(self, "_name_to_idx", {name: idx for idx, name in enumerate(order_list)})
        object.__setattr__(self, "_vector_cache", None)

    # ---- convenience API ----

    def get_order(self) -> list[str]:
        return self._order

    def get_states(self, i: int) -> State:
        return self._states[i]

    def as_vector(self) -> np.ndarray:
        """
        Concatenate component vectors in the configured order.
        Assumes each component State implements `as_vector()`.
        """
        if self._vector_cache is None:
            vectors = [s.as_vector() for s in self._states]
            object.__setattr__(self, "_vector_cache", np.concatenate(vectors) if vectors else np.array([]))
        return self._vector_cache  # type: ignore[return-value]

    def __getitem__(self, key: int | str) -> State:
        if isinstance(key, int):
            return self._states[key]
        if isinstance(key, str):
            idx = self._name_to_idx[key]
            return self._states[idx]
        raise KeyError(f"Unsupported key type: {key!r}")

    def with_update(self, name: str, new_state: State) -> "CorrelatedState":
        """Return a new CorrelatedState with one component replaced."""
        if name not in self._name_to_idx:
            raise KeyError(name)
        idx = self._name_to_idx[name]
        # rebuild tuple of states with the updated one
        new_states = list(self._states)
        new_states[idx] = new_state
        new_components = {n: new_states[i] for i, n in enumerate(self._order)}
        return CorrelatedState(new_components, order=self._order)

    def items(self) -> Iterator[tuple[str, State]]:
        return zip(self._order, self._states)

    # Optional: nicer repr
    def __repr__(self) -> str:
        parts = ", ".join(f"{name}={st!r}" for name, st in self.items())
        return f"CorrelatedState({parts})"
