from typing import Mapping, Optional, Iterable

import numpy as np

from fds import State


class CorrelatedState(State):
    def __init__(self, components: Mapping[str, State], order: Optional[Iterable[str]] = None):
        """
        components: mapping from system name to its State
        order: deterministic ordering of names; if None, uses sorted(components.keys())
        """
        self._order = list(order) if order is not None else sorted(components.keys())
        self._states = tuple(components[name] for name in self._order)
        # name → index for quick lookup
        self._name_to_idx = {name: idx for idx, name in enumerate(self._order)}
        # Optionally cache concatenated vector
        self._vector_cache: Optional[np.ndarray] = None

    def get_order(self):
        return self._order

    def get_states(self,i):
        return self._states[i]

    def as_vector(self) -> np.ndarray:
        if self._vector_cache is None:
            vectors = [s.as_vector() for s in self._states]
            self._vector_cache = np.concatenate(vectors)
        return self._vector_cache

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._states[key]
        elif isinstance(key, str):
            idx = self._name_to_idx[key]
            return self._states[idx]
        else:
            raise KeyError(f"Unsupported key type: {key}")

    def with_update(self, name: str, new_state: State) -> 'CorrelatedState':
        """Return a new CorrelatedState with one component replaced."""
        if name not in self._name_to_idx:
            raise KeyError(name)
        lst = list(self._states)
        lst[self._name_to_idx[name]] = new_state
        new_components = {n: lst[i] for i, n in enumerate(self._order)}
        return CorrelatedState(new_components, order=self._order)


    def __eq__(self, other):
        if not isinstance(other, CorrelatedState):
            return False
        return self.get_order() == other.get_order() and all(
            self._states[i] == other.get_states(i) for i in range(len(self._states))
        )

    def __hash__(self):
        return hash((tuple(self._order), tuple(self._states)))

    def items(self):
        return zip(self._order, self._states)
