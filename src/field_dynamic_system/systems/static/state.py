from typing import Any, Optional, Union, Sequence, Type
import numpy as np

# Use the Generic Interface
from src.field_dynamic_system.core.state.interfaces import StateSpace


class StaticStateSystem:
    """
    A System that holds the Initial Configuration of N entities.
    """

    def __init__(self,
                 initial_state: Any,
                 state_space: Optional[StateSpace] = None):
        self.state_space = state_space
        self.initial_state = initial_state
        self._raw_configuration = self._process_initial_state(initial_state)

    @classmethod
    def from_raw_data(cls,
                      raw_initial_state: Union[Any, np.ndarray],
                      state_space: StateSpace,
                      num_entities: Optional[int] = None,
                      state_class: Optional[Type] = None):
        """
        Factory for Raw Data.
        Args:
            num_entities: N (Number of entities). Essential for broadcasting single values.
        """
        raise NotImplementedError

    def _process_initial_state(self, initial_state: Any) -> Any:
        raise NotImplementedError

    def get_state(self, entity_id: int = 0) -> Any:
        if hasattr(self.initial_state, '__getitem__') and hasattr(self.initial_state, '__len__') and not isinstance(
                self.initial_state, str):
            return self.initial_state[entity_id]
        return self.initial_state

    def get_raw_state(self, entity_id: int = -1) -> Any:
        # If -1, return the whole configuration matrix
        if entity_id == -1:
            return self._raw_configuration

        if hasattr(self._raw_configuration, '__getitem__'):
            return self._raw_configuration[entity_id]
        return self._raw_configuration


class DiscreteStaticStateSystem(StaticStateSystem):
    """
    Implementation for Discrete States (Abstract Strings or Discrete Vectors).
    """

    def _process_initial_state(self, initial_state: Any) -> np.ndarray:
        if isinstance(initial_state, (list, tuple)) and not isinstance(initial_state, str):
            return np.array([s.values for s in initial_state])
        if hasattr(initial_state, 'values'):
            return initial_state.values
        return initial_state

    @classmethod
    def from_raw_data(cls,
                      raw_initial_state: Union[Any, np.ndarray],
                      state_space: StateSpace,
                      num_entities: Optional[int] = None,  # <--- OPTIONAL
                      state_class: Optional[Type] = None):
        """
        Smart Factory:
        1. If num_entities is MISSING -> Infers N from input array length.
        2. If num_entities is PROVIDED -> Broadcasts single value to N.
        """
        instance = cls.__new__(cls)
        instance.state_space = state_space

        # 1. Standardize Input to Numpy Array
        # Copy=False allows us to use existing arrays without overhead
        data = np.array(raw_initial_state, copy=False)

        # 2. Logic: Inference vs Broadcasting

        # CASE A: User provided N (Potentially Broadcasting)
        if num_entities is not None:
            # Check if input matches N. If not, Broadcast.
            # Heuristic: If first dim != N, assume it's a single state to broadcast.
            if data.shape[0] != num_entities:
                # Tile the single state N times
                # Handle scalar vs vector input
                if data.ndim == 0:  # Scalar (e.g. 5)
                    instance._raw_configuration = np.full((num_entities,), data)
                elif data.ndim == 1:  # Vector (e.g. [0,0,0])
                    instance._raw_configuration = np.tile(data, (num_entities, 1))
                else:
                    # Only tile if strictly necessary (rare for higher dims)
                    instance._raw_configuration = np.tile(data, (num_entities,) + (1,) * (data.ndim - 1))
            else:
                # Already N items
                instance._raw_configuration = data

        # CASE B: User did NOT provide N (Inference)
        else:
            # We assume the input IS the full configuration
            # N = length of the array
            if data.ndim == 0:
                # Edge Case: Passed a single scalar without N.
                # e.g. from_raw_data(5). Is it 1 agent with value 5? Yes.
                instance._raw_configuration = data.reshape(1)
            else:
                instance._raw_configuration = data

        # Store virtual initial state
        instance.initial_state = raw_initial_state

        return instance