from typing import Any, Optional, Union, Sequence, Type
import numpy as np

# Use the Generic Interface
from src.field_dynamic_system.core.state.interfaces import StateSpace


# --- 1. THE GENERIC BASE CLASS ---

class StaticStateSystem:
    """
    A System that holds the Initial Configuration of N entities.
    """

    def __init__(self,
                 initial_state: Any,
                 state_space: Optional[StateSpace] = None):
        self.state_space = state_space
        self.initial_state = initial_state

        # Save the class for later OOP inflation
        self.state_class = type(initial_state[0]) if isinstance(initial_state, (list, tuple)) else type(initial_state)

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

    # --- NEW: RETRIEVAL API ---

    def get_raw_state_space(self) -> Any:
        """
        Returns the raw ingredients required to construct or define the StateSpace.
        - For Discrete Systems: Returns an np.ndarray of configuration states.
        - For Continuous Systems: Returns bounds, radii, or parametric definitions.
        """
        raise NotImplementedError

    def get_lazy_state_space(self) -> StateSpace:
        raise NotImplementedError

    def get_state_space(self) -> StateSpace:
        raise NotImplementedError


# --- 2. THE CONCRETE IMPLEMENTATION ---

class DiscreteStaticStateSystem(StaticStateSystem):
    """
    Implementation for Discrete States (Abstract Strings or Discrete Vectors).
    """

    def _process_initial_state(self, initial_state: Any) -> np.ndarray:
        # Robust unwrapping (Identical to what we built for Topology)
        def extract_raw(obj):
            if hasattr(obj, 'value'): return obj.value
            if hasattr(obj, 'values'): return obj.values
            if hasattr(obj, 'vector'): return obj.vector
            if hasattr(obj, 'state'): return obj.state
            return obj

        if isinstance(initial_state, (list, tuple)) and not isinstance(initial_state, str):
            if len(initial_state) > 0 and hasattr(initial_state[0], '__class__'):
                test_val = extract_raw(initial_state[0])
                if test_val is not initial_state[0]:
                    return np.array([extract_raw(s) for s in initial_state])
            return np.array(initial_state)

        return extract_raw(initial_state)

    @classmethod
    def from_raw_data(cls,
                      raw_initial_state: Union[Any, np.ndarray],
                      state_space: StateSpace,
                      num_entities: Optional[int] = None,
                      state_class: Optional[Type] = None):
        instance = cls.__new__(cls)
        instance.state_space = state_space
        instance.state_class = state_class

        # FIXED: Use np.asarray to avoid NumPy 2.0 strict copy errors
        data = np.asarray(raw_initial_state)

        if num_entities is not None:
            if data.shape[0] != num_entities:
                if data.ndim == 0:
                    instance._raw_configuration = np.full((num_entities,), data)
                elif data.ndim == 1:
                    instance._raw_configuration = np.tile(data, (num_entities, 1))
                else:
                    instance._raw_configuration = np.tile(data, (num_entities,) + (1,) * (data.ndim - 1))
            else:
                instance._raw_configuration = data
        else:
            if data.ndim == 0:
                instance._raw_configuration = data.reshape(1)
            else:
                instance._raw_configuration = data

        instance.initial_state = raw_initial_state
        return instance

    # --- NEW: IMPLEMENTING THE RETRIEVAL METHODS ---

    def get_raw_state_space(self) -> np.ndarray:
        """Returns the raw configuration matrix representing all agents."""
        return self._raw_configuration

    def get_lazy_state_space(self) -> StateSpace:
        """Returns a high-performance StateSpace powered by Lazy Proxies."""
        if self.state_space is None or self.state_class is None:
            raise ValueError("Missing state_space or state_class reference to build Lazy space.")

        dim = self._raw_configuration.shape[1] if self._raw_configuration.ndim > 1 else 1

        return self.state_space.__class__.from_raw_data(
            raw_data=self._raw_configuration,
            wrapper=self.state_class,
            dim=dim
        )

    def get_state_space(self) -> StateSpace:
        """Returns the fully instantiated OOP StateSpace."""
        if self.state_space is None or self.state_class is None:
            raise ValueError("Missing state_space or state_class reference to build objects.")

        objects = []
        for val in self._raw_configuration:
            clean_val = tuple(val.tolist()) if hasattr(val, 'tolist') else val
            objects.append(self.state_class(clean_val))

        # Constructor Routing based on StateSpace Type
        space_class = self.state_space.__class__
        space_name = space_class.__name__

        if "VectorStateSpace" in space_name:
            dim = self._raw_configuration.shape[1] if self._raw_configuration.ndim > 1 else 1
            return space_class(vectors=objects, dim=dim)
        else:
            return space_class(states=objects)