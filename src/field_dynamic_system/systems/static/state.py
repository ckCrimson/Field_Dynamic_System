from abc import ABC, abstractmethod
from typing import Any, Optional, Type, List, Union
import jax.numpy as jnp
import numpy as np

# Core Interfaces
from src.field_dynamic_system.core.state.interfaces import StateSpace, IDiscreteStateSpace
from src.field_dynamic_system.core.state.discrete import LazyDiscreteStateSpace


# =========================================================
# 1. ABSTRACT BASE: StaticStateSystem
# =========================================================

class StaticStateSystem(ABC):
    """
    Base Snapshot System.
    Stores mandatory initial state and optional state space.
    """

    def __init__(self, initial_state: Any, state_space: Optional[StateSpace] = None):
        # We store the object reference as requested
        self._state_obj = initial_state
        self._space_ref = state_space

        # Validation: Ensure state object has .value attribute
        if not hasattr(initial_state, 'value'):
            # In strict mode raise error, but for flexibility we might allow primitives
            # assuming the primitive IS the value.
            pass

    @property
    def state_space(self) -> Optional[StateSpace]:
        return self._space_ref

    @abstractmethod
    def get_state(self) -> Any:
        """Returns the high-level State Object."""
        pass

    @abstractmethod
    def get_raw_state(self) -> Any:
        """Returns the raw value of the initial state."""
        pass

    @abstractmethod
    def get_raw_data(self) -> List[Any]:
        """Returns [initial_state_raw_value, Any (context/space data)]"""
        pass

    @classmethod
    @abstractmethod
    def from_raw_data(cls,
                      initial_state_raw_value: Any,
                      **kwargs) -> 'StaticStateSystem':
        """Factory from raw values."""
        pass


# =========================================================
# 2. CONCRETE: DiscreteStaticStateSystem
# =========================================================

class DiscreteStaticStateSystem(StaticStateSystem):
    """
    Discrete Implementation.
    Stores fast arrays for IDs and Raw State Values.
    """

    def __init__(self,
                 initial_state: Any,
                 state_space: Optional[IDiscreteStateSpace] = None):

        super().__init__(initial_state, state_space)

        # 1. Extract Initial State Raw Value
        if hasattr(initial_state, 'value'):
            self._raw_state_val = initial_state.value
        else:
            self._raw_state_val = initial_state  # Fallback for primitives

        # 2. Fast Arrays Storage (Initialize Empty)
        self._all_raw_states = None
        self._all_ids = None

        # 3. Extract Data from Space (If provided)
        if state_space is not None:
            # Assumption: state_space has methods/properties to access raw data
            # We fetch raw states and IDs to store in fast arrays

            # Helper to extract raw matrix (N, D)
            if hasattr(state_space, 'get_matrix'):
                self._all_raw_states = jnp.array(state_space.get_matrix())
            elif hasattr(state_space, 'raw_states'):
                self._all_raw_states = jnp.array(state_space.raw_states)

            # Helper to extract IDs (0..N)
            # Usually implied by index in the array, but we make it explicit
            if self._all_raw_states is not None:
                self._all_ids = jnp.arange(len(self._all_raw_states))

        # Store class reference for reconstruction
        self._state_cls = type(initial_state)
        # Default space class
        self._space_cls = LazyDiscreteStateSpace

        # --- FACTORY ---

    @classmethod
    def from_raw_data(cls,
                      initial_state_raw_value: Any,
                      state_class: Type,
                      list_of_raw_states: List[Any] = None,
                      state_space_class: Type = LazyDiscreteStateSpace) -> 'DiscreteStaticStateSystem':

        # 1. Bypass Init
        instance = cls.__new__(cls)

        # 2. Assign Initial State Data
        instance._raw_state_val = initial_state_raw_value
        instance._state_cls = state_class

        # Inflate the initial state object immediately (as per "assign the initial_state variable")
        # We assume constructor takes value as arg
        instance._state_obj = state_class(initial_state_raw_value)

        # 3. Store Raw Lists in Fast Arrays
        instance._all_raw_states = None
        instance._all_ids = None

        if list_of_raw_states is not None:
            # Convert list to JAX Array
            raw_arr = jnp.array(list_of_raw_states)
            if raw_arr.ndim == 1: raw_arr = raw_arr.reshape(-1, 1)

            instance._all_raw_states = raw_arr
            instance._all_ids = jnp.arange(len(raw_arr))  # 0 to N-1

        # 4. Set Space Class Reference (Lazy default or User provided)
        instance._space_cls = state_space_class
        instance._space_ref = None  # Lazy, created in get_space()

        return instance

    # --- GETTERS ---

    def get_state(self) -> Any:
        return self._state_obj

    def get_raw_state(self) -> Any:
        return self._raw_state_val

    def get_raw_data(self) -> List[Any]:
        # Returns [initial_state_raw_value, all_raw_states_array]
        return [self._raw_state_val, self._all_raw_states]

    @property
    def state_space(self) -> StateSpace:
        """
        Returns state space.
        By default returns a LazyDiscreteStateSpace constructed from stored raw data.
        Can be overridden by user to return custom class.
        """
        # If we already have a reference (from __init__), return it
        if self._space_ref is not None:
            return self._space_ref

        # If we have raw data, construct the Lazy Space on the fly
        if self._all_raw_states is not None:
            # Create instance of _space_cls (Default LazyDiscreteStateSpace)
            # Signature: (raw_data, wrapper_class)
            self._space_ref = self._space_cls(
                raw_data=self._all_raw_states,
                wrapper_class=self._state_cls
            )
            return self._space_ref

        return None