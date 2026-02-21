import copy
from typing import Any, Optional, Dict, Type

# Core Framework Imports
from src.field_dynamic_system.systems.static.interface import ISystem
from src.field_dynamic_system.core.state.interfaces import StateSpace
from src.field_dynamic_system.clock.interfaces import IInternalClock

# Component Imports
from src.field_dynamic_system.systems.static.state import StaticStateSystem
from src.field_dynamic_system.operator  import IOperator, InteractionContext


class DynamicSystem(ISystem):
    """
    The concrete implementation of a pure 4-Tuple Dynamic System: (s, S, Theta, C).
    Acts as an isolated, time-aware state machine.
    """

    def __init__(self,
                 state_system: StaticStateSystem,
                 operator: IOperator,
                 clock: IInternalClock):
        """
        The OOP 'Booster Rocket' Path.
        Extracts raw data and types from a StaticStateSystem, then discards it.
        """
        # 1. Extract and isolate the raw data (O(1) runtime access)
        self._initial_raw_state = copy.deepcopy(state_system.get_raw_state(entity_id=-1))
        self._current_raw_state = copy.deepcopy(self._initial_raw_state)

        # 2. Extract OOP references for later reconstruction
        self._state_space_ref = getattr(state_system, 'state_space', None)
        self._state_class_ref = getattr(state_system, 'state_class', None)

        # 3. Attach the Engine (Operator) and Ledger (Clock)
        self._operator = operator
        self._clock = clock

        # 4. Record Genesis State (T=0)
        self._record_current_state()

    @classmethod
    def from_raw_data(cls,
                      raw_initial_state: Any,
                      operator: IOperator,
                      clock: IInternalClock,
                      state_space: Optional[StateSpace] = None,
                      state_class: Optional[Type] = None) -> 'DynamicSystem':
        """
        High-Performance Factory (Bare Metal Path).
        Bypasses the StaticStateSystem entirely for maximum serialization/loop speed.
        """
        instance = cls.__new__(cls)

        # Inject bare-metal data safely
        instance._initial_raw_state = copy.deepcopy(raw_initial_state)
        instance._current_raw_state = copy.deepcopy(instance._initial_raw_state)

        # Inject references and components
        instance._state_space_ref = state_space
        instance._state_class_ref = state_class
        instance._operator = operator
        instance._clock = clock

        # Initialize the clock ledger
        instance._record_current_state()

        return instance

    # --- 4-TUPLE PROPERTIES (Read-Only) ---

    @property
    def state(self) -> Any:
        return self._current_raw_state

    @property
    def state_space(self) -> Optional[StateSpace]:
        return self._state_space_ref

    @property
    def operator(self) -> IOperator:
        return self._operator

    @property
    def clock(self) -> IInternalClock:
        return self._clock

    # --- CORE EVOLUTION ---

    def apply_operator(self, context_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """
        The Master Transition: Theta(s_t) -> s_{t+1}
        """
        # 1. Build interaction context
        context_data = context_kwargs or {}
        context = InteractionContext(**context_data)

        # 2. Apply the Operator to get the next state
        next_state = self._operator.observe(self._current_raw_state, context)

        # 3. Mutate the System Reality
        self._current_raw_state = next_state

        # 4. Log the Transition in the Clock
        self._record_current_state()

    def _record_current_state(self) -> None:
        """Helper to package the snapshot for the temporal ledger."""
        snapshot = {
            "state": copy.deepcopy(self._current_raw_state)
        }
        self._clock.record_snapshot(snapshot)

    # --- ISYSTEM CONTRACT FULFILLMENT ---

    def reset(self) -> None:
        """Restores the system to Time T=0 and clears history."""
        self._current_raw_state = copy.deepcopy(self._initial_raw_state)
        self._clock.reset()
        self._record_current_state()

    def get_raw_data(self) -> Any:
        """Returns the raw state array (Satisfies ISystem master loop)."""
        return self._current_raw_state

    def get_raw_state_space(self) -> Any:
        """Returns the underlying spatial bounds or raw matrix if available."""
        if hasattr(self._state_space_ref, 'get_raw_data'):
            return self._state_space_ref.get_raw_data()
        elif hasattr(self._state_space_ref, 'get_raw_state_space'):
            return self._state_space_ref.get_raw_state_space()
        return None