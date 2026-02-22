import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Callable
import jax.numpy as jnp

from src.field_dynamic_system.clock.interfaces import IInternalClock
from src.field_dynamic_system.operator import InteractionContext, IOperator
from src.field_dynamic_system.systems.static.interface import ISystem
from src.field_dynamic_system.systems.static.topology import ContinuousStaticTopologySystem, \
    DiscreteStaticTopologySystem


class TopologyDynamicSystem(ISystem, ABC):
    """
    A unified system for Topology-based evolution.
    Supports both strict OOP Dependency Injection and bare-metal DOD instantiation
    for GPU/TPU acceleration.
    """

    def __init__(self,
                 topology_system: 'StaticTopologySystem',
                 operator: 'IOperator',
                 clock: 'IInternalClock',
                 is_dynamic: bool = False):
        """
        Strict OOP Constructor.
        Expects a fully initialized StaticTopologySystem as the spatial database.
        """
        self._spatial_db = topology_system
        self._operator = operator
        self._clock = clock
        self.is_dynamic = is_dynamic

        # Extract raw data directly from the injected spatial database
        self._initial_raw_state = copy.deepcopy(self._spatial_db.get_raw_state())
        self._current_raw_state = copy.deepcopy(self._initial_raw_state)
        self._current_topology_data = self._spatial_db.get_raw_data()

        # --- ZERO-COST ROUTING ---
        if not self.is_dynamic:
            self.update_topology = self._do_nothing

        # Route physics execution to the OOP path
        self._apply_physics = self._apply_oop_physics

    @classmethod
    def from_raw_data(cls,
                      raw_initial_state: Any,
                      raw_topology_data: Any,
                      raw_operator_fn: Callable,  # Accepts the pure selection_strategy directly
                      clock: 'IInternalClock',
                      is_dynamic: bool = False,
                      state_space: Optional['StateSpace'] = None,
                      state_class: Optional[Type] = None) -> 'TopologyDynamicSystem':
        """
        High-Performance Factory (Bare Metal Path).
        Bypasses the OOP wrappers entirely for maximum JAX execution speed.
        """
        instance = cls.__new__(cls)

        instance._initial_raw_state = copy.deepcopy(raw_initial_state)
        instance._current_raw_state = copy.deepcopy(instance._initial_raw_state)
        instance._current_topology_data = copy.deepcopy(raw_topology_data)

        instance._operator = raw_operator_fn
        instance._clock = clock
        instance.is_dynamic = is_dynamic
        instance._state_space_ref = state_space
        instance._state_class_ref = state_class
        instance._spatial_db = None

        # --- ZERO-COST ROUTING ---
        if not instance.is_dynamic:
            instance.update_topology = instance._do_nothing

        # Route physics execution to the Bare Metal path
        instance._apply_physics = instance._apply_raw_physics

        return instance

    def _do_nothing(self) -> None:
        """Fast-path bypass for static topologies."""
        pass

    @abstractmethod
    def update_topology(self) -> None:
        """
        Rewires the topology based on self._current_raw_state.
        Concrete implementations must update self._current_topology_data.
        """
        pass

    # --- THE PHYSICS ROUTERS ---

    def _apply_oop_physics(self, context: 'InteractionContext') -> Any:
        """Passes the strict context to the IOperator wrapper."""
        return self._operator.observe(
            state=self._current_raw_state,
            context=context
        )

    def _apply_raw_physics(self, context: 'InteractionContext') -> Any:
        """
        Bare Metal Path: Bypasses the IOperator wrapper.
        Passes state and context directly to the pure JAX selection_strategy.
        """
        return self._operator(self._current_raw_state, context)

    # --- THE MASTER EXECUTION LOOP ---

    def apply_operator(self, context_kwargs: Dict[str, Any] = None) -> None:
        """The Core Execution Loop."""
        context_kwargs = context_kwargs or {}

        # 1. Rewire if dynamic
        self.update_topology()

        # 2. Safely pack the topology into the global_params dictionary
        user_params = context_kwargs.get('global_params') or {}
        merged_params = {
            **user_params,
            'topology_data': self._current_topology_data
        }

        # 3. Build the strict, immutable InteractionContext
        context = InteractionContext(
            rng_key=context_kwargs.get('rng_key'),
            action_id=context_kwargs.get('action_id', 0),
            global_params=merged_params
        )

        # 4. Apply Physics via the dynamically assigned router
        self._current_raw_state = self._apply_physics(context)

        # 5. Advance time
        self._clock.tick(1)

    # --- RAW DATA LIFECYCLE METHODS ---

    def get_raw_data(self) -> Any:
        return self._current_raw_state

    def reset(self) -> None:
        self._current_raw_state = copy.deepcopy(self._initial_raw_state)
        if self._spatial_db:
            self._spatial_db.reset()
            self._current_topology_data = self._spatial_db.get_raw_data()
        self._clock.reset()

    def get_raw_state_space(self) -> Any:
        """
        Satisfies the ISystem requirement for spatial bounds retrieval.
        Routes the request to the spatial database or the injected raw state space.
        """
        # Path A: OOP Flow (Ask the Spatial DB)
        if self._spatial_db is not None:
            return self._spatial_db.get_raw_state_space()

        # Path B: Bare Metal Flow (Use the injected raw data)
        if hasattr(self, '_state_space_ref') and self._state_space_ref is not None:
            if hasattr(self._state_space_ref, 'get_raw_data'):
                return self._state_space_ref.get_raw_data()
            return self._state_space_ref

        return None



class ContinuousTopologyDynamicSystem(TopologyDynamicSystem):
    """
    Concrete Dynamic System for Continuous Spaces.
    Manages the evolution of states constrained by moving geometric boundaries.
    """

    def __init__(self,
                 topology_system: 'ContinuousStaticTopologySystem',
                 operator: 'IOperator',
                 clock: 'IInternalClock',
                 is_dynamic: bool = True):  # Continuous topologies almost always move with the state

        # Standard OOP initialization
        super().__init__(
            topology_system=topology_system,
            operator=operator,
            clock=clock,
            is_dynamic=is_dynamic
        )

    @classmethod
    def from_raw_data(cls,
                      raw_initial_state: Any,
                      raw_topology_fn: Callable[[jnp.ndarray], Any],  # MUST be a pure JAX function
                      raw_operator_fn: Callable,  # The pure JAX physics logic
                      clock: 'IInternalClock',
                      is_dynamic: bool = True,
                      state_space: Optional['StateSpace'] = None,
                      state_class: Optional[Type] = None) -> 'ContinuousTopologyDynamicSystem':
        """
        High-Performance Factory (Bare Metal Path).
        """
        # Call the parent factory to set up the base chassis
        # We pass a dummy None for raw_topology_data initially, as we will compute it immediately
        instance = super().from_raw_data(
            raw_initial_state=raw_initial_state,
            raw_topology_data=None,
            raw_operator_fn=raw_operator_fn,
            clock=clock,
            is_dynamic=is_dynamic,
            state_space=state_space,
            state_class=state_class
        )

        # Store the pure JAX topology boundary calculator
        instance._raw_topology_fn = raw_topology_fn

        # Compute the initial boundaries (T_0) immediately using the raw function
        instance._current_topology_data = instance._raw_topology_fn(instance._current_raw_state)

        return instance

    def update_topology(self) -> None:
        """
        The Rewiring Step: Recalculates the bounding box/geometric limits
        based on the current continuous position.
        """
        if self._spatial_db is not None:
            # Path A: OOP Flow
            # Ask the topology object inside the database for the new raw limits
            self._current_topology_data = self._spatial_db.topology.get_raw_successor(
                self._current_raw_state
            )
        else:
            # Path B: Bare Metal Flow
            # Execute the pure JAX function to get the new raw limits
            self._current_topology_data = self._raw_topology_fn(
                self._current_raw_state
            )





class DiscreteTopologyDynamicSystem(TopologyDynamicSystem):
    """
    Concrete Dynamic System for Discrete Spaces (Graphs, Grids, Networks).
    Manages the evolution of states constrained by an Adjacency Matrix.
    """

    def __init__(self,
                 topology_system: 'DiscreteStaticTopologySystem',
                 operator: 'IOperator',
                 clock: 'IInternalClock',
                 is_dynamic: bool = False):  # Default to False: most grids/graphs are static

        # Standard OOP initialization
        super().__init__(
            topology_system=topology_system,
            operator=operator,
            clock=clock,
            is_dynamic=is_dynamic
        )

    @classmethod
    def from_raw_data(cls,
                      raw_initial_state: Any,
                      raw_topology_data: Any,  # The initial adjacency matrix (e.g., jax.experimental.sparse.BCOO)
                      raw_operator_fn: Callable,  # Pure JAX physics logic
                      clock: 'IInternalClock',
                      is_dynamic: bool = False,
                      raw_topology_fn: Optional[Callable[[jnp.ndarray, Any], Any]] = None,
                      state_space: Optional['StateSpace'] = None,
                      state_class: Optional[Type] = None) -> 'DiscreteTopologyDynamicSystem':
        """
        High-Performance Factory (Bare Metal Path).
        """
        if is_dynamic and raw_topology_fn is None:
            raise ValueError(
                "A dynamic discrete system running on the Bare Metal path MUST provide "
                "a 'raw_topology_fn' to recalculate the adjacency matrix."
            )

        # Call the parent factory to set up the base chassis
        instance = super().from_raw_data(
            raw_initial_state=raw_initial_state,
            raw_topology_data=raw_topology_data,  # Inject the initial matrix directly
            raw_operator_fn=raw_operator_fn,
            clock=clock,
            is_dynamic=is_dynamic,
            state_space=state_space,
            state_class=state_class
        )

        # Store the pure JAX graph rewiring function (if dynamic)
        instance._raw_topology_fn = raw_topology_fn

        return instance

    def update_topology(self) -> None:
        """
        The Rewiring Step: Recalculates the Adjacency Matrix based on
        the current state and the previous matrix.
        """
        if self._spatial_db is not None:
            # Path A: OOP Flow
            # Ask the spatial database's topology to compute the new matrix
            # Note: The topology needs to know both the current matrix and the state to rewire
            self._current_topology_data = self._spatial_db.topology.get_raw_rewired_matrix(
                current_state_raw=self._current_raw_state,
                current_matrix=self._current_topology_data
            )
        else:
            # Path B: Bare Metal Flow
            # Execute the pure JAX rewiring function directly on the GPU
            self._current_topology_data = self._raw_topology_fn(
                self._current_raw_state,
                self._current_topology_data
            )