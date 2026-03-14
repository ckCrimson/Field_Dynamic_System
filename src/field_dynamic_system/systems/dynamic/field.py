import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Callable
import jax.numpy as jnp

from src.field_dynamic_system.clock.interfaces import IInternalClock
from src.field_dynamic_system.core import StateSpace
from src.field_dynamic_system.operator import InteractionContext, IOperator
from src.field_dynamic_system.systems.static.interface import ISystem
from src.field_dynamic_system.systems.static.field import AbstractStaticFieldGeneratorSystem
from src.field_dynamic_system.generator.generator_interfaces import IFieldGenerator


class FieldDynamicSystem(ISystem, ABC):
    """
    A unified system for evolving an entity within a dynamic field.
    Supports strict OOP Injection and bare-metal JAX DOD execution.
    """

    def __init__(self,
                 field_system: AbstractStaticFieldGeneratorSystem,
                 field_generator: IFieldGenerator,
                 entity_operator: IOperator,
                 clock: IInternalClock,
                 is_dynamic_field: bool = False):

        self._spatial_db = field_system
        self._field_generator = field_generator
        self._entity_operator = entity_operator
        self._clock = clock
        self.is_dynamic_field = is_dynamic_field

        # Entity State
        self._initial_raw_state = copy.deepcopy(self._spatial_db.get_raw_state())
        self._current_raw_state = copy.deepcopy(self._initial_raw_state)

        # Field State
        self._current_field_data = self._spatial_db.get_raw_fields()

        # --- ZERO-COST ROUTING ---
        if not self.is_dynamic_field:
            self.update_field = self._do_nothing

        self._apply_entity_physics = self._apply_oop_physics

    @classmethod
    def from_raw_data(cls,
                      raw_initial_state: Any,
                      raw_field_data: Any,
                      raw_entity_operator_fn: Callable,
                      clock: IInternalClock,
                      is_dynamic_field: bool = False,
                      raw_field_update_fn: Optional[Callable] = None,  # Matches generate_raw_multi_step
                      state_space: Optional[StateSpace] = None) -> 'FieldDynamicSystem':

        instance = cls.__new__(cls)

        # Entity Data
        instance._initial_raw_state = copy.deepcopy(raw_initial_state)
        instance._current_raw_state = copy.deepcopy(instance._initial_raw_state)

        # Field Data
        instance._initial_field_data = copy.deepcopy(raw_field_data)  # Required for reset()
        instance._current_field_data = copy.deepcopy(raw_field_data)

        instance._entity_operator = raw_entity_operator_fn
        instance._clock = clock
        instance.is_dynamic_field = is_dynamic_field
        instance._raw_field_update_fn = raw_field_update_fn

        instance._state_space_ref = state_space
        instance._spatial_db = None
        instance._field_generator = None

        # --- ZERO-COST ROUTING ---
        if not instance.is_dynamic_field:
            instance.update_field = instance._do_nothing

        instance._apply_entity_physics = instance._apply_raw_physics

        return instance

    def _do_nothing(self, context_kwargs: Dict[str, Any] = None) -> None:
        pass

    @abstractmethod
    def update_field(self, context_kwargs: Dict[str, Any] = None) -> None:
        """Concrete implementations must update self._current_field_data."""
        pass

    @abstractmethod
    def collapse_field(self) -> None:
        """
        Forces the field to unity at the known entity state and 0 elsewhere.
        Derived from Bayesian first principles: Observation = absolute certainty.
        """
        pass

    # --- THE PHYSICS ROUTERS ---
    def _apply_oop_physics(self, context: InteractionContext) -> Any:
        return self._entity_operator.observe(state=self._current_raw_state, context=context)

    def _apply_raw_physics(self, context: InteractionContext) -> Any:
        return self._entity_operator(self._current_raw_state, context)

    # --- THE MASTER EXECUTION LOOP ---
    def apply_operator(self, context_kwargs: Dict[str, Any] = None) -> None:
        context_kwargs = context_kwargs or {}

        # 1. Evolve the Field Tensor/Params BEFORE the entity moves
        self.update_field(context_kwargs)

        # 2. Pack the field data for the entity's context
        user_params = context_kwargs.get('global_params') or {}
        merged_params = {**user_params, 'field_data': self._current_field_data}

        context = InteractionContext(
            rng_key=context_kwargs.get('rng_key'),
            action_id=context_kwargs.get('action_id', 0),
            global_params=merged_params
        )

        # 3. Move the Entity (The "Observation")
        self._current_raw_state = self._apply_entity_physics(context)

        # 4. FIRST PRINCIPLE: Sync the field to the new absolute reality
        self.collapse_field()

        # 5. Advance Time
        self._clock.tick(1)

    # --- LIFECYCLE METHODS ---
    def reset(self) -> None:
        """ Forces the system to return to its initial/default state. """
        # 1. Reset Entity State
        self._current_raw_state = copy.deepcopy(self._initial_raw_state)

        # 2. Reset Field State
        if self._spatial_db is not None:
            # Path A: OOP Flow
            self._spatial_db.reset()
            self._current_field_data = self._spatial_db.get_raw_fields()
        else:
            # Path B: Bare Metal Flow
            self._current_field_data = copy.deepcopy(self._initial_field_data)

        # 3. Reset Time
        self._clock.reset()

    def get_raw_data(self) -> Any:
        """ Returns the state of the entity. """
        return self._current_raw_state

    def get_raw_field_data(self) -> Any:
        """ Dedicated getter for the field tensor/parameters. """
        return self._current_field_data

    def set_raw_field_data(self, new_field_data: Any) -> None:
        """ Manually overwrites the field state (useful for test scripts). """
        self._current_field_data = new_field_data

    def get_raw_state_space(self) -> Any:
        """ Returns the spatial coordinates or bounds of the system. """
        if self._spatial_db is not None:
            return self._spatial_db.get_raw_state_space()

        if hasattr(self, '_state_space_ref') and self._state_space_ref is not None:
            if hasattr(self._state_space_ref, 'get_raw_data'):
                return self._state_space_ref.get_raw_data()
            return self._state_space_ref

        return None


class DiscreteFieldDynamicSystem(FieldDynamicSystem):
    """
    Manages dense/sparse arrays mapped to discrete spaces.
    Requires topology data to perform field updates.
    """

    @classmethod
    def from_raw_data(cls,
                      raw_initial_state: Any,
                      raw_field_data: Any,
                      raw_topology_data: Any,  # <--- Discrete fields NEED the graph matrix
                      raw_entity_operator_fn: Callable,
                      clock: IInternalClock,
                      is_dynamic_field: bool = False,
                      raw_field_update_fn: Optional[Callable] = None,
                      state_space: Optional[StateSpace] = None) -> 'DiscreteFieldDynamicSystem':

        if is_dynamic_field and raw_field_update_fn is None:
            raise ValueError("Dynamic discrete fields require 'raw_field_update_fn'.")

        instance = super().from_raw_data(
            raw_initial_state=raw_initial_state,
            raw_field_data=raw_field_data,
            raw_entity_operator_fn=raw_entity_operator_fn,
            clock=clock,
            is_dynamic_field=is_dynamic_field,
            raw_field_update_fn=raw_field_update_fn,
            state_space=state_space
        )
        # Store topology strictly for field matrix multiplication
        instance._current_topology_data = raw_topology_data
        return instance

    def update_field(self, context_kwargs: Dict[str, Any] = None) -> None:
        """Evolves the N-dimensional field tensor using the Adjacency Matrix."""
        context_kwargs = context_kwargs or {}

        # Extract the user's step count (default to 1 if not provided)
        steps = context_kwargs.get('global_params', {}).get('steps', 1)

        if self._spatial_db is not None:
            # Path A: OOP Flow
            self._current_field_data = self._field_generator.generate_raw_multi_step(
                raw_field=self._current_field_data,
                raw_topology=self._spatial_db.topology.get_raw_data(),
                steps=steps
            )
        else:
            # Path B: Bare Metal Flow
            self._current_field_data = self._raw_field_update_fn(
                raw_field=self._current_field_data,
                raw_topo=self._current_topology_data,
                steps=steps
            )

    def collapse_field(self) -> None:
        """
        Generic discrete collapse: Sets the current state index to 1
        (preserving exact shape and dtype) and all other states to 0.
        """
        current_dtype = self._current_field_data.dtype
        new_field = jnp.zeros_like(self._current_field_data)
        unity = jnp.array(1, dtype=current_dtype)
        self._current_field_data = new_field.at[self._current_raw_state].set(unity)

    @classmethod
    def compile_bare_metal(cls,
                           topology: Any,
                           generator: IFieldGenerator,
                           start_state: Any,
                           entity_operator_fn: Callable,
                           clock: IInternalClock,
                           expansion_depth: int = 0,
                           global_field_builder: Optional[
                               Callable[[list, int], Any]] = None) -> 'DiscreteFieldDynamicSystem':
        """
        The 'Bridge' Factory: Takes high-level OOP objects, extracts their raw
        mathematical tensors, and compiles the bare-metal DOD execution chassis.
        """
        print(f"-> [Compiler] Baking Topology (Depth {expansion_depth})...")
        if expansion_depth > 0:
            topology.expand_frontier([start_state], depth=expansion_depth)

        num_nodes = len(topology._id_to_raw)
        adj_matrix = topology.adjacency_matrix
        start_node_id = topology._raw_to_id[start_state]

        print("-> [Compiler] Extracting Kernel Weights...")
        # Automatically pull weights if the generator's kernel supports it
        if hasattr(generator, 'kernel') and hasattr(generator.kernel, 'compute_raw_batch'):
            weights = generator.kernel.compute_raw_batch(adj_matrix.indices).flatten()
            dtype = weights.dtype
        else:
            weights = jnp.ones(adj_matrix.indices.shape[0], dtype=jnp.float32)
            dtype = jnp.float32

        raw_topology_tuple = (
            adj_matrix.indices[:, 0],
            adj_matrix.indices[:, 1],
            weights,
            num_nodes
        )

        print("-> [Compiler] Initializing Dirac Delta Field...")
        # Match the dtype of the kernel (e.g., complex64 for quantum, float32 for heat)
        unity = jnp.array(1, dtype=dtype)
        raw_initial_field = jnp.zeros((num_nodes, 1), dtype=dtype).at[start_node_id].set(unity)

        print("-> [Compiler] Constructing Global Field...")
        if global_field_builder:
            raw_global_field = global_field_builder(topology._id_to_raw, num_nodes)
        else:
            raw_global_field = None

        print("-> [Compiler] Binding JAX Physics Closure...")

        def raw_field_update_wrapper(raw_field, raw_topo, steps):
            return generator.generate_raw_multi_step(
                raw_field=raw_field,
                raw_topology=raw_topo,
                steps=steps,
                raw_global_field=raw_global_field
            )

        return cls.from_raw_data(
            raw_initial_state=start_node_id,
            raw_field_data=raw_initial_field,
            raw_topology_data=raw_topology_tuple,
            raw_entity_operator_fn=entity_operator_fn,
            clock=clock,
            is_dynamic_field=True,
            raw_field_update_fn=raw_field_update_wrapper
        )


class ContinuousFieldDynamicSystem(FieldDynamicSystem):
    """
    Manages continuous parametric fields (e.g., PyTrees of NN weights).
    Ignores discrete topology entirely.
    """

    @classmethod
    def from_raw_data(cls,
                      raw_initial_state: Any,
                      raw_field_params: Any,  # <--- PyTree of params, not a grid
                      raw_entity_operator_fn: Callable,
                      clock: IInternalClock,
                      is_dynamic_field: bool = False,
                      raw_field_update_fn: Optional[Callable] = None,
                      state_space: Optional[StateSpace] = None) -> 'ContinuousFieldDynamicSystem':

        if is_dynamic_field and raw_field_update_fn is None:
            raise ValueError("Dynamic continuous fields require 'raw_field_update_fn'.")

        instance = super().from_raw_data(
            raw_initial_state=raw_initial_state,
            raw_field_data=raw_field_params,
            raw_entity_operator_fn=raw_entity_operator_fn,
            clock=clock,
            is_dynamic_field=is_dynamic_field,
            raw_field_update_fn=raw_field_update_fn,
            state_space=state_space
        )
        return instance

    def update_field(self, context_kwargs: Dict[str, Any] = None) -> None:
        """Evolves the parameters of the mathematical field equation."""
        context_kwargs = context_kwargs or {}
        steps = context_kwargs.get('global_params', {}).get('steps', 1)

        if self._spatial_db is not None:
            # Path A: OOP Flow
            self._current_field_data = self._field_generator.generate_raw_multi_step(
                raw_field=self._current_field_data,
                raw_topology=None,
                steps=steps
            )
        else:
            # Path B: Bare Metal Flow
            self._current_field_data = self._raw_field_update_fn(
                params=self._current_field_data,
                steps=steps
            )

    def collapse_field(self) -> None:
        """
        Continuous collapse.
        Will eventually reset the parametric PyTree to represent a localized Dirac delta
        or Gaussian centered perfectly at self._current_raw_state.
        """
        pass  # To be implemented when Continuous systems are built out