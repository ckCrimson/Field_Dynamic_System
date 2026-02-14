from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from typing import Optional, Any, Union

from src.field_dynamic_system.core.field.mappings import IFieldMapper, DiscreteFieldMapper
from src.field_dynamic_system.core.field.algebra import IFieldAlgebra
from src.field_dynamic_system.generator.generator_interfaces import IFieldGenerator

from src.field_dynamic_system.systems.static.state import StaticStateSystem
from src.field_dynamic_system.systems.static.topology import StaticTopologySystem



class AbstractStaticFieldGeneratorSystem(ABC):
    """
    Generic Base Orchestrator for Field Evolution.
    """

    def __init__(self,
                 generator: IFieldGenerator,
                 field_algebra: IFieldAlgebra,
                 is_raw_mode: bool):

        self.generator = generator
        self.algebra = field_algebra
        self.is_raw_mode = is_raw_mode

        self.current_mapper: Optional[IFieldMapper] = None
        self.current_raw_data: Any = None

        # --- FEATURE 1 & 2: DYNAMIC STATE MUTATION ---

    def save_generated_field(self, steps: int, **kwargs) -> None:
        """ Runs the simulation for N steps and permanently overwrites the internal state. """
        if self.is_raw_mode:
            self.current_raw_data = self.generate_raw_field(steps, **kwargs)
        else:
            self.current_mapper = self.generate_field(steps, **kwargs)
            self._sync_raw_from_mapper()

    def set_field(self, field_data: Union[IFieldMapper, jax.Array]) -> None:
        """ Injects a custom field into the system at any time. """
        if isinstance(field_data, IFieldMapper):
            self.current_mapper = field_data
            self._sync_raw_from_mapper()
        else:
            self.current_raw_data = field_data
            if not self.is_raw_mode:
                self.current_mapper = self._wrap_raw_to_mapper(field_data)

    # --- FEATURE 3 & 4: ACCESSORS ---

    def get_field(self) -> IFieldMapper:
        """ Returns the heavy OOP Mapper. Lazily wraps raw data if in raw mode. """
        if self.is_raw_mode and self.current_mapper is None:
            self.current_mapper = self._wrap_raw_to_mapper(self.current_raw_data)
        return self.current_mapper

    def get_raw_fields(self) -> Any:
        """ Returns the lightning-fast raw JAX array/function. """
        return self.current_raw_data

    # --- ABSTRACT CONTRACTS (Including Features 5 & 6) ---

    @abstractmethod
    def generate_field(self, steps: int, **kwargs) -> IFieldMapper:
        pass

    @abstractmethod
    def generate_raw_field(self, steps: int, **kwargs) -> Any:
        pass

    @abstractmethod
    def clear_field(self) -> None:
        """ FEATURE 5: Resets the field to an impulse at the origin. """
        pass

    @abstractmethod
    def get_state_space(self) -> Any:
        """ FEATURE 6a: Returns the OOP State Space. """
        pass

    @abstractmethod
    def get_raw_state_space(self) -> Any:
        """ FEATURE 6b: Returns the Raw State Space coordinates/bounds. """
        pass

    @abstractmethod
    def _wrap_raw_to_mapper(self, raw_data: Any) -> IFieldMapper:
        pass

    @abstractmethod
    def _sync_raw_from_mapper(self) -> None:
        pass



class DiscreteStaticFieldGeneratorSystem(AbstractStaticFieldGeneratorSystem):

    def __init__(self, generator: IFieldGenerator, field_algebra: IFieldAlgebra, is_raw_mode: bool):
        super().__init__(generator, field_algebra, is_raw_mode)

        # OOP Pointers
        self.state_system: Optional[StaticStateSystem] = None
        self.topology_system: Optional[StaticTopologySystem] = None

        # Raw Pointers
        self._raw_topology_tuple: Optional[tuple] = None
        self._state_space_ref: Any = None

    def _process_initial_state(self, initial_state):
        # Convert the raw list of Python tuples into a JAX array of coordinates
        return jnp.array(initial_state)

    def get_state_space(self):
        # Returns the OOP representation. For now, returning self is fine
        # until you build a dedicated DiscreteStateSpace class.
        return self

    def get_raw_state_space(self):
        # Returns the raw JAX array of coordinates
        return self._raw_configuration

    @property
    def shape(self):
        # The orchestrator uses this to find num_nodes
        return self._raw_configuration.shape

    @classmethod
    def from_systems(cls, generator: IFieldGenerator, field_algebra: IFieldAlgebra,
                     state_system: StaticStateSystem, topology_system: StaticTopologySystem,
                     initial_mapper: Optional[DiscreteFieldMapper] = None) -> 'DiscreteStaticFieldGeneratorSystem':
        instance = cls(generator, field_algebra, is_raw_mode=False)
        instance.state_system = state_system
        instance.topology_system = topology_system

        if initial_mapper:
            instance.set_field(initial_mapper)
        else:
            instance.clear_field()  # Auto-initialize to impulse
        return instance

    @classmethod
    def from_raw(cls, generator: IFieldGenerator, field_algebra: IFieldAlgebra,
                 raw_topology_tuple: tuple, state_space_ref: Any,
                 raw_initial_field: Optional[jnp.ndarray] = None) -> 'DiscreteStaticFieldGeneratorSystem':
        instance = cls(generator, field_algebra, is_raw_mode=True)
        instance._raw_topology_tuple = raw_topology_tuple
        instance._state_space_ref = state_space_ref

        if raw_initial_field is not None:
            instance.set_field(raw_initial_field)
        else:
            instance.clear_field()
        return instance

    # ==========================================
    # EVOLUTION (Pure Math / No internal mutation)
    # ==========================================

    def generate_field(self, steps: int, **kwargs) -> DiscreteFieldMapper:
        # Executes and returns, but does NOT mutate internal state (use save_generated_field for that)
        return self.generator.generate_multi_step(self.get_field(), steps, **kwargs)

    def generate_raw_field(self, steps: int, raw_global_field: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        # If in OOP mode, we lazily build the tuple and APPLY THE KERNEL here.
        if not self.is_raw_mode and self._raw_topology_tuple is None:
            adj_matrix = self.topology_system.topology.adjacency_matrix

            # Auto-apply the kernel if the generator has one
            if self.generator.kernel is not None:
                weights = self.generator.kernel.compute_raw_batch(adj_matrix.indices).flatten()
            else:
                weights = adj_matrix.data

            self._raw_topology_tuple = (
                adj_matrix.indices[:, 0],
                adj_matrix.indices[:, 1],
                weights,
                self.get_raw_state_space().shape[0]
            )

        return self.generator.generate_raw_multi_step(
            raw_field=self.current_raw_data,
            raw_topology=self._raw_topology_tuple,
            steps=steps,
            raw_global_field=raw_global_field
        )

        # ==========================================
        # FEATURE IMPLEMENTATIONS
        # ==========================================

    def clear_field(self) -> None:
        """ FEATURE 5: Resets field to an impulse at ID 0 """
        if self.is_raw_mode:
            num_nodes = self._raw_topology_tuple[3]
            # Safest check: Look directly at the weights array dtype
            weights_array = self._raw_topology_tuple[2]
            is_complex = jnp.issubdtype(weights_array.dtype, jnp.complexfloating)
        else:
            num_nodes = self.state_system.get_raw_state_space().shape[0]
            # Fallback for OOP mode: Check the attached kernel
            is_complex = hasattr(self.generator.kernel, 'compute_raw_batch') and 'complex' in str(
                type(self.generator.kernel)).lower()

        # Determine the correct dtype (Complex vs Real)
        dtype = jnp.complex64 if is_complex else jnp.float32

        # Create pure Zero array and set index 0 to 1.0 (or 1.0+0j)
        impulse_array = jnp.zeros((num_nodes, 1), dtype=dtype).at[0].set(1.0)
        self.set_field(impulse_array)



    # ==========================================
    # INTERNAL UTILS
    # ==========================================

    def _wrap_raw_to_mapper(self, raw_data: jnp.ndarray) -> DiscreteFieldMapper:
        return DiscreteFieldMapper(self.get_state_space(), self.algebra, explicit_buffer=raw_data)

    def _sync_raw_from_mapper(self) -> None:
        if self.current_mapper:
            self.current_raw_data = self.current_mapper.raw_buffer