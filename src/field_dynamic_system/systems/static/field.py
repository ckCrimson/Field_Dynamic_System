from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from typing import Optional, Any, Union

from src.field_dynamic_system.core.field.mappings import IFieldMapper, DiscreteFieldMapper
from src.field_dynamic_system.core.field.algebra import IFieldAlgebra
from src.field_dynamic_system.generator.generator_interfaces import IFieldGenerator # Note: Fixed import path
from src.field_dynamic_system.systems.static.state import StaticStateSystem
from src.field_dynamic_system.systems.static.topology import StaticTopologySystem


class AbstractStaticFieldGeneratorSystem(ABC):
    def __init__(self, generator: IFieldGenerator, field_algebra: IFieldAlgebra, is_raw_mode: bool):
        self.generator = generator
        self.algebra = field_algebra
        self.is_raw_mode = is_raw_mode
        self.current_mapper: Optional[IFieldMapper] = None
        self.current_raw_data: Any = None

    def save_generated_field(self, steps: int, **kwargs) -> None:
        if self.is_raw_mode:
            self.current_raw_data = self.generate_raw_field(steps, **kwargs)
        else:
            self.current_mapper = self.generate_field(steps, **kwargs)
            self._sync_raw_from_mapper()

    def set_field(self, field_data: Union[IFieldMapper, jax.Array]) -> None:
        if isinstance(field_data, IFieldMapper):
            self.current_mapper = field_data
            self._sync_raw_from_mapper()
        else:
            self.current_raw_data = field_data
            if not self.is_raw_mode:
                self.current_mapper = self._wrap_raw_to_mapper(field_data)

    # --- THE ROUTING LOGIC ---
    def generate_field(self, steps: int, **kwargs) -> IFieldMapper:
        # 1. Unwrap the OOP global mapper into a raw array
        if 'global_mapper' in kwargs and kwargs['global_mapper'] is not None:
            mapper = kwargs.pop('global_mapper')
            kwargs['raw_global_field'] = mapper.raw_buffer

        # 2. Execute the fast raw pipeline
        raw_result = self.generate_raw_field(steps, **kwargs)

        # 3. Wrap and sync
        self.current_mapper = self._wrap_raw_to_mapper(raw_result)
        self.current_raw_data = raw_result
        return self.current_mapper

    def get_field(self) -> IFieldMapper:
        if self.is_raw_mode and self.current_mapper is None:
            self.current_mapper = self._wrap_raw_to_mapper(self.current_raw_data)
        return self.current_mapper

    def get_raw_fields(self) -> Any:
        return self.current_raw_data

    @abstractmethod
    def generate_raw_field(self, steps: int, **kwargs) -> Any:
        pass

    @abstractmethod
    def clear_field(self) -> None:
        pass

    @abstractmethod
    def get_state_space(self) -> Any:
        pass

    @abstractmethod
    def get_raw_state_space(self) -> Any:
        pass

    @abstractmethod
    def _wrap_raw_to_mapper(self, raw_data: Any) -> IFieldMapper:
        pass

    @abstractmethod
    def _sync_raw_from_mapper(self) -> None:
        pass
# ==========================================
# DISCRETE IMPLEMENTATION
# ==========================================

class DiscreteStaticFieldGeneratorSystem(AbstractStaticFieldGeneratorSystem):

    def __init__(self, generator: IFieldGenerator, field_algebra: IFieldAlgebra, is_raw_mode: bool):
        super().__init__(generator, field_algebra, is_raw_mode)
        self.state_system: Optional[StaticStateSystem] = None
        self.topology_system: Optional[StaticTopologySystem] = None
        self._raw_topology_tuple: Optional[tuple] = None
        self._state_space_ref: Any = None

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
            instance.clear_field()
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

    # --- EVOLUTION ---
    def generate_raw_field(self, steps: int, raw_global_field: Optional[jnp.ndarray] = None, **kwargs) -> jnp.ndarray:
        if not self.is_raw_mode and self._raw_topology_tuple is None:
            adj_matrix = self.topology_system.topology.adjacency_matrix
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

    # --- FEATURE IMPLEMENTATIONS (Fixed missing accessors) ---
    def clear_field(self) -> None:
        if self.is_raw_mode:
            num_nodes = self._raw_topology_tuple[3]
            weights_array = self._raw_topology_tuple[2]
            is_complex = jnp.issubdtype(weights_array.dtype, jnp.complexfloating)
        else:
            num_nodes = self.state_system.get_raw_state_space().shape[0]
            is_complex = hasattr(self.generator.kernel, 'compute_raw_batch') and 'complex' in str(type(self.generator.kernel)).lower()

        dtype = jnp.complex64 if is_complex else jnp.float32
        impulse_array = jnp.zeros((num_nodes, 1), dtype=dtype).at[0].set(1.0)
        self.set_field(impulse_array)

    def get_state_space(self) -> Any:
        return self.state_system.get_state_space() if not self.is_raw_mode else self._state_space_ref

    def get_raw_state_space(self) -> jax.Array:
        if self.is_raw_mode:
            return jnp.array(self._state_space_ref)
        return self.state_system.get_raw_state_space()

    # --- INTERNAL UTILS ---
    def _wrap_raw_to_mapper(self, raw_data: jnp.ndarray) -> DiscreteFieldMapper:
        return DiscreteFieldMapper(self.get_state_space(), self.algebra, explicit_buffer=raw_data)

    def _sync_raw_from_mapper(self) -> None:
        if self.current_mapper:
            self.current_raw_data = self.current_mapper.raw_buffer