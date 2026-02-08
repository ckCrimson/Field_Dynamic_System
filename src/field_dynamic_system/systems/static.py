from typing import Any, Dict, Optional, Tuple, Union
import jax.numpy as jnp
from jax.experimental import sparse

# Interfaces
from src.field_dynamic_system.core.state.interfaces import StateSpace
from src.field_dynamic_system.core.field.mappings import FieldMapper, DiscreteFieldMapper, ContinuousFieldMapper
from src.field_dynamic_system.neighbor.interfaces import ITopology
from src.field_dynamic_system.neighbor.discrete import DiscreteTopology


# =========================================================
# 1. THE MIXINS (Raw Data Handlers)
# =========================================================

class TopologyMixin:
    """
    Capability: Connectivity Data.
    Strategy:
    - DiscreteTopology: Extract 'Adjacency Matrix' (JAX BCOO).
    - Other: Hold object reference.
    """

    def __init__(self, topology: ITopology, *args, **kwargs):
        self._topology_ref = topology
        self._topology_data = None
        self._topology_type = "object_ref"

        # OPTIMIZATION: Extract Raw Matrix if available
        if isinstance(topology, DiscreteTopology):
            try:
                # Use the method defined in your DiscreteTopology class
                matrix = topology.get_adjacency_matrix()
                if isinstance(matrix, (sparse.BCOO, jnp.ndarray)):
                    self._topology_data = matrix
                    self._topology_type = "explicit_matrix"
            except Exception:
                # Fallback to object reference if matrix build fails/isn't ready
                pass

        # FIX: Do NOT call super().__init__ here.
        # The concrete class manages the MRO chain explicitly.

    @property
    def topology_data(self) -> Any:
        """Returns Raw Matrix (fast) or Topology Object (slow)."""
        return self._topology_data if self._topology_data is not None else self._topology_ref


class FieldMixin:
    """
    Capability: Field Data Registry.
    Strategy:
    - DiscreteFieldMapper: Extract .raw_buffer (JAX Array).
    - ContinuousFieldMapper: Extract .background_func (Callable).
    """

    def __init__(self, field_mappers: Dict[str, FieldMapper] = None,
                 raw_fields: Dict[str, Any] = None, *args, **kwargs):

        # STORAGE: Name -> Raw Data (Array or Callable)
        self._fields: Dict[str, Any] = {}

        # PATH A: Initialize from Mappers (Extraction)
        if field_mappers:
            for name, mapper in field_mappers.items():
                self._fields[name] = self._extract_raw_data(mapper)

        # PATH B: Initialize from Raw Data (Copy Constructor)
        if raw_fields:
            self._fields.update(raw_fields)

        # FIX: Do NOT call super().__init__ here.

    def _extract_raw_data(self, mapper: Any) -> Any:
        """Helper to strip the Object Wrapper."""
        # Case 1: Discrete (Vectorized Array)
        if isinstance(mapper, DiscreteFieldMapper):
            return mapper.raw_buffer

        # Case 2: Continuous (Function)
        if isinstance(mapper, ContinuousFieldMapper):
            return mapper.background_func

        # Case 3: Fallback (Unknown Mapper Type)
        if hasattr(mapper, 'raw_buffer'):
            return mapper.raw_buffer

        return mapper

    @property
    def field_names(self) -> Tuple[str, ...]:
        return tuple(self._fields.keys())

    def get_field(self, name: str) -> Any:
        """Returns the RAW data (Array or Function)."""
        if name not in self._fields:
            raise KeyError(f"Field '{name}' not found.")
        return self._fields[name]

    def with_field(self, name: str, field_mapper: FieldMapper) -> 'StaticFieldTopologySystem':
        """
        IMMUTABLE ADDITION.
        Returns a NEW System with the new field added.
        """
        # 1. Shallow Copy current raw fields
        new_raw_fields = self._fields.copy()

        # 2. Extract and add new field
        new_raw_fields[name] = self._extract_raw_data(field_mapper)

        # 3. Return NEW Instance
        # We assume 'self' is the concrete class (StaticFieldTopologySystem)
        return type(self)(
            state=self.state,
            topology=self._topology_ref,  # Preserve original topology obj
            space=self.space,
            raw_fields=new_raw_fields
        )


# =========================================================
# 2. THE BASE (Identity)
# =========================================================

class StaticSystem:
    """
    The Core Identity.
    Holds the 'state' and 'space' (validation context).
    """

    def __init__(self, state: Any, space: Optional[StateSpace] = None, *args, **kwargs):
        self._state = state
        self._space = space

    @property
    def state(self) -> Any: return self._state

    @property
    def space(self) -> Optional[StateSpace]: return self._space


# =========================================================
# 3. THE CONCRETE SYSTEM (The Product)
# =========================================================

class StaticFieldTopologySystem(TopologyMixin, FieldMixin, StaticSystem):
    """
    The Full Simulation Context.

    COMPONENTS:
    1. State: Raw JAX Array or Value.
    2. Topology: Raw Adjacency Matrix (if discrete).
    3. Fields: Dict of Raw Arrays (if discrete) or Functions.
    """

    def __init__(self,
                 state: Any,
                 topology: ITopology,
                 space: Optional[StateSpace] = None,
                 field_mappers: Dict[str, FieldMapper] = None,
                 raw_fields: Dict[str, Any] = None):
        # Explicit Composition
        # We initialize each component manually. This avoids MRO complexity
        # and ensures specific arguments go to specific parents.

        TopologyMixin.__init__(self, topology)
        FieldMixin.__init__(self, field_mappers, raw_fields)
        StaticSystem.__init__(self, state, space)

    def __repr__(self):
        # Intelligent Repr for debugging
        s_info = getattr(self.state, 'shape', 'Scalar')
        t_info = self._topology_type
        f_list = list(self._fields.keys())
        return f"<System State={s_info} | Topology={t_info} | Fields={f_list}>"