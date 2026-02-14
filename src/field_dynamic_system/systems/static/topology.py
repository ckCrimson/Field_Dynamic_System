from typing import Any, Optional, Type
import numpy as np

from src.field_dynamic_system.neighbor.interfaces import Topology
from src.field_dynamic_system.core.state.interfaces import StateSpace


# --- 1. THE GENERIC BASE CLASS ---

class StaticTopologySystem:
    """
    A System that explores reachable states from an initial configuration
    over a specified number of steps (l).

    Architecture Decision:
    This system DOES NOT manage matrix resizing or graph exploration caches.
    It holds a reference to a `Topology` object, treating it as a "Spatial Database"
    to query for adjacency matrices or raw neighbor expansions.
    """

    def __init__(self,
                 initial_state: Any,
                 topology: Topology,
                 state_space: Optional[StateSpace] = None):
        """
        Standard Constructor (Object Path).
        """
        self.topology = topology
        self.initial_state = initial_state
        self.state_class = type(initial_state) if initial_state else None
        self.state_space = state_space

        # Extract and cache the raw representation immediately for physics readiness
        self._raw_initial_state = self._process_initial_state(initial_state)

        # Baking State
        self._is_space_baked = False

    @classmethod
    def from_raw_data(cls,
                      raw_initial_state: Any,
                      topology: Topology,
                      state_space: Optional[StateSpace] = None,
                      state_class: Optional[Type] = None):
        """
        High-Performance Factory (Data Path).
        Bypasses object instantiation entirely.
        """
        raise NotImplementedError

    def _process_initial_state(self, initial_state: Any) -> Any:
        raise NotImplementedError

    def get_raw_state(self) -> Any:
        return self._raw_initial_state

    def multi_step_successor(self, steps: int, initial_state: Optional[Any] = None) -> StateSpace:
        raise NotImplementedError

    def get_raw_multistep(self, steps: int, initial_state_raw: Optional[Any] = None) -> Any:
        raise NotImplementedError

    # --- NEW: BAKING AND SPACE RETRIEVAL API ---

    def create_multi_step_states_space(self, steps: int) -> None:
        """Bakes the reachable universe into memory."""
        raise NotImplementedError

    def get_raw_state_space(self) -> Any:
        """Returns the raw ingredients (Arrays for Discrete, Bounds for Continuous)."""
        raise NotImplementedError

    def get_state_space(self) -> StateSpace:
        """Returns the fully instantiated OOP StateSpace."""
        raise NotImplementedError

    def get_lazy_state_space(self) -> StateSpace:
        """Returns a high-performance StateSpace powered by Lazy Proxies."""
        raise NotImplementedError


# --- 2. THE CONCRETE IMPLEMENTATION ---

class DiscreteStaticTopologySystem(StaticTopologySystem):
    """
    Concrete implementation for Discrete topologies (Grids, Graphs, Strings).
    """

    def build_space(self, initial_states: list, depth: int):
        """ The System handles the frontier expansion internally. """
        self.topology.expand_frontier(initial_states, depth)

    def get_raw_states(self) -> list:
        """ Exposes the discovered coordinates for the State System to sync. """
        return self.topology._id_to_raw

    @property
    def adjacency_matrix(self):
        # The system manages its own matrix access
        return self.topology.adjacency_matrix

    def _process_initial_state(self, initial_state: Any) -> Any:
        # Helper to aggressively unwrap known State object properties
        def extract_raw(obj):
            if hasattr(obj, 'values'): return obj.values
            if hasattr(obj, 'vector'): return obj.vector
            if hasattr(obj, 'state'): return obj.state
            if hasattr(obj, 'coordinates'): return obj.coordinates
            # If it's already a raw type (tuple, list, int, float, str), return it
            return obj

        # 1. Handle Sequence (List/Tuple) of Objects
        if isinstance(initial_state, (list, tuple)) and not isinstance(initial_state, str):
            if len(initial_state) > 0 and hasattr(initial_state[0], '__class__'):
                # Try extracting the first one to see if it's an object
                test_val = extract_raw(initial_state[0])
                if test_val is not initial_state[0]:
                    return np.array([extract_raw(s) for s in initial_state])
            # Otherwise, it's already raw data (e.g., tuple like (0,0))
            return np.array(initial_state)

        # 2. Handle Single Object or Raw Value
        return extract_raw(initial_state)

    @classmethod
    def from_raw_data(cls,
                      raw_initial_state: Any,
                      topology: Topology,
                      state_space: Optional[StateSpace] = None,
                      state_class: Optional[Type] = None):
        instance = cls.__new__(cls)
        instance.topology = topology
        instance._raw_initial_state = np.asarray(raw_initial_state)
        instance.initial_state = None
        instance.state_space = state_space
        instance.state_class = state_class
        instance._is_space_baked = False
        return instance

    def multi_step_successor(self, steps: int, initial_state: Optional[Any] = None) -> StateSpace:
        target_state = initial_state if initial_state is not None else self.initial_state
        if target_state is None:
            raise ValueError("No initial state provided for multi_step_successor.")
        return self.topology.multi_step_successor(target_state, steps)

    def get_raw_multistep(self, steps: int, initial_state_raw: Optional[Any] = None) -> Any:
        target_raw = initial_state_raw if initial_state_raw is not None else self._raw_initial_state
        if target_raw is None:
            raise ValueError("No raw initial state provided.")

        if isinstance(target_raw, np.ndarray):
            if target_raw.ndim == 1:
                batch = [target_raw.tolist()]
            else:
                batch = target_raw.tolist()
        elif isinstance(target_raw, (list, tuple)) and not isinstance(target_raw, str):
            if len(target_raw) > 0 and isinstance(target_raw[0], (int, float)):
                batch = [target_raw]
            else:
                batch = target_raw
        else:
            batch = [target_raw]

        return self.topology.get_raw_multi_step_successor(batch, steps)

    # --- NEW: IMPLEMENTING THE BAKING LOGIC ---

    def create_multi_step_states_space(self, steps: int):
        """
        Explores the universe up to 'steps' and locks the discovered states internally.
        """
        # 1. Run the raw BFS. (Updates Topology internal cache dynamically)
        self.get_raw_multistep(steps=steps)

        # 2. Synchronize memory! Pull exact array from cache to preserve IDs
        self._baked_raw_space_array = np.asarray(self.topology._id_to_raw)
        self._is_space_baked = True

        # print(f"Baked State Space with {len(self._baked_raw_space_array)} states.")

    def get_raw_state_space(self) -> np.ndarray:
        if not self._is_space_baked:
            raise RuntimeError("Must call create_multi_step_states_space() first.")
        # The index of this array acts as the implicit integer ID!
        return self._baked_raw_space_array

    def get_lazy_state_space(self) -> StateSpace:
        if not self._is_space_baked:
            raise RuntimeError("Must call create_multi_step_states_space() first.")
        if self.state_space is None or self.state_class is None:
            raise ValueError("Missing state_space or state_class reference to build Lazy space.")

        dim = self._baked_raw_space_array.shape[1] if self._baked_raw_space_array.ndim > 1 else 1

        # Uses the factory we built earlier to bypass OOP bottlenecks
        return self.state_space.__class__.from_raw_data(
            raw_data=self._baked_raw_space_array,
            wrapper=self.state_class,
            dim=dim
        )

    def get_state_space(self) -> StateSpace:
        if not getattr(self, '_is_space_baked', False):
            raise RuntimeError("Must call create_multi_step_states_space() first.")
        if self.state_class is None:
            raise ValueError("Missing state_class reference to build objects.")

        # Manually inflate objects (the slow way)
        objects = []
        for val in self._baked_raw_space_array:
            # Ensure proper conversion for tuples if necessary
            clean_val = tuple(val.tolist()) if hasattr(val, 'tolist') else val
            objects.append(self.state_class(clean_val))

        # --- NEW: Constructor Routing based on StateSpace Type ---
        space_class = self.state_space.__class__
        space_name = space_class.__name__

        if "VectorStateSpace" in space_name:
            # VectorStateSpace requires 'vectors' and 'dim'
            dim = self._baked_raw_space_array.shape[1] if self._baked_raw_space_array.ndim > 1 else 1
            return space_class(vectors=objects, dim=dim)
        else:
            # Generic StateSpaces (like String/Abstract) expect 'states'
            return space_class(states=objects)