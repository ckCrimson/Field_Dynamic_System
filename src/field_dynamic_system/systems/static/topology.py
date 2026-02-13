from typing import Any, Optional, Type
import numpy as np

from src.field_dynamic_system.neighbor.interfaces import Topology
from src.field_dynamic_system.core.state.interfaces import StateSpace


# --- 1. THE GENERIC BASE CLASS ---

class StaticTopologySystem:
    """
    A System that explores reachable states from an initial configuration
    over a specified number of steps (l).
ssssss
    Architecture Decision:
    This system DOES NOT manage matrix resizing or graph exploration caches.
    It holds a reference to a `Topology` object, treating it as a "Spatial Database"
    to query for adjacency matrices or raw neighbor expansions.
    """

    def __init__(self,
                 initial_state: Any,
                 topology: Topology):
        """
        Standard Constructor (Object Path).
        """
        self.topology = topology
        self.initial_state = initial_state
        self.state_class = type(initial_state) if initial_state else None

        # Extract and cache the raw representation immediately for physics readiness
        self._raw_initial_state = self._process_initial_state(initial_state)

    @classmethod
    def from_raw_data(cls,
                      raw_initial_state: Any,
                      topology: Topology,
                      state_class: Optional[Type] = None):
        """
        High-Performance Factory (Data Path).
        Bypasses object instantiation entirely.
        """
        raise NotImplementedError

    def _process_initial_state(self, initial_state: Any) -> Any:
        """Extracts raw data from State Objects."""
        raise NotImplementedError

    def get_raw_state(self) -> Any:
        """Returns the raw initial state configuration."""
        return self._raw_initial_state

    def multi_step_successor(self,
                             steps: int,
                             initial_state: Optional[Any] = None) -> StateSpace:
        """
        Object Path: Returns a StateSpace containing all reachable states.
        Uses JAX/Matrix multiplication under the hood via the Topology object.
        """
        raise NotImplementedError

    def get_raw_multistep(self,
                          steps: int,
                          initial_state_raw: Optional[Any] = None) -> Any:
        """
        Raw Path: Returns the raw data of all reachable states.
        Uses dynamic Breadth-First Search (BFS) directly on raw data, zero object overhead.
        """
        raise NotImplementedError


# --- 2. THE CONCRETE IMPLEMENTATION ---

class DiscreteStaticTopologySystem(StaticTopologySystem):
    """
    Concrete implementation for Discrete topologies (Grids, Graphs, Strings).
    """

    def _process_initial_state(self, initial_state: Any) -> Any:
        # 1. Handle Sequence of Objects -> Extract .value
        if isinstance(initial_state, (list, tuple)) and not isinstance(initial_state, str):
            return np.array([s.value for s in initial_state])

        # 2. Handle Single Object -> Extract .value
        if hasattr(initial_state, 'value'):
            return initial_state.value

        # Fallback (User accidentally passed raw data to __init__)
        return initial_state

    @classmethod
    def from_raw_data(cls,
                      raw_initial_state: Any,
                      topology: Topology,
                      state_class: Optional[Type] = None):
        """
        Fast Factory. Assigns the database (Topology) and the raw data directly.
        """
        instance = cls.__new__(cls)
        instance.topology = topology

        # Standardize input. copy=False ensures zero-copy if it's already an array.
        instance._raw_initial_state = np.array(raw_initial_state, copy=False)

        # Lazy inflation: No objects created until requested
        instance.initial_state = None
        instance.state_class = state_class

        return instance

    def multi_step_successor(self,
                             steps: int,
                             initial_state: Optional[Any] = None) -> StateSpace:
        """
        The "Safe & Math-Heavy" Path.
        Delegates to DiscreteTopology.multi_step_successor().
        This path relies on the Topology having a built adjacency_matrix.
        """
        target_state = initial_state if initial_state is not None else self.initial_state
        if target_state is None:
            raise ValueError("No initial state provided for multi_step_successor.")

        return self.topology.multi_step_successor(target_state, steps)

    def get_raw_multistep(self,
                          steps: int,
                          initial_state_raw: Optional[Any] = None) -> Any:
        """
        The "Fast & Dynamic" Path.
        Delegates to DiscreteTopology.get_raw_multi_step_successor().
        This path bypasses matrix math and dynamically builds the graph cache if needed.
        """
        target_raw = initial_state_raw if initial_state_raw is not None else self._raw_initial_state
        if target_raw is None:
            raise ValueError("No raw initial state provided.")

        # DEFENISVE BATCHING:
        # The Topology's raw BFS engine strictly expects a Batch (List/Sequence) of states.
        # We must format the input correctly so 'x in batch' checks work.

        if isinstance(target_raw, np.ndarray):
            # If it's a 1D array representing a single vector (e.g., [0,0,0]), wrap it.
            # Otherwise, assume it's an (N, D) batch matrix.
            if target_raw.ndim == 1:
                batch = [target_raw.tolist()]
            else:
                batch = target_raw.tolist()
        elif isinstance(target_raw, (list, tuple)) and not isinstance(target_raw, str):
            # If it's a single tuple like (0,0,0), wrap it so it's a batch of 1
            if len(target_raw) > 0 and isinstance(target_raw[0], (int, float)):
                batch = [target_raw]
            else:
                batch = target_raw
        else:
            # Single abstract state like "Group_A"
            batch = [target_raw]

        # Execute the high-performance raw BFS
        return self.topology.get_raw_multi_step_successor(batch, steps)