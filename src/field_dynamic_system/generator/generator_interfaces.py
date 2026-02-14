from abc import ABC, abstractmethod
from typing import Optional, Any

from src.field_dynamic_system.core.field.mappings import IFieldMapper, DiscreteFieldMapper
from src.field_dynamic_system.neighbor.interfaces import ITopology


# =========================================================
# 1. THE BASE GENERATOR (The Operator)
# =========================================================
class IFieldGenerator(ABC):
    """
    The Operator: F_new = G(F_old)

    Dual-Path Architecture:
    - OOP Path: Uses `self.topology` and `generate_multi_step`.
    - Raw Path: Ignores `self.topology`. Uses `generate_raw_multi_step`
      where the raw topology data is injected statelessly at runtime.
    """

    def __init__(self, topology: Optional[ITopology] = None):
        """
        Standard Constructor.
        If using the Raw Path exclusively, initialize with topology=None.
        """
        self.topology = topology

    # --- PATH 1: Object-Oriented Evolution ---
    @abstractmethod
    def generate_multi_step(self,
                            current_mapper: IFieldMapper,
                            steps: int,
                            global_mapper: Optional[IFieldMapper] = None) -> IFieldMapper:
        """
        Evolves the field for N steps using heavy objects.
        Relies on `self.topology` being set during initialization.
        """
        pass

    # --- PATH 2: High-Performance Raw Evolution ---
    @abstractmethod
    def generate_raw_multi_step(self,
                                raw_field: Any,
                                raw_topology: Any,
                                steps: int,
                                raw_global_field: Optional[Any] = None) -> Any:
        """
        Stateless, high-performance evolution.
        Bypasses objects entirely. Applies the operator's math using
        pure JAX/NumPy arrays or compiled functions.

        Args:
            raw_field: The dense matrix (Discrete) or (func, cache) tuple (Continuous).
            raw_topology: The Adjacency Matrix (Discrete) or spatial bounds (Continuous).
            steps: Number of iterations.
            raw_global_field: Optional environmental/bias raw data.

        Returns:
            The evolved raw field data.
        """
        pass


# =========================================================
# 2. CONTINUOUS GENERATOR
# =========================================================
class IContinuousFieldGenerator(IFieldGenerator):
    """
    Abstract base for infinite state spaces (e.g., Euclidean Space).
    User must implement the integration logic (PDE solvers, integral transforms).
    """
    # Inherits both generate_multi_step and generate_raw_multi_step
    pass


# =========================================================
# 3. DISCRETE GENERATOR
# =========================================================
class IDiscreteFieldGenerator(IFieldGenerator):
    """
    Interface for Discrete Field Generators (Grids, Graphs, Networks).
    Responsible for evolving a field state over discrete time steps
    using Adjacency Matrices and discrete algebra.
    """

    @abstractmethod
    def generate_multi_step(self,
                            current_mapper: DiscreteFieldMapper,
                            steps: int,
                            global_mapper: Optional[DiscreteFieldMapper] = None) -> DiscreteFieldMapper:
        """
        OOP Path: Type-hinted specifically for DiscreteFieldMapper.
        """
        pass

    @abstractmethod
    def generate_raw_multi_step(self,
                                raw_field: Any,  # Expected: JAX Array (N, V)
                                raw_topology: Any,  # Expected: Sparse/Dense Adjacency Matrix (N, N)
                                steps: int,
                                raw_global_field: Optional[Any] = None) -> Any:
        """
        Raw Path: Evolves the field using matrix multiplication.
        Example Equation: F_new = (TopologyMatrix @ F_old) + Bias
        """
        pass