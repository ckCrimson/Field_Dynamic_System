from abc import ABC, abstractmethod
from typing import Optional, Any

from src.field_dynamic_system.core.field.compositions import FieldComposition
from src.field_dynamic_system.core.field.mappings import IFieldMapper, DiscreteFieldMapper
from src.field_dynamic_system.core.field.transform import FieldTransform
from src.field_dynamic_system.generator.kernel import AbstractTransitionKernel
from src.field_dynamic_system.neighbor.discrete import DiscreteTopology
from src.field_dynamic_system.neighbor.interfaces import ITopology


class IFieldGenerator(ABC):
    """
    The Operator: F_new = G(F_old)
    """

    def __init__(self, topology: Optional[ITopology] = None):
        self.topology = topology

    @abstractmethod
    def generate_multi_step(self,
                            current_mapper: IFieldMapper,
                            steps: int,
                            global_mapper: Optional[IFieldMapper] = None) -> IFieldMapper:
        """
        Evolves the field for N steps.
        This is the ONLY evolution entry point.
        """
        pass


class IContinuousFieldGenerator(IFieldGenerator):
    """
    Abstract base for infinite state spaces.
    User must implement the integration logic.
    """
    pass


# src/field_dynamic_system/generator/interfaces.py


class IDiscreteFieldGenerator(ABC):
    """
    Interface for Discrete Field Generators.
    Responsible for evolving a field state over discrete time steps.
    """

    @abstractmethod
    def generate_multi_step(self,
                          field_mapper,
                          steps: int,
                          global_mapper: Optional['DiscreteFieldMapper'] = None):
        """
        Evolves the field for N steps.

        Args:
            field_mapper: The current state of the field (x_t).
            steps: Number of iterations to perform.
            global_mapper: (Optional) A static or dynamic external field (Global Context).
                           Acts as the 'Bias' or 'Forcing Function' in the system.

        Returns:
            A new DiscreteFieldMapper containing the evolved state (x_t+n).
        """
        pass