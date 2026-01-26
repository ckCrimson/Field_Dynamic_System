from abc import ABC, abstractmethod
from typing import Optional, Any

from src.field_dynamic_system.core.field.compositions import FieldComposition
from src.field_dynamic_system.core.field.mappings import IFieldMapper
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


class IDiscreteFieldGenerator(IFieldGenerator):
    """
    The 'Full Fledged' Engine.
    Implements the Chain Rule with Extrinsic Transform:
    Pipeline: F_t -> Z^j(Extrinsic) -> Z^i(Intrinsic) -> Chain(K, F_g) -> F_t+1
    """

    def __init__(self,
                 topology: DiscreteTopology,
                 kernel: AbstractTransitionKernel,
                 # The 4 Pillars + Extrinsic Transform:
                 intrinsic_transform: FieldTransform,  # Z^i (Pre-process Node)
                 extrinsic_transform: FieldTransform,  # Z^j (Pre-process Step/Time)
                 intrinsic_composer: FieldComposition,  # alpha^i (Global Mix: K + F_g)
                 chain_composer: FieldComposition,  # alpha_c (Flow: F_src + Jump)
                 global_field_mapper: Optional[IFieldMapper] = None):  # F_g

        super().__init__(topology)
        self.kernel = kernel

        # Transforms
        self.Z_i = intrinsic_transform
        self.Z_j = extrinsic_transform

        # Composers
        self.alpha_i = intrinsic_composer
        self.alpha_c = chain_composer

        # Global Context
        self.F_g = global_field_mapper

        # CACHE: The Field Matrix (Optimization)
        self._compiled_jump_matrix = None

    @abstractmethod
    def precompute_matrix(self):
        """
        Builds the Flux Matrix once (K + alpha^i + F_g).
        """
        pass