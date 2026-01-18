from abc import ABC, abstractmethod
from typing import Any, Tuple
import jax.numpy as jnp

from .data import FieldValue
from .compositions import AdditionComposition, MultiplicationComposition, InnerFieldProduct
from .transform import NormFieldTransform
from .stratergies import (
    RealAddition, RealMultiplication, RealInnerProduct, RealNorm,
    ComplexAddition, ComplexMultiplication, ComplexInnerProduct, ComplexNorm
)


class IFieldAlgebra(ABC):
    """
    Interface Definition.
    Must define 'dtype' to handle memory allocation correctly.
    """

    @property
    @abstractmethod
    def dim(self) -> int: pass

    @property
    @abstractmethod
    def dtype(self) -> Any: pass  # <--- CRITICAL PROPERTY

    @abstractmethod
    def add(self, a, b): pass

    @abstractmethod
    def mul(self, a, b): pass

    @abstractmethod
    def inner_product(self, a, b): pass

    @abstractmethod
    def norm(self, a): pass

    @abstractmethod
    def get_zero(self, shape=(1,)): pass

    @abstractmethod
    def get_unity(self, shape=(1,)): pass


class FieldAlgebra(IFieldAlgebra):
    """
    Abstract Base Aggregator.
    """
    _add_strategy: AdditionComposition
    _mul_strategy: MultiplicationComposition
    _inner_strategy: InnerFieldProduct
    _norm_strategy: NormFieldTransform

    def add(self, a, b): return self._add_strategy.compose(a, b)

    def mul(self, a, b): return self._mul_strategy.compose(a, b)

    def inner_product(self, a, b): return self._inner_strategy.compose(a, b)

    def norm(self, a): return self._norm_strategy.transform(a)

    def get_zero(self, shape=(1,)): return self._add_strategy.get_identity(shape)

    def get_unity(self, shape=(1,)): return self._mul_strategy.get_identity(shape)


# --- Concrete Assemblers ---

class RealFieldAlgebra(FieldAlgebra):
    _add_strategy = RealAddition()
    _mul_strategy = RealMultiplication()
    _inner_strategy = RealInnerProduct()
    _norm_strategy = RealNorm()

    @property
    def dim(self) -> int: return 1

    @property
    def dtype(self): return jnp.float64  # <--- Real Fields use Float


class ComplexFieldAlgebra(FieldAlgebra):
    _add_strategy = ComplexAddition()
    _mul_strategy = ComplexMultiplication()
    _inner_strategy = ComplexInnerProduct()
    _norm_strategy = ComplexNorm()

    @property
    def dim(self) -> int: return 1

    @property
    def dtype(self): return jnp.complex128  # <--- Complex Fields use Complex