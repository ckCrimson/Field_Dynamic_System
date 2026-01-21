from abc import ABC, abstractmethod
from typing import Type
import jax.numpy as jnp
from src.field_dynamic_system.core.field.algebra import IFieldAlgebra, RealFieldAlgebra, VectorFieldAlgebra


class FieldTransform(ABC):
    """
    Base class for any operation that maps one Field Value to another.
    gamma(f_v1) = f_v2
    """
    def __init__(self, output_algebra_type: Type[IFieldAlgebra]):
        self.output_algebra_type = output_algebra_type

    @abstractmethod
    def __call__(self, raw_data: jnp.ndarray) -> jnp.ndarray:
        """
        The mathematical kernel.
        Must operate on a single field value (or be JAX-vectorizable).
        """
        pass

class LinearTransform(FieldTransform):
    """
    Represents T(x) = M*x.
    Optimized for Matrix Multiplication.
    """
    def __init__(self, matrix: jnp.ndarray, output_algebra_type: Type[IFieldAlgebra]):
        super().__init__(output_algebra_type)
        self.matrix = jnp.asarray(matrix)

    def __call__(self, raw_data: jnp.ndarray) -> jnp.ndarray:
        # matrix: (OutDim, InDim), raw_data: (InDim,)
        # result: (OutDim,)
        return jnp.dot(self.matrix, raw_data)

class NonLinearTransform(FieldTransform):
    """
    Represents T(x) = f(x).
    Example: ReLU, Sigmoid, etc.
    """
    def __init__(self, func, output_algebra_type: Type[IFieldAlgebra]):
        super().__init__(output_algebra_type)
        self.func = func

    def __call__(self, raw_data: jnp.ndarray) -> jnp.ndarray:
        return self.func(raw_data)

class NormTransform(FieldTransform):
    """
    Special Case: Maps any Vector/Tensor -> Real Scalar >= 0.
    """
    def __init__(self):
        # Norm always produces a Real scalar
        super().__init__(RealFieldAlgebra)

    def __call__(self, raw_data: jnp.ndarray) -> jnp.ndarray:
        # Returns shape (1,)
        return jnp.linalg.norm(raw_data, keepdims=True)

class VectorNormTransform(FieldTransform):
    """ T(v) = ||v|| """
    def __init__(self):
        super().__init__(VectorFieldAlgebra)

    def __call__(self, raw_data: jnp.ndarray) -> jnp.ndarray:
        # Input: (Dim,) -> Output: (1,)
        return jnp.linalg.norm(raw_data, keepdims=True)