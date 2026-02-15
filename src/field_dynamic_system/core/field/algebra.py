# =========================================================
# 1. INTERFACE
# =========================================================
from abc import ABC
from typing import Any, Tuple

import jax.numpy as jnp
from .compositions import (
    AdditionComposition,
    MultiplicationComposition,
    InnerProductComposition
)
from .transform import NormTransform


class IFieldAlgebra(ABC):
    dim: int
    dtype: jnp.dtype

    # Strategies
    _add_strategy: AdditionComposition
    _mul_strategy: MultiplicationComposition
    _inner_strategy: InnerProductComposition
    _norm_strategy: NormTransform  # From previous section

    def add(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return self._add_strategy.compose(a, b)

    def mul(self, a: jnp.ndarray, scalar: jnp.ndarray) -> jnp.ndarray:
        # Note: 'scalar' here implies scaling, which uses multiplication logic
        return self._mul_strategy.compose(a, scalar)

    def inner_product(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return self._inner_strategy.compose(a, b)

    def norm(self, a: jnp.ndarray) -> jnp.ndarray:
        return self._norm_strategy(a)

    def get_zero(self, shape=(1,)) -> jnp.ndarray:
        # Zero comes from the Additive Strategy
        final_shape = shape + (self.dim,)
        return self._add_strategy.get_identity(final_shape, self.dtype)

    def get_unity(self, shape=(1,)) -> jnp.ndarray:
        # Unity comes from the Multiplicative Strategy
        final_shape = shape + (self.dim,)
        return self._mul_strategy.get_identity(final_shape, self.dtype)


class VectorFieldAlgebra(IFieldAlgebra):
    def __init__(self, dim=3, dtype=jnp.float64):
        self.dim = dim
        self.dtype = dtype

        # Inject Strategies
        self._add_strategy = AdditionComposition()
        self._mul_strategy = MultiplicationComposition()
        self._inner_strategy = InnerProductComposition()
        self._norm_strategy = NormTransform()


# =========================================================
# 2. REAL FIELD ALGEBRA (Scalar)
# =========================================================
class RealFieldAlgebra(IFieldAlgebra):
    def __init__(self, dtype=jnp.float64):
        self.dim = 1
        self.dtype = dtype

    def add(self, a, b): return a + b

    def mul(self, a, scalar): return a * scalar

    def inner_product(self, a, b): return a * b

    def norm(self, a): return jnp.abs(a)

    def get_zero(self, shape=(1,)):
        final_shape = shape + (self.dim,)
        return jnp.zeros(final_shape, dtype=self.dtype)

    def get_unity(self, shape=(1,)):
        final_shape = shape + (self.dim,)
        return jnp.ones(final_shape, dtype=self.dtype)

class ComplexFieldAlgebra(IFieldAlgebra):
    """
    Algebra for fields mapping to the Complex Plane (C).
    Essential for Quantum Mechanics, Electromagnetism, and Wave Optics.
    """

    def __init__(self, use_double_precision: bool = False):
        # Defaulting to complex64 for GPU/TPU speed, complex128 for deep precision
        self._dtype = jnp.complex128 if use_double_precision else jnp.complex64

    @property
    def dtype(self) -> Any:
        return self._dtype

    def get_zero(self, shape: Tuple[int, ...] = ()) -> jnp.ndarray:
        """
        Returns 0.0 + 0.0j.
        Crucial for empty background generation in FieldSpaceComposer.
        """
        return jnp.zeros(shape, dtype=self._dtype)

    def get_one(self, shape: Tuple[int, ...] = ()) -> jnp.ndarray:
        """
        Returns 1.0 + 0.0j.
        Crucial for multiplication identities and mask setups.
        """
        return jnp.ones(shape, dtype=self._dtype)

    def cast(self, values: Any) -> jnp.ndarray:
        """
        Safely casts scalars, lists, or real arrays into the complex field.
        """
        # If it's already a JAX array of the exact type, return it immediately (Zero Overhead)
        if isinstance(values, jnp.ndarray) and values.dtype == self._dtype:
            return values

        # Otherwise, force the cast to the complex plane
        return jnp.asarray(values, dtype=self._dtype)

    # --- Algebra-Specific Utility Methods ---

    def get_imaginary_unit(self, shape: Tuple[int, ...] = ()) -> jnp.ndarray:
        """ Returns 0.0 + 1.0j (i). Highly useful for Schrödinger phase shifts. """
        return jnp.full(shape, 1j, dtype=self._dtype)

    def absolute_square(self, values: jnp.ndarray) -> jnp.ndarray:
        """ Returns |z|^2 (Probability Density). Converts complex back to real. """
        # We ensure it returns a purely real datatype (float32/64)
        real_dtype = jnp.float64 if self._dtype == jnp.complex128 else jnp.float32
        return (jnp.abs(values) ** 2).astype(real_dtype)



