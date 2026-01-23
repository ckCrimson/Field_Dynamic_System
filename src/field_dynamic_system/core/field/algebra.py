# =========================================================
# 1. INTERFACE
# =========================================================
from abc import ABC
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


