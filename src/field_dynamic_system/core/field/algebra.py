from abc import ABC, abstractmethod
import jax.numpy as jnp

# =========================================================
# 1. INTERFACE
# =========================================================
class IFieldAlgebra(ABC):
    dim: int
    dtype: jnp.dtype

    @abstractmethod
    def add(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray: pass

    @abstractmethod
    def mul(self, a: jnp.ndarray, scalar: jnp.ndarray) -> jnp.ndarray: pass

    @abstractmethod
    def inner_product(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray: pass

    @abstractmethod
    def norm(self, a: jnp.ndarray) -> jnp.ndarray: pass

    @abstractmethod
    def get_zero(self, shape=(1,)) -> jnp.ndarray: pass

    @abstractmethod
    def get_unity(self, shape=(1,)) -> jnp.ndarray: pass


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


# =========================================================
# 3. VECTOR FIELD ALGEBRA (Dim N)
# =========================================================
class VectorFieldAlgebra(IFieldAlgebra):
    def __init__(self, dim=3, dtype=jnp.float64):
        self.dim = dim
        self.dtype = dtype

    def add(self, a, b): return a + b

    def mul(self, a, scalar): return a * scalar

    def inner_product(self, a, b):
        return jnp.sum(a * b, axis=-1, keepdims=True)

    def norm(self, a):
        # Direct JAX implementation breaks the dependency on Transform class
        return jnp.linalg.norm(a, axis=-1, keepdims=True)

    def get_zero(self, shape=(1,)):
        final_shape = shape + (self.dim,)
        return jnp.zeros(final_shape, dtype=self.dtype)

    def get_unity(self, shape=(1,)):
        final_shape = shape + (self.dim,)
        return jnp.ones(final_shape, dtype=self.dtype)