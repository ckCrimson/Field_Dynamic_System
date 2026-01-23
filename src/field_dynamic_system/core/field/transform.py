from abc import ABC, abstractmethod
from typing import Callable
import jax.numpy as jnp

class FieldTransform(ABC):
    """
    Base class for any operation that maps one Field Value to another.
    gamma(f_v1) = f_v2
    Operates strictly on Values (JAX Arrays).
    """
    @abstractmethod
    def __call__(self, raw_data: jnp.ndarray) -> jnp.ndarray:
        """ The mathematical kernel. """
        pass

class LinearTransform(FieldTransform):
    """ T(x) = M @ x """
    def __init__(self, matrix: jnp.ndarray):
        # No Algebra Type stored here anymore
        self.matrix = jnp.asarray(matrix)

    def __call__(self, raw_data: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(self.matrix, raw_data)

class NonLinearTransform(FieldTransform):
    """ T(x) = f(x) """
    def __init__(self, func: Callable):
        self.func = func

    def __call__(self, raw_data: jnp.ndarray) -> jnp.ndarray:
        return self.func(raw_data)

class NormTransform(FieldTransform):
    """ T(v) = ||v|| """
    def __call__(self, raw_data: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(raw_data, axis=-1, keepdims=True)