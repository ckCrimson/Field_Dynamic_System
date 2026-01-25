from abc import ABC, abstractmethod
from typing import Callable
import jax.numpy as jnp
from jax._src.tree_util import register_pytree_node_class


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

@register_pytree_node_class
class FunctionalTransform(FieldTransform):
    """ Generic non-linear transform (e.g., Norm, Log, Activation) """
    def __init__(self, func: Callable[[jnp.ndarray], jnp.ndarray]):
        self.func = func

    def __call__(self, vector: jnp.ndarray) -> jnp.ndarray:
        return self.func(vector)

    def tree_flatten(self): return ((), (self.func,))
    @classmethod
    def tree_unflatten(cls, aux, children): return cls(aux[0])