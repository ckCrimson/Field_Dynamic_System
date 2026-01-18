from abc import ABC
from typing import Union
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

RawData = Union[float, int, jnp.ndarray]

@register_pytree_node_class
class FieldValue(ABC):
    """Abstract Base for Field Data."""
    def __init__(self, value: RawData):
        self.value = jnp.asarray(value)

    def tree_flatten(self): return ((self.value,), None)
    @classmethod
    def tree_unflatten(cls, aux, children): return cls(*children)

class RealFieldValue(FieldValue):
    """Scalar Real Field Value."""
    pass

def extract_val(x: Union[FieldValue, RawData]) -> jnp.ndarray:
    return x.value if isinstance(x, FieldValue) else jnp.asarray(x)