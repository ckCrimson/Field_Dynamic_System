"""
Geometric State Space Definitions (Continuous Module).
Implements Constructive Solid Geometry (CSG) for physical manifolds.
"""
import dataclasses
from abc import ABC, abstractmethod
from typing import Union, Sequence, Any, Optional, Type
from enum import Enum
from dataclasses import dataclass
import jax.numpy as jnp
from jax._src.tree_util import register_pytree_node

# Assuming these imports are correct based on your previous files
from .interfaces import StateSpace, StateEncoder, IContinuousStateSpace
from .encoding import IdentityEncoder


# --- 1. Operation Types ---
class CSGOp(Enum):
    """Defines how two spaces are combined."""
    UNION = "UNION"         # Logic: OR
    INTERSECTION = "INTERSECTION" # Logic: AND


# --- 2. The Abstract Base Class ---
class ContinuousStateSpace(IContinuousStateSpace, ABC):
    """
    Represents a geometric manifold of valid states.
    Uses the Composite Pattern to allow recursive combinations (CSG).
    """

    # --- Implementing IContinuousStateSpace Contract ---
    # Concrete classes MUST implement 'dim' property

    @abstractmethod
    def contains(self, state: Union[Any, Sequence[Any], jnp.ndarray]) -> Union[bool, jnp.ndarray]:
        """Returns True if the vector is inside the valid geometry."""
        ...

    @abstractmethod
    def project(self, raw_data: jnp.ndarray) -> jnp.ndarray:
        """
        Projects invalid points back onto the manifold.
        Crucial for physics stability (Constraint Enforcement).
        """
        ...



    def _resolve_to_array(self, state: Union[Any, Sequence[Any], jnp.ndarray]) -> jnp.ndarray:
        if isinstance(state, jnp.ndarray):
            return state
        if hasattr(state, 'values'):
            return jnp.array(state.values, dtype=jnp.float32)
        if isinstance(state, list) and len(state) > 0 and hasattr(state[0], 'values'):
            return jnp.array([s.values for s in state], dtype=jnp.float32)
        return jnp.array(state, dtype=jnp.float32)

    # --- Unified API for Building the Tree ---
    def union(self, other: 'StateSpace') -> 'StateSpace':
        """Returns a space representing (Self OR Other)."""
        if not isinstance(other, IContinuousStateSpace):
            raise TypeError("Cannot union Continuous with Non-Continuous space.")
        # Dimension Check
        if self.dim != other.dim:
             raise ValueError(f"Dimension mismatch in Union: {self.dim} vs {other.dim}")

        return CompositeSpace(self, other, CSGOp.UNION)

    def intersection(self, other: 'StateSpace') -> 'StateSpace':
        """Returns a space representing (Self AND Other)."""
        if not isinstance(other, IContinuousStateSpace):
            raise TypeError("Cannot intersect Continuous with Non-Continuous space.")
        # Dimension Check
        if self.dim != other.dim:
             raise ValueError(f"Dimension mismatch in Intersection: {self.dim} vs {other.dim}")

        return CompositeSpace(self, other, CSGOp.INTERSECTION)

    # Standard Protocol Implementation
    def validate(self, raw_data: jnp.ndarray) -> jnp.ndarray:
        return self.project(raw_data)

    @property
    def is_empty(self) -> bool:
        # Default implementation, subclasses can optimize
        return False

    def __init_subclass__(cls, **kwargs):
        """
        AUTOMAGIC: Every time a user creates a subclass, this runs.
        It automatically registers the new class as a JAX Pytree.
        """
        super().__init_subclass__(**kwargs)

        # Only register if it's a dataclass (standard way users will define spaces)
        if dataclasses.is_dataclass(cls):
            register_pytree_node(
                cls,
                cls._tree_flatten,
                cls._tree_unflatten
            )

    @classmethod
    def _tree_flatten(cls, instance):
        """
        Splits the object into Children (JAX arrays) and Aux (Static data).
        Convention: Fields starting with '_' are Static. Others are Dynamic.
        """
        children = []
        aux_data = []
        aux_keys = []

        # dynamic introspection of the dataclass fields
        for field in dataclasses.fields(instance):
            value = getattr(instance, field.name)
            if field.name.startswith('_'):
                # Private -> Static Metadata (e.g., _encoding, _dim if private)
                aux_data.append(value)
                aux_keys.append(field.name)
            else:
                # Public -> Dynamic JAX Array (e.g. low, high, center)
                children.append(value)

        return tuple(children), (tuple(aux_data), tuple(aux_keys))

    @classmethod
    def _tree_unflatten(cls, aux, children):
        aux_values, aux_keys = aux
        kwargs = {}

        # 1. Fill in dynamic children (public fields)
        child_iter = iter(children)
        for field in dataclasses.fields(cls):
            if not field.name.startswith('_'):
                kwargs[field.name] = next(child_iter)

        # 2. Fill in static aux data (private fields)
        for key, val in zip(aux_keys, aux_values):
            kwargs[key] = val

        return cls(**kwargs)


# --- 3. The Optimized Leaf Node (Hypercube) ---
@dataclass
class HypercubeSpace(ContinuousStateSpace):
    """
    Defined by simple Min/Max bounds.
    """
    low: Union[float, list, jnp.ndarray]
    high: Union[float, list, jnp.ndarray]
    _encoding: Optional[StateEncoder] = None

    def __post_init__(self):
        # 1. Force Conversion to JAX Arrays
        self.low = jnp.array(self.low, dtype=jnp.float32)
        self.high = jnp.array(self.high, dtype=jnp.float32)

        # 2. Handle Scalars (1D Case)
        # If user passes 0.0, JAX makes it shape (), we need shape (1,)
        if self.low.ndim == 0: self.low = self.low.reshape(1)
        if self.high.ndim == 0: self.high = self.high.reshape(1)

        # 3. Setup Encoder
        if self._encoding is None:
            self._encoding = IdentityEncoder(dim=self.dim)

    @property
    def dim(self) -> int:
        return self.low.shape[0]

    @property
    def encoding(self) -> StateEncoder:
        return self._encoding

    def contains(self, state: Union[Any, Sequence[Any], jnp.ndarray]) -> Union[bool, jnp.ndarray]:
        vecs = self._resolve_to_array(state)
        # axis=-1 ensures we check bounds per-vector
        in_bounds = jnp.all((vecs >= self.low) & (vecs <= self.high), axis=-1)
        return in_bounds

    def project(self, raw_data: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(raw_data, self.low, self.high)

    # --- Optimization Overrides ---
    def union(self, other: 'StateSpace') -> 'StateSpace':
        if isinstance(other, HypercubeSpace) and self.dim == other.dim:
            # Box + Box = Bounding Box
            return HypercubeSpace(
                jnp.minimum(self.low, other.low),
                jnp.maximum(self.high, other.high),
                self._encoding
            )
        return super().union(other)

    def intersection(self, other: 'StateSpace') -> 'StateSpace':
        if isinstance(other, HypercubeSpace) and self.dim == other.dim:
            # Box AND Box = Intersection Box
            return HypercubeSpace(
                jnp.maximum(self.low, other.low),
                jnp.minimum(self.high, other.high),
                self._encoding
            )
        return super().intersection(other)


# --- 4. The Recursive Glue (Composite Node) ---
class CompositeSpace(ContinuousStateSpace):
    """
    A Branch Node in the CSG Tree.
    """

    def __init__(self, a: ContinuousStateSpace, b: ContinuousStateSpace, op: CSGOp):
        if a.dim != b.dim:
            raise ValueError(f"Cannot combine spaces of different dimensions: {a.dim} vs {b.dim}")

        self.a = a
        self.b = b
        self.op = op
        self._dim = a.dim # Store dimension
        self._encoding = a.encoding

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def encoding(self) -> StateEncoder:
        return self._encoding

    def contains(self, state: Union[Any, Sequence[Any], jnp.ndarray]) -> Union[bool, jnp.ndarray]:
        in_a = self.a.contains(state)
        in_b = self.b.contains(state)

        if self.op == CSGOp.UNION:
            return jnp.logical_or(in_a, in_b)
        elif self.op == CSGOp.INTERSECTION:
            return jnp.logical_and(in_a, in_b)
        return False

    def project(self, vec: jnp.ndarray) -> jnp.ndarray:
        # Recursive Projection
        if self.op == CSGOp.UNION:
            p_a = self.a.project(vec)
            p_b = self.b.project(vec)
            d_a = jnp.linalg.norm(vec - p_a)
            d_b = jnp.linalg.norm(vec - p_b)
            return jnp.where(d_a < d_b, p_a, p_b)

        elif self.op == CSGOp.INTERSECTION:
            x = vec
            for _ in range(3):
                x = self.a.project(x)
                x = self.b.project(x)
            return x
        return vec

# Register CompositeSpace manually since it's not a dataclass
register_pytree_node(
    CompositeSpace,
    lambda obj: ((obj.a, obj.b), (obj.op,)), # Flatten: Children=(a,b), Aux=(op,)
    lambda aux, children: CompositeSpace(children[0], children[1], aux[0]) # Unflatten
)


# --- 5. The Hypersphere ---
@dataclass
class HypersphereSpace(ContinuousStateSpace):
    """
    A Circle/Sphere defined by center and radius.
    """
    center: Union[float, list, jnp.ndarray]
    radius: float
    _encoding: Optional[StateEncoder] = None

    def __post_init__(self):
        # 1. Force Conversion to JAX Arrays
        self.center = jnp.array(self.center, dtype=jnp.float32)

        # 2. Handle Scalar (1D)
        # If user passes 0.0, JAX makes it shape (), we need shape (1,)
        if self.center.ndim == 0:
            self.center = self.center.reshape(1)

        # 3. Setup Encoder
        if self._encoding is None:
            self._encoding = IdentityEncoder(dim=self.dim)

    @property
    def dim(self) -> int:
        return self.center.shape[0]

    @property
    def encoding(self) -> StateEncoder:
        return self._encoding

    def contains(self, state: Union[Any, Sequence[Any], jnp.ndarray]) -> Union[bool, jnp.ndarray]:
        vecs = self._resolve_to_array(state)
        # axis=-1 calculates norm for each vector independently
        if vecs.ndim == 1:
             dist = jnp.linalg.norm(vecs - self.center)
        else:
             dist = jnp.linalg.norm(vecs - self.center[None, :], axis=-1)

        return dist <= self.radius

    def project(self, raw_data: jnp.ndarray) -> jnp.ndarray:
        diff = raw_data - self.center
        dist = jnp.linalg.norm(diff)
        safe_dist = jnp.where(dist == 0, 1.0, dist)
        scale = jnp.where(dist > self.radius, self.radius / safe_dist, 1.0)
        return self.center + diff * scale