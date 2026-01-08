"""
Geometric State Space Definitions (Continuous Module).
Implements Constructive Solid Geometry (CSG) for physical manifolds.
"""
from abc import ABC, abstractmethod
from typing import Union, Sequence, Any
from enum import Enum
from dataclasses import dataclass
import jax.numpy as jnp

from .interfaces import StateSpace, StateEncoder, IDiscreteStateSpace, IContinuousStateSpace
from .state import State


# --- 1. Operation Types ---
class CSGOp(Enum):
    """Defines how two spaces are combined."""
    UNION = "UNION"  # Logic: OR
    INTERSECTION = "INTERSECTION"  # Logic: AND


# --- 2. The Abstract Base Class ---
class ContinuousStateSpace(IContinuousStateSpace, ABC):
    """
    Represents a geometric manifold of valid states.
    Uses the Composite Pattern to allow recursive combinations (CSG).
    """

    @abstractmethod
    def contains(self, state_vector: jnp.ndarray) -> bool:
        """Returns True if the vector is inside the valid geometry."""
        ...

    @abstractmethod
    def project(self, raw_data: jnp.ndarray) -> jnp.ndarray:
        """
        Projects invalid points back onto the manifold.
        Crucial for physics stability (Constraint Enforcement).
        """
        ...

    # --- Unified API for Building the Tree ---
    def union(self, other: 'StateSpace') -> 'StateSpace':
        """Returns a space representing (Self OR Other)."""
        if not isinstance(other, ContinuousStateSpace):
            raise TypeError("Cannot union Continuous with Non-Continuous space.")
        return CompositeSpace(self, other, CSGOp.UNION)

    def intersection(self, other: 'StateSpace') -> 'StateSpace':
        """Returns a space representing (Self AND Other)."""
        if not isinstance(other, ContinuousStateSpace):
            raise TypeError("Cannot intersect Continuous with Non-Continuous space.")
        return CompositeSpace(self, other, CSGOp.INTERSECTION)

    # Standard Protocol Implementation
    def validate(self, raw_data: jnp.ndarray) -> jnp.ndarray:
        return self.project(raw_data)


# --- 3. The Optimized Leaf Node (Hypercube) ---
@dataclass
class HypercubeSpace(ContinuousStateSpace):
    """
    Defined by simple Min/Max bounds.
    optimized for performance.
    """
    low: jnp.ndarray
    high: jnp.ndarray
    _encoding: StateEncoder

    @property
    def encoding(self) -> StateEncoder:
        return self._encoding

    def contains(self, state: Union[State, Sequence[State], jnp.ndarray]) -> Union[bool, jnp.ndarray]:
        # 1. Resolve Input
        vecs = self._resolve_to_array(state)

        # 2. Vectorized Logic
        # axis=-1 ensures we check bounds per-vector, regardless of batch size
        in_bounds = jnp.all((vecs >= self.low) & (vecs <= self.high), axis=-1)
        return in_bounds

    def project(self, raw_data: jnp.ndarray) -> jnp.ndarray:
        # Simple clamping is the fastest projection
        return jnp.clip(raw_data, self.low, self.high)

    # --- Optimization Overrides ---
    # We override these to prevent building a deep tree when
    # combining two simple boxes. We just return a new, better box.

    def union(self, other: 'StateSpace') -> 'StateSpace':
        if isinstance(other, HypercubeSpace):
            # Box + Box = Bounding Box (Convex Hull)
            return HypercubeSpace(
                jnp.minimum(self.low, other.low),
                jnp.maximum(self.high, other.high),
                self._encoding
            )
        return super().union(other)  # Fallback to Composite

    def intersection(self, other: 'StateSpace') -> 'StateSpace':
        if isinstance(other, HypercubeSpace):
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
    Combines two child spaces using a boolean operation.
    """

    def __init__(self, a: ContinuousStateSpace, b: ContinuousStateSpace, op: CSGOp):
        self.a = a
        self.b = b
        self.op = op
        # We assume spaces being combined share dimensionality/encoding
        self._encoding = a.encoding

    @property
    def encoding(self) -> StateEncoder:
        return self._encoding

    def contains(self, state: Union[State, Sequence[State], jnp.ndarray]) -> Union[bool, jnp.ndarray]:
        # Recursive check (Polymorphism handles the batching automatically)
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
            # Union Logic: "Project to whichever valid space is closer"
            p_a = self.a.project(vec)
            p_b = self.b.project(vec)

            d_a = jnp.linalg.norm(vec - p_a)
            d_b = jnp.linalg.norm(vec - p_b)

            return jnp.where(d_a < d_b, p_a, p_b)

        elif self.op == CSGOp.INTERSECTION:
            # Intersection Logic: Alternating Projections (POCS)
            # We bounce the point between A and B until it satisfies both.
            x = vec
            for _ in range(3):  # 3 Iterations is standard for stability
                x = self.a.project(x)
                x = self.b.project(x)
            return x

        return vec


# Append to src/field_dynamic_system/core/state/continuous.py

@dataclass
class HypersphereSpace(ContinuousStateSpace):
    """
    A Circle/Sphere defined by center and radius.
    """
    center: jnp.ndarray
    radius: float
    _encoding: StateEncoder

    @property
    def encoding(self) -> StateEncoder:
        return self._encoding

    def contains(self, state: Union[State, Sequence[State], jnp.ndarray]) -> Union[bool, jnp.ndarray]:
        # 1. Resolve Input
        vecs = self._resolve_to_array(state)

        # 2. Vectorized Logic
        # axis=-1 calculates norm for each vector in the batch independently
        dist = jnp.linalg.norm(vecs - self.center, axis=-1)
        return dist <= self.radius


    def project(self, raw_data: jnp.ndarray) -> jnp.ndarray:
        # If inside, keep it. If outside, scale it to the surface.
        diff = raw_data - self.center
        dist = jnp.linalg.norm(diff)

        # Avoid division by zero if point is exactly at center
        safe_dist = jnp.where(dist == 0, 1.0, dist)

        # Scale factor: if dist > radius, shrink it. Else 1.0
        scale = jnp.where(dist > self.radius, self.radius / safe_dist, 1.0)
        return self.center + diff * scale