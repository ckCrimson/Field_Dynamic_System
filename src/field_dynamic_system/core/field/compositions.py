from typing import Optional, Tuple, Any

import jax.numpy as jnp
import jax.ops
from abc import ABC, abstractmethod


# =========================================================
# 1. ROOT: Generic Field Composition (The Contract)
# =========================================================
class FieldComposition(ABC):
    """
    Defines the Algebra of the Field.

    1. compose: Binary Operation (A op B).
    2. compose_reduction: Aggregation Operation (Collapse A, B, C...).
    3. get_identity: The Neutral Element (Scalar preferred for broadcasting).
    """

    @abstractmethod
    def compose(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """
        Binary Op: Field A op Field B.
        Must handle "Null" (Zero) inputs intelligently based on algebraic rules.
        """
        pass

    def get_identity(self) -> float:
        """
        Returns the SCALAR identity element (Default: 0.0).
        Useful for initialization or external checks.
        """
        return 0.0

    def compose_reduction(self, values: jnp.ndarray, indices: jnp.ndarray, num_segments: int) -> jnp.ndarray:
        """
        Reduction Op: Collapses incoming edges -> Node State.
        Default: Summation (segment_sum).
        """
        return jax.ops.segment_sum(values, indices, num_segments=num_segments)


# =========================================================
# 2. CONTAINED (Taxonomy Layer)
# =========================================================
class ContainedFieldComposition(FieldComposition):
    """
    Marker Class: Inputs and Outputs share a compatible type structure.
    """
    pass


# =========================================================
# 3. CLOSED (Monoid Logic)
# =========================================================
class ClosedFieldComposition(ContainedFieldComposition):
    """
    Represents a Monoid: Input Type == Output Type.
    Must define an Identity element.
    """

    @abstractmethod
    def get_identity(self) -> float:
        """
        Must return the scalar identity (0.0 for Add, 1.0 for Mult).
        """
        pass


# =========================================================
# 4. CONCRETE IMPLEMENTATIONS
# =========================================================

class AdditionComposition(ClosedFieldComposition):
    """
    Algebra: Summation (+)
    Identity: Zero (Scalar 0.0 or Vector [0,0,...])
    """

    def compose(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return a + b

    def get_identity(self, shape: Optional[Tuple[int, ...]] = None, dtype=jnp.float32) -> Any:
        if shape is None:
            return 0.0
        return jnp.zeros(shape, dtype=dtype)

    # compose_reduction uses default (segment_sum)


class MultiplicationComposition(ClosedFieldComposition):
    """
    Algebra: Product (*)
    Identity: Unity (Scalar 1.0 or Vector [1,1,...])
    """

    def compose(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        # 1. Define Null/Empty Value (Scalar zero broadcasts to vector zero)
        zeros = 0.0

        # 2. Context Safety (Asymmetric Identity)
        # We treat "Empty Context" (b=0) as Identity (1.0) so signals can enter.
        # We DO NOT touch 'a' (Signal). If Signal is 0, it stays 0.
        b_safe = jnp.where(b == zeros, 1.0, b)

        return a * b_safe

    def get_identity(self, shape: Optional[Tuple[int, ...]] = None, dtype=jnp.float32) -> Any:
        if shape is None:
            return 1.0
        return jnp.ones(shape, dtype=dtype)

    def compose_reduction(self, values: jnp.ndarray, indices: jnp.ndarray, num_segments: int) -> jnp.ndarray:
        # Critical: Product reduction must initialize with 1.0
        return jax.ops.segment_prod(values, indices, num_segments=num_segments)


class InnerProductComposition(ContainedFieldComposition):
    """
    Algebra: Dot Product.
    Output is always a Scalar Field (N, 1), regardless of input vector size.
    """

    def compose(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(a * b, axis=-1, keepdims=True)

    def get_identity(self, shape: Optional[Tuple[int, ...]] = None, dtype=jnp.float32) -> Any:
        # Inner Product identity is technically zero (orthogonality)
        if shape is None:
            return 0.0
        return jnp.zeros(shape, dtype=dtype)