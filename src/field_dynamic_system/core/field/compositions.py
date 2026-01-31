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
    Robust to Zeros automatically (a + 0 = a).
    """

    def compose(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return a + b

    def get_identity(self) -> float:
        return 0.0

    # compose_reduction uses default (segment_sum)


class MultiplicationComposition(ClosedFieldComposition):
    """
    Algebra: Product (*)

    FIXED LOGIC: Asymmetric Identity.
    - 'a' (Signal): Must be present. If 0, result is 0.
    - 'b' (Context): If 0 (Empty Space), treated as 1.0 (Identity) to allow signal entry.
    """

    def compose(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        # 1. Define Null/Empty Value
        zeros = 0.0

        # 2. Context Safety ONLY:
        # We only swap 'b' (Target/Global) to Identity.
        # We DO NOT swap 'a' (Incoming Signal). If Signal is 0, Output must be 0.
        b_safe = jnp.where(b == zeros, 1.0, b)

        # 3. Compute Product
        # Case: Signal(0.5) * Empty(0.0->1.0) = 0.5 (Propagates)
        # Case: NoSignal(0.0) * Full(1.0)     = 0.0 (No Ghost Mass)
        return a * b_safe

    def get_identity(self) -> float:
        return 1.0

    def compose_reduction(self, values: jnp.ndarray, indices: jnp.ndarray, num_segments: int) -> jnp.ndarray:
        return jax.ops.segment_prod(values, indices, num_segments=num_segments)


class InnerProductComposition(ContainedFieldComposition):
    """
    Algebra: Dot Product.
    Note: Zeros here usually mean orthogonality or lack of projection,
    so standard 0.0 behavior is mathematically correct.
    """

    def compose(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(a * b, axis=-1, keepdims=True)

    def get_identity(self) -> float:
        return 0.0