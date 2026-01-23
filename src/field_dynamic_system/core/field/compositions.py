from abc import ABC, abstractmethod
import jax.numpy as jnp


# =========================================================
# 1. ROOT: Generic Field Composition (alpha)
# =========================================================
class FieldComposition(ABC):
    """
    Represents the operator 'alpha': f_v1 alpha f_v2 = f_v3
    Generic: Inputs and Outputs can be anything.
    """

    @abstractmethod
    def compose(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """
        The mathematical kernel.
        Must be compatible with JAX broadcasting/vectorization.
        """
        pass


# =========================================================
# 2. CONTAINED (beta): Same Input Types
# =========================================================
class ContainedFieldComposition(FieldComposition):
    """
    Represents the operator 'beta': f_v1 beta f_v2 = f_v3
    Constraint: Type(f_v1) == Type(f_v2).
    """
    pass


# =========================================================
# 3. CONTAINED PRODUCT: Same Input -> Real Output
# =========================================================
class ContainedProduct(ContainedFieldComposition):
    """
    Maps two similar fields to a Real (Scalar) Field.
    Output dim is always 1.
    """
    pass


class InnerProductComposition(ContainedProduct):
    """
    Standard Dot Product <u, v>.
    Linearity is assumed but not enforced at runtime for performance.
    """

    def compose(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        # Sum over the last axis (feature dimension)
        # keepdims=True ensures result is (N, 1) not (N,)
        return jnp.sum(a * b, axis=-1, keepdims=True)


# =========================================================
# 4. CLOSED: Same Input -> Same Output (Monoid)
# =========================================================
class ClosedFieldComposition(ContainedFieldComposition):
    """
    Input and Output are the same type.
    Must define an Identity element (e.g., 0 for +, 1 for *).
    """

    @abstractmethod
    def get_identity(self, shape: tuple, dtype=jnp.float64) -> jnp.ndarray:
        """Returns the neutral element for this composition."""
        pass


class AdditionComposition(ClosedFieldComposition):
    """ Element-wise Addition (+) """

    def compose(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return a + b

    def get_identity(self, shape: tuple, dtype=jnp.float64) -> jnp.ndarray:
        # Additive Identity: Zero
        return jnp.zeros(shape, dtype=dtype)


class MultiplicationComposition(ClosedFieldComposition):
    """ Element-wise Multiplication (x) """

    def compose(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return a * b

    def get_identity(self, shape: tuple, dtype=jnp.float64) -> jnp.ndarray:
        # Multiplicative Identity: One
        return jnp.ones(shape, dtype=dtype)