import jax.numpy as jnp
from .data import RealFieldValue, extract_val
from .compositions import (
    AdditionComposition, MultiplicationComposition, InnerFieldProduct
)
from .transform import NormFieldTransform


# --- Real Field Logic ---

class RealAddition(AdditionComposition):
    def compose(self, a, b):
        return RealFieldValue(extract_val(a) + extract_val(b))

    def get_identity(self, shape):
        return RealFieldValue(jnp.zeros(shape))


class RealMultiplication(MultiplicationComposition):
    def compose(self, a, b):
        return RealFieldValue(extract_val(a) * extract_val(b))

    def get_identity(self, shape):
        return RealFieldValue(jnp.ones(shape))


class RealInnerProduct(InnerFieldProduct):
    def compose(self, a, b):
        # <a, b> = a * b
        return RealFieldValue(extract_val(a) * extract_val(b))


class RealNorm(NormFieldTransform):
    def transform(self, a):
        return RealFieldValue(jnp.abs(extract_val(a)))


# ... (Previous imports: jnp, RealFieldValue, etc.)
from .data import FieldValue, extract_val


# We might need a ComplexFieldValue if strict typing is desired,
# but for now reusing FieldValue logic works as it wraps any JAX array.

# --- Complex Field Strategies ---

class ComplexAddition(AdditionComposition):
    def compose(self, a, b):
        # Standard vector addition
        return FieldValue(extract_val(a) + extract_val(b))

    def get_identity(self, shape):
        return FieldValue(jnp.zeros(shape, dtype=jnp.complex64))


class ComplexMultiplication(MultiplicationComposition):
    def compose(self, a, b):
        # Standard complex multiplication
        return FieldValue(extract_val(a) * extract_val(b))

    def get_identity(self, shape):
        return FieldValue(jnp.ones(shape, dtype=jnp.complex64))


class ComplexInnerProduct(InnerFieldProduct):
    def compose(self, a, b):
        # <a, b> = a * conj(b) (Physics convention)
        # Note: Math convention is often conj(a) * b. JAX/Numpy follows physics usually.
        # We enforce: a * conj(b)
        val_a = extract_val(a)
        val_b = extract_val(b)
        return RealFieldValue(jnp.real(val_a * jnp.conj(val_b)))
        # Note: Inner product result is structurally complex, but for "Norm squared" context
        # it simplifies. Strictly speaking, <u,v> is complex.
        # However, if your InnerFieldProduct expects RealFieldValue output (as per hierarchy),
        # this might be an issue.
        # CORRECTON: A generic Inner Product in C returns a Complex Number.
        # But if our architecture forces RealFieldValue return, we are restricted.
        # Let's return a FieldValue (wrapping complex) for general <u,v>,
        # and RealFieldValue only for Norm.

        # Implementation fix: InnerFieldProduct was defined to return RealFieldValue?
        # If so, we can only support Euclidean spaces.
        # For Quantum mechanics, <u|v> is complex.
        # Let's assume for now we return the complex scalar wrapped in generic FieldValue.
        return FieldValue(val_a * jnp.conj(val_b))


class ComplexNorm(NormFieldTransform):
    def transform(self, a):
        # |z| = sqrt(z * conj(z)) -> Real Number
        return RealFieldValue(jnp.abs(extract_val(a)))