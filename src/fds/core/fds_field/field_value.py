from __future__ import annotations
from dataclasses import dataclass
from typing import Union, ClassVar
import numpy as np

from fds.core.fds_field.single_field_value import SingleFieldValue
from fds.core.fds_field.single_field_value_composition import AdditionComposition
from fds.core.fds_field.single_field_value_transform import NormTransform

# assume these exist in your project
# from .single_field_value import SingleFieldValue
# from .field_ops import NormTransform, AdditionComposition

ArrayLike = Union[np.ndarray, float, int]

@dataclass(slots=True)
class FieldValue:
    """
    Abstract wrapper combining raw fds_field data and a norm transform.

    NOTE: Method names and call semantics intentionally unchanged.
    """
    data: "SingleFieldValue"
    normTransform: "NormTransform"=None
    additionComposition: "AdditionComposition"=None


    DEFAULT_NORM: ClassVar["NormTransform"] =None  # set below via configure() or subclass
    DEFAULT_ADD: ClassVar["AdditionComposition"] = None

    # ---- configuration helpers ----
    @classmethod
    def configure(cls, norm: "NormTransform", add: "AdditionComposition") -> None:
        """Set class-level shared operators (per subclass)."""
        cls.DEFAULT_NORM = norm
        cls.DEFAULT_ADD = add

    # -------- internal helpers (do NOT change external API) --------
    @staticmethod
    def _as_sfv(x: Union["SingleFieldValue", ArrayLike]) -> "SingleFieldValue":
        """Accept either SingleFieldValue or raw array/scalar and return SingleFieldValue (no needless copy)."""
        if isinstance(x, SingleFieldValue):
            return x
        # Ensure numeric contiguous ndarray for speed
        arr = np.asarray(x)
        if arr.dtype == object:
            raise TypeError("FieldValue expects numeric arrays; got dtype=object.")
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        return SingleFieldValue(arr)

    # ----------------- existing API (optimized) -----------------

    def internal_transform(self) -> float:
        """
        Compute and return the norm of the field data via the injected NormTransform.

        Fast path: normTransform.apply returns a scalar or 0-D ndarray.
        """
        out = self.normTransform.apply(self.data)      # expect it to accept SingleFieldValue
        # normalize result to Python float without extra copies
        if isinstance(out, SingleFieldValue):
            out_arr = out.get_data()
        else:
            out_arr = np.asarray(getattr(out, "get_data", lambda: out)())
        # out_arr may be 0-D ndarray or scalar
        return float(out_arr if np.isscalar(out_arr) else out_arr.item())

    @classmethod
    def get_unitary_field(cls) -> "FieldValue":
        """
        Return the unit (identity) field value for this type.

        Keep the signature; by default we cannot infer shape/ops here,
        so raise with a clear message unless subclasses override or the app
        installs a factory on the class.
        """
        raise NotImplementedError(
            "FieldValue.get_unitary_field() needs a concrete shape/dtype and ops. "
            "Either override in a subclass or provide a class-level factory."
        )

    @classmethod
    def get_zero_field(cls) -> "FieldValue":
        """
        Return the zero field value for this type.

        Same rationale as get_unitary_field(); override where you know shape/ops.
        """
        raise NotImplementedError(
            "FieldValue.get_zero_field() needs a concrete shape/dtype and ops. "
            "Either override in a subclass or provide a class-level factory."
        )

    def internal_composition(self, fv: "SingleFieldValue"):
        """
        Compose 'fv' into *this* FieldValue's data using the injected AdditionComposition.

        This keeps the name & call site but is now robust to both composition styles:
        - in-place composers (mutate and return None)  -> we keep self.data
        - functional composers (return new data)       -> we update self.data
        """
        res = self.additionComposition.compose(self.data, fv)
        # If composer returns a new value, adopt it; if None, assume in-place.
        if res is not None:
            self.data = self._as_sfv(res)

    def get_norm(self) -> float:
        """Alias kept for compatibility."""
        return self.internal_transform()

    def addition(self, fv: "FieldValue") -> "FieldValue":
        """
        Return a NEW FieldValue = self (+) fv, using the injected AdditionComposition.
        Avoids extra copies by honoring whatever the composer returns.
        """
        res = self.additionComposition.compose(self.data, fv.data)
        new_data = self._as_sfv(res if res is not None else self.data.get_data())
        # If composer was in-place and returned None, we must materialize the result.
        # Using .get_data() here avoids double-wrapping if it already returned SFV.
        return FieldValue(new_data, self.normTransform, self.additionComposition)

    # NEW (kept name/semantics of equals): approximate equality with tolerance
    def equals(self, other: "FieldValue", atol: float = 1e-12) -> bool:
        if other is None:
            return False
        arr1 = np.asarray(self.data.get_data())
        arr2 = np.asarray(other.data.get_data())
        # short-circuit on shape mismatch (np.allclose would broadcast)
        if arr1.shape != arr2.shape:
            return False
        # exact int path is faster; otherwise use allclose with rtol=0 to be strict
        if np.issubdtype(arr1.dtype, np.integer) and np.issubdtype(arr2.dtype, np.integer):
            return np.array_equal(arr1, arr2)
        return np.allclose(arr1, arr2, rtol=0.0, atol=atol)


    def get_value(self):return self.data.value
