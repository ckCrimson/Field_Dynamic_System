from abc import ABC, abstractmethod
from typing import Tuple, Generic, TypeVar
from .data import FieldValue, RealFieldValue

T_In = TypeVar("T_In", bound=FieldValue)
T_Out = TypeVar("T_Out", bound=FieldValue)

# --- 1. Generalized (Root) ---
class GeneralizedFieldComposition(ABC, Generic[T_In, T_Out]):
    """
    Abstract Binary Operation: f1 x f2 -> f3
    Can map to any output type.
    """
    @abstractmethod
    def compose(self, fv_1: T_In, fv_2: T_In) -> T_Out: pass


# --- 2. Contained (Inherits Generalized) ---
class ContainedFieldComposition(GeneralizedFieldComposition[T_In, T_Out]):
    """
    Constraint: Inputs are same type. Output can be different.
    Used for: Metrics, Inner Products.
    """
    pass

class InnerFieldProduct(ContainedFieldComposition[T_In, RealFieldValue]):
    """
    Specific Contained Op: f x f -> RealFieldValue
    """
    pass


# --- 3. Closed (Inherits Contained) ---
class ClosedFieldComposition(ContainedFieldComposition[T_In, T_In]):
    """
    Constraint: Inputs and Output are SAME type.
    Must define Identity.
    """
    @abstractmethod
    def get_identity(self, shape: Tuple[int, ...]) -> T_In: pass

class AdditionComposition(ClosedFieldComposition[T_In]):
    """Defines Additive Group structure (Zero)."""
    pass

class MultiplicationComposition(ClosedFieldComposition[T_In]):
    """Defines Multiplicative Ring structure (Unity)."""
    pass