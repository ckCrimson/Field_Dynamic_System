from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from .data import FieldValue, RealFieldValue

T_In = TypeVar("T_In", bound=FieldValue)
T_Out = TypeVar("T_Out", bound=FieldValue)

class FieldTransform(ABC, Generic[T_In, T_Out]):
    """
    Abstract Unary Operation: f1 -> f2
    """
    @abstractmethod
    def transform(self, fv_1: T_In) -> T_Out: pass

class NormFieldTransform(FieldTransform[T_In, RealFieldValue]):
    """
    Specific Transform: f -> RealFieldValue (Magnitude/Energy).
    """
    pass