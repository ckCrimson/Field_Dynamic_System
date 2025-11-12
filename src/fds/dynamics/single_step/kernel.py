
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from fds import State, FieldValue

S = TypeVar('S', bound=State)

# --- unchanged interfaces you showed (one tiny typo fix in __init__) ---

class Kernel(ABC, Generic[S]):
    def __init__(self, **param):
        self.param = param

    @abstractmethod
    def get_kernel_value(self, state: S, current_state: S) -> "FieldValue":
        pass

