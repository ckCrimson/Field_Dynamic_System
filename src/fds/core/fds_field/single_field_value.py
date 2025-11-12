from dataclasses import dataclass
from typing import Any

import numpy as np

@dataclass(slots=True, frozen=True, eq=True)
class SingleFieldValue:
    """
    Immutable, hashable container for field data.
    Stores a contiguous NumPy view; never dtype=object in hot paths.
    """
    value: Any

    def __post_init__(self):
        v = np.asarray(self.value)
        # forbid object dtype in compute paths (you can relax if you must)
        if v.dtype == object:
            raise TypeError("SingleFieldValue.value must not be dtype=object")
        # ensure contiguous numeric buffer
        if not v.flags['C_CONTIGUOUS']:
            v = np.ascontiguousarray(v)
        object.__setattr__(self, "value", v)

    def get_data(self) -> np.ndarray:
        # return the same view (no copy)
        return self.value

    def __repr__(self) -> str:
        # short repr to avoid huge logs
        return f"SFV(shape={self.value.shape}, dtype={self.value.dtype})"
