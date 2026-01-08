# src/field_dynamic_system/core/__init__.py

# Expose the sub-modules so you can do: from core import state
from . import field
from . import state

# Optional: Expose common classes directly for easier access
# from .field.grid import Field
from .state.state import VectorState, AbstractState
from .state.continous import HypercubeSpace
from .state.discrete import AbstractDiscreteStateSpace, VectorStateSpace