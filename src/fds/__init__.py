# src/fds/__init__.py
from ._version import __version__

from .core.fds_state import State, StatSpace
from .core.fds_field import Field, FieldValue

__all__ = ["__version__", "State", "StatSpace", "Field", "FieldValue"]
