from __future__ import annotations
from typing import Any, Dict

def dump_field(field) -> Dict[str, Any]:
    """Return a plain-JSON-serializable dict for a Field (placeholder)."""
    return {"type": type(field).__name__}

def load_field(obj: Dict[str, Any]):
    """Reconstruct a Field from dump (placeholder)."""
    raise NotImplementedError

def dump_space(space) -> Dict[str, Any]:
    return {"type": type(space).__name__}

def load_space(obj: Dict[str, Any]):
    raise NotImplementedError
