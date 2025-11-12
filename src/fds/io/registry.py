from __future__ import annotations
from typing import Callable, Dict, Any

class CodecRegistry:
    """Registry for pluggable (de)serializers keyed by type name."""
    def __init__(self):
        self._dumps: Dict[str, Callable[[Any], dict]] = {}
        self._loads: Dict[str, Callable[[dict], Any]] = {}

    def register(self, type_name: str, dump, load) -> None:
        self._dumps[type_name] = dump
        self._loads[type_name] = load

    def dump(self, obj: Any) -> dict:
        t = type(obj).__name__
        if t not in self._dumps:
            raise KeyError(f"No dumper for {t}")
        return self._dumps[t](obj)

    def load(self, obj: dict) -> Any:
        t = obj.get("type")
        if t not in self._loads:
            raise KeyError(f"No loader for {t}")
        return self._loads[t](obj)
