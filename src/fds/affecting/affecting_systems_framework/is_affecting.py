from abc import ABC, abstractmethod
from typing import Literal, Generic, TypeVar, Any, Dict, Optional

from fds import State
from fds.dynamic_systems.field_dynamic_system import FieldDynamicSystem

Connectivity = Literal["weak", "mutual", "strong"]

T = TypeVar("T", bound=State)

# ============================
# Policy interface: IsAffecting
# ============================
class IsAffecting(ABC, Generic[T]):
    """
    Pure policy that decides if one system's global/published field affects another.
    Directional relation: affects(src, dst) -> bool

    Subclasses should be *pure* and fast—avoid building heavy reachables here.
    """

    def __init__(self, **default_params: Any) -> None:
        self._defaults: Dict[str, Any] = dict(default_params)

    def _merge(self, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        out = dict(self._defaults)
        if params:
            out.update(params)
        return out

    @abstractmethod
    def affects(
        self,
        system_src: FieldDynamicSystem,
        system_dst: FieldDynamicSystem,
        **params: Any,
    ) -> bool:
        """Return True iff src's field affects dst's field (directional)."""
        raise NotImplementedError



