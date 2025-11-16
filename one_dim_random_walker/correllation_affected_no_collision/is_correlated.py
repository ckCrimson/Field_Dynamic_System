from typing import Any

from fds.affecting.affecting_systems_framework.is_affecting import IsAffecting
from one_dim_random_walker.dynamic_systems.dynamic_system import OneDimWalkerFieldDynamicSystem


class IsRWCorrelated(IsAffecting):
    def __init__(self, distance: int = 1):
        self.distance = distance
        super().__init__()

    def affects(
            self,
            system_src: OneDimWalkerFieldDynamicSystem,
            system_dst: OneDimWalkerFieldDynamicSystem,
            **params: Any,
    ) -> bool:
        if abs(system_src.initial_state.state - system_dst.initial_state.state) <= self.distance:
            return True
        return False

