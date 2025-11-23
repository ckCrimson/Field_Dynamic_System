from typing import Any

from fds.affecting.affecting_systems_framework.is_affecting import IsAffecting
from fds.dynamic_systems import FieldDynamicSystem


class IsSymmetric(IsAffecting):

    def __init__(self):
        super().__init__()

    def affects(self,system_src: FieldDynamicSystem,system_dst: FieldDynamicSystem,  **params: Any) -> bool:
        return True