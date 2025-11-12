from abc import ABC
from typing import Dict, Optional, Any

from fds.affecting.affecting_systems_framework.affected_reaching import AffectedReaching
from fds.affecting.correlation_systems_framework.correlated_reachable import CorrelatedAffectedReachable
from fds.dynamic_systems.field_dynamic_system import FieldDynamicSystem


class CorrelatedAffectedReaching(AffectedReaching, ABC):
    def __init__(self,members: Dict[str, FieldDynamicSystem],
        params: Optional[Dict[str, Any]] = None,
        reachable: Optional[CorrelatedAffectedReachable] = None):
        super().__init__(members, params,reachable)