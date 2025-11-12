from typing import Dict

from fds.affecting.affecting_systems_framework.affecting_groups import InterAffectingGroups
from fds.affecting.correlation_systems_framework.is_correlated import IsCorrelationAffected
from fds.dynamic_systems.field_dynamic_system import FieldDynamicSystem


class CorrelatinAffectingGroups(InterAffectingGroups):
    def __init__(self, is_correlated: IsCorrelationAffected, systems: Dict[str,FieldDynamicSystem]):
        super().__init__(rule = is_correlated, systems = systems)