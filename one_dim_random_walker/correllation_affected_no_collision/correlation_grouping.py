from typing import Dict

from fds.affecting.affecting_systems_framework.affecting_groups import InterAffectingGroups
from one_dim_random_walker.correllation_affected_no_collision.is_correlated import IsRWCorrelated
from one_dim_random_walker.dynamic_systems.dynamic_system import OneDimWalkerFieldDynamicSystem


class CorrelationAffectingGroups(InterAffectingGroups):

    def __init__(self,systems: Dict[str, OneDimWalkerFieldDynamicSystem],
        rule: IsRWCorrelated):
        super().__init__(systems, rule)