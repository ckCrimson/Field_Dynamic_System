from typing import Dict

from fds.affecting.affecting_systems_framework.grouped_evolution import AffectingGroupsEvolution
from fds.affecting.symmetric_systems_framework.symmetric_fds import SymmetricFDS
from fds.affecting.symmetric_systems_framework.symmetric_grouping import SymmetryGrouping
from fds.core.fds_state import StateSpaceMapping
from fds.dynamic_systems import FieldDynamicSystem


class SymmetricGroupEvolution(AffectingGroupsEvolution):
    def __init__(self, systems: Dict[str, FieldDynamicSystem], pivot_system_name: str,mapping_from_pivot: Dict[str,StateSpaceMapping],
                 group_generator: SymmetryGrouping ):
        self.pivot_system_name = pivot_system_name
        self.mapping_from_pivot = mapping_from_pivot
        super().__init__(systems, group_generator)

    def form_affected_system_from_dict(
        self,
        group: Dict[str, FieldDynamicSystem]
    ) -> SymmetricFDS:
        """
        Build a concrete AffectingFDS from a dict of member systems.
        Implement this in your subclass (wire mapping, reachable, operator, etc.).
        """
        return SymmetricFDS(group,self.pivot_system_name,self.mapping_from_pivot)



