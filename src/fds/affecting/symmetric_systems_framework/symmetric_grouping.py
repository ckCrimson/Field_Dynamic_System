from typing import Dict, List, Set

from fds.affecting.affecting_systems_framework.affecting_groups import InterAffectingGroups
from fds.affecting.symmetric_systems_framework.is_symmetric import IsSymmetric
from fds.dynamic_systems import FieldDynamicSystem


class SymmetryGrouping(InterAffectingGroups):
    def __init__(self, systems: Dict[str, FieldDynamicSystem]):
        super().__init__(systems, IsSymmetric())


    def get_affected_groups_dict(self) -> Dict[int, Dict[str, FieldDynamicSystem]]:
        return_dict : Dict[int,Dict[str, FieldDynamicSystem]] ={}
        return_dict[0]=self._systems
        return return_dict