from abc import ABC
from typing import Dict

from fds.affecting.affecting_systems_framework.grouped_operators import AffectedSystemsOperator
from fds.affecting.correlation_systems_framework.system_correlated_state_space_mapping import \
    SystemCorrelatedSpaceMapping
from fds.dynamic_systems.field_dynamic_system import FieldDynamicSystem


class CorrelationAffectedOperator(AffectedSystemsOperator, ABC):
    def __init__(self,
                 affecting_group:  Dict[str, FieldDynamicSystem],
                 mapping: SystemCorrelatedSpaceMapping ,
                 *args,**kwargs):
        super().__init__(affecting_group,mapping,*args,**kwargs)