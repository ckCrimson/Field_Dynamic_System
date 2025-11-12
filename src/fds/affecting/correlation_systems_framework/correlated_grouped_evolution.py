from abc import abstractmethod, ABC
from typing import Dict

from fds import Field
from fds.affecting.affecting_systems_framework.grouped_evolution import AffectingGroupsEvolution
from fds.affecting.correlation_systems_framework.correlated_affecting import CorrelationAffectingSystem
from fds.affecting.correlation_systems_framework.correlated_operator import CorrelationAffectedOperator
from fds.affecting.correlation_systems_framework.correlated_reachable import CorrelatedAffectedReachable
from fds.affecting.correlation_systems_framework.correlated_reaching import CorrelatedAffectedReaching
from fds.affecting.correlation_systems_framework.correlation_group import CorrelatinAffectingGroups
from fds.affecting.correlation_systems_framework.system_correlated_state_space_mapping import \
    SystemCorrelatedSpaceMapping
from fds.dynamic_systems.field_dynamic_system import FieldDynamicSystem
from fds.dynamics.multi_step.multi_step_field import MultiStepField


class CorrelationGroupEvolution(AffectingGroupsEvolution,ABC):
    def __init__(self,
            systems: Dict[str, FieldDynamicSystem],
            group_generator: CorrelatinAffectingGroups,
            correlated_reachable:CorrelatedAffectedReachable = None,
            field : Field = None,
            multi_step_field_generator: MultiStepField=None,
            operator: CorrelationAffectedOperator=None,global_field: Field=None,
            reaching: CorrelatedAffectedReaching=None,all_independent_systems: bool = False,):
        super().__init__(systems,group_generator,SystemCorrelatedSpaceMapping(),all_independent_systems =all_independent_systems)
        self.field=field
        self.multi_step_field_gen = multi_step_field_generator
        self.reachable = correlated_reachable
        self.operator = operator
        self.global_field=global_field
        self.reaching = reaching

    @abstractmethod
    def form_affected_system_from_dict(
        self,
        group: Dict[str, FieldDynamicSystem]
    ) -> CorrelationAffectingSystem:
        raise NotImplementedError


