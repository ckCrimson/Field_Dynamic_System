import copy
from abc import ABC
from typing import Dict

from fds import Field, StatSpace
from fds.affecting.affecting_systems_framework.affecting_fds import AffectingFDS
from fds.affecting.affecting_systems_framework.correlated_state import CorrelatedState
from fds.affecting.affecting_systems_framework.correlated_state_space import CorrelatedStateSpace
from fds.affecting.correlation_systems_framework.correlated_operator import CorrelationAffectedOperator
from fds.affecting.correlation_systems_framework.correlated_reachable import CorrelatedAffectedReachable
from fds.affecting.correlation_systems_framework.correlated_reaching import CorrelatedAffectedReaching
from fds.affecting.correlation_systems_framework.system_correlated_state_space_mapping import \
    SystemCorrelatedSpaceMapping
from fds.dynamic_systems.field_dynamic_system import FieldDynamicSystem
from fds.dynamics.multi_step.multi_step_field import MultiStepField


class CorrelationAffectingSystem(AffectingFDS,ABC):
    def __init__(self,members: Dict[str, FieldDynamicSystem],
                 mapping: SystemCorrelatedSpaceMapping,
                 reachable: CorrelatedAffectedReachable,
                 multi_step_field: MultiStepField,
                 operator: CorrelationAffectedOperator,
                 system_global_field: Field = None,
                 field: Field = None,
                 state_space: CorrelatedStateSpace=None,
                 transitionList =[],
                 reaching : CorrelatedAffectedReaching = None,
                 build_correlated_state: bool = False,
                 build_from_reachable: bool =False
                 ):
        initial_state :CorrelatedState = None
        if state_space is None:
            if  build_correlated_state:
                if not build_from_reachable:
                    state_space_dict = [str,StatSpace]
                    for sys_id,sys in members:
                        state_space_dict[sys_id] = sys.get_state_space()
                    state_space = CorrelatedStateSpace(state_space_dict)
                else:
                    state_space_dict = [str,StatSpace]
                    for sys_id,sys in members:
                        state_space_dict[sys_id] = sys.reachable.get_reachable(sys.initial_state)
                    state_space = CorrelatedStateSpace(state_space_dict)
            else:
                #generating a minimal state space
                dict_of_individual_systems_space: Dict[str,StatSpace] ={}
                for sysid, system in members.items():
                    system_state = system.initial_state
                    set_of_system = {system_state}
                    draft_state_space  = copy.deepcopy(system.state_space)
                    draft_state_space.build_from_states(set_of_system)
                    dict_of_individual_systems_space[sysid] = draft_state_space
                state_space = CorrelatedStateSpace(dict_of_individual_systems_space)

        else:
            dictionary_of_state_space = [str,StatSpace]
            for sys_id,sys in members.items():
                copy_state_space = copy.deepcopy(sys.state_space)
                copy_state_space.copy_state_space.build_from_states({sys.initial_state})
                dictionary_of_state_space[sys_id] = copy_state_space
            state_space = CorrelatedStateSpace(dictionary_of_state_space)
        if system_global_field is None:
            system_global_field = copy.deepcopy(field)
            system_global_field.set_empty_field()
        super().__init__(members,mapping,state_space,field,reachable,transitionList,initial_state,reaching, multi_step_field,operator,system_global_field)

