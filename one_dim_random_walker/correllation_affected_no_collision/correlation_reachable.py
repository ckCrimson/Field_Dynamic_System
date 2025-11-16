from fds.affecting.affecting_systems_framework.affected_reachable import AffectedReachable
from fds.affecting.affecting_systems_framework.correlated_state import CorrelatedState
from fds.affecting.affecting_systems_framework.correlated_state_space import CorrelatedStateSpace


class RandomWalkerCorrelatedReachable(AffectedReachable):
    def __init__(self, dict_of_systems):
        super().__init__(dict_of_systems)

    def get_affected_reachable(self,correlated_state: CorrelatedState):
        dict_of_each_system_reachable = {}

        for sys, items in correlated_state.items():
            dict_of_each_system_reachable[sys] = self.members[sys].reachable.get_reachable(items)
        correlated_full_state = CorrelatedStateSpace(dict_of_each_system_reachable)
        ids =  correlated_full_state.ids_view()
        list_of_states = []
        for sids in ids:
            state = correlated_full_state.get_state_by_id(int(sids))
            values = state.get_all_components()
            unique_values = set(values.values())
            if len(values.values())==len(unique_values):
                list_of_states.append(state)
        if list_of_states is not None and len(list_of_states)>0:
            return correlated_full_state.build_from_states(list_of_states,list_of_states[0])
        raise Exception("No reachable states found")


