from typing import List

from fds.core.fds_state.state_space import DiscreteFiniteStatSpace
from one_dim_random_walker.core.states.state import IntegerState


class IntegerLine(DiscreteFiniteStatSpace):
    """
        current_state : Dontes the current position on the ineteger number line
        state_list

    """
    def __init__(self, current_state: IntegerState,states_list: List[IntegerState]=None, left_limit: int=None, right_limit:int=None) -> None:
        if states_list is not None:
            super().__init__(states_list,current_state,key=None,dim=1)
        if left_limit is not None and right_limit is not None and right_limit>left_limit and states_list is None:
            self._L = int(left_limit)
            self._R = int(right_limit)
            states_list = (IntegerState(i) for i in range(self._L, self._R + 1))
            super().__init__(states=states_list,
                             current=current_state if current_state is not None else None,
                             key=None,
                             dim=1)

    def __str__(self) -> str:
        return_string = ""
        counter = 0
        for s in self.get_all_states():
            return_string += str(s)
            return_string += ","
            if counter % 5 ==0 and counter!=0:
                return_string += "\n"
            counter += 1
        return return_string
