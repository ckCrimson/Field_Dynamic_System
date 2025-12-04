from matplotlib import pyplot as plt
from numpy import integer

from Research_paper_systems_implementation.dice_system_implementation.dynamics.multi_step.multi_step_field_generator import \
    DiceMultiStepFieldGenerator
from Research_paper_systems_implementation.dice_system_implementation.dynamics.operator.dice_operator import \
    DiceOperator
from Research_paper_systems_implementation.dice_system_implementation.field.dice_real_field import DiceRealField
from Research_paper_systems_implementation.dice_system_implementation.state.dice_reachable import DiceReachable
from Research_paper_systems_implementation.dice_system_implementation.state.dice_state import DiceState
from Research_paper_systems_implementation.dice_system_implementation.state.dice_state_space import DiceStateSpace
from fds.dynamic_systems import FieldDynamicSystem

"""def __init__(self, initial_state : State , field: Field[S], multi_step_field: MultiStepField[S], reachable: Reachable[S],
                 operator: Operator[S], transition_list: list[S]=None,state_space: StatSpace[S] = None,
                 reaching: Reaching[S]=None,
                 system_global_field: Optional[Field[S]] = None, own_produced_field: Optional[Field[S]] = None,
                 build_from_reachable = False,within_state_space = False, multi_step_reaching: MultiStepReaching=None) -> None:"""

class DiceFieldDynamicSystem(FieldDynamicSystem):
        def __init__(self, initial_state: DiceState,
                     field: DiceRealField=None,
                     multi_step_field: DiceMultiStepFieldGenerator=None,
                     reachable: DiceReachable=None,
                     operator: DiceOperator=None,
                     transition_list: list = [],
                     state_space: DiceStateSpace = None,
                     reaching: DiceReachable = None,
                     alpha: int = 25,
                     N_0: int = 100,
                     dice_number: int = 6
                     ):
            self.alpha = alpha
            self.N_0 = N_0
            self.dice_number = dice_number
            if multi_step_field is None:
                multi_step_field = DiceMultiStepFieldGenerator(alpha=alpha,n_0=N_0,dice_number=dice_number)
            if reachable is None:
                reachable = DiceReachable(alpha=alpha,n_0=N_0,d=dice_number)
            if operator is None:
                operator=DiceOperator()
            if transition_list is None:
                transition_list=[]
            if state_space is None:
                state_space = DiceStateSpace(initial_state,self.dice_number)
            if field is None:
                field=DiceRealField(state_space=state_space)
            super().__init__(initial_state,field,multi_step_field,reachable,operator,transition_list,state_space, reaching)

        def multi_step_field_generator(self, steps:integer, curr_state: DiceState =None,**params) -> DiceRealField:
            if curr_state is None:
                curr_state = self.initial_state
            return  self.multi_step_field.generate_multi_step_field(self.state_space,curr_state,steps,**params)

        def plot_evolution(self):
            if self.transition_list is not None:
                if len(self.transition_list) > 0:
                    transition_list = [ x.state for x in  self.transition_list]
                    plt.axhline(y=(self.dice_number+1)/2, color='k', linestyle='--')
                    plt.plot(transition_list)
                    plt.show()

        def evolve_multiple_times(self, number_of_iteration:int,steps: int):
            for _ in range(number_of_iteration):
                self.evolve(steps)

        def __repr__(self):
            return  f"D={self.dice_number}, N_0 ={self.N_0},\u0391={self.alpha}"