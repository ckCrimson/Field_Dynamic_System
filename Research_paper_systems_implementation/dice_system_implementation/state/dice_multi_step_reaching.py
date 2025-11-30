from typing import Optional, Tuple, Callable

import numpy as np

from Research_paper_systems_implementation.dice_system_implementation.state.dice_reachable import DiceReachable
from Research_paper_systems_implementation.dice_system_implementation.state.dice_state import DiceState
from Research_paper_systems_implementation.dice_system_implementation.state.dice_state_space import DiceStateSpace
from fds.dynamics.multi_step import MultiStepReaching


class DiceMultiStepReaching(MultiStepReaching):
    def __init__(self,alpha,n_0,dice_number):
        self.alpha = alpha
        self.n_0 = n_0
        self.dice_number = dice_number
        reachable = DiceReachable(alpha,n_0,dice_number)
        super().__init__(reachable)

    def get_multi_step_reaching(
            self,
            initial_state: DiceState,
            l: int=None,
            reference_space=None,
            *,
            csr: Optional[Tuple[np.ndarray, np.ndarray]] = None,  # (indptr, indices)
            step_ids: Optional[Callable[[int], np.ndarray]] = None,  # top-level callable returning np.ndarray[int32]
            parallel: bool = False,
            use_processes: bool = True,
            workers: Optional[int] = None,
            chunk_min_size: int = 2048,
            parallel_threshold: int = 20_000,
            exactly_l_steps: bool = True,  # if False → ≤ l steps via visited mask
    )-> DiceStateSpace:
        total_throws= (self.alpha * l) +self.n_0

        lower =( (initial_state.state * self.n_0) + (self.alpha*l))/total_throws
        upper = ((initial_state.state * self.n_0) + (self.alpha*l*self.dice_number))/total_throws

        return DiceStateSpace(initial_state,self.dice_number,lower,upper)
