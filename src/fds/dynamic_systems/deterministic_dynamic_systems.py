from abc import ABC
from numpy import integer


from fds import State
from fds.dynamic_systems import FieldDynamicSystem



class DeterministicDynamicSystem(ABC, FieldDynamicSystem):
    """A type of Dynamic Systems in which the operator is deterministic and can be written down in its closed form.

    """
    def __init__(self, initial_state: State):
        self.initial_state = initial_state
        super().__init__(
            initial_state=initial_state
        )

    def evolve(self,steps: integer, **params ):
        prev_state = self.initial_state
        for _ in range(steps):
            prev_state=self.find_next_state(prev_state)
        self.initial_state=prev_state

    def _find_next_state(self, prev_state: State) -> State:
        raise NotImplementedError("Deterministic Dynamic System is not implemented.")
