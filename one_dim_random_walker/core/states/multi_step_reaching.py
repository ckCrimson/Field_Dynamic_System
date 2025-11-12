from fds.dynamics.multi_step import MultiStepReaching
from one_dim_random_walker.core.states.reachable import OneDimensionReachable


class OneDimensionMultiStepReaching(MultiStepReaching):
    def __init__(self, distance: int = 1):
        reachable = OneDimensionReachable(distance)
        super().__init__(reachable)