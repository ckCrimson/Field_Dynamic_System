from typing import Dict

from fds.affecting.affecting_systems_framework.grouped_evolution import AffectingGroupsEvolution


class EnsembleOperator:
    """
    Applies one-step evolution to each correlated system by delegating
    to the system's own operator.
    """
    def __init__(self, affected_groups: Dict[str, "AffectingGroupsEvolution[S]"]) -> None:
        self.affected_groups = affected_groups

    def evolve(self) -> None:
        """
        Advance all correlated systems by exactly `l` steps in one call.
        For one-iteration semantics, call with l=1.
        """
        for channel_id, channel in self.affected_groups.items():
            channel.evolve()