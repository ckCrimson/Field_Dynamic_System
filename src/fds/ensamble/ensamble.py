from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Dict

from fds import State, Field
from fds.affecting.affecting_systems_framework.grouped_evolution import AffectingGroupsEvolution
from fds.dynamic_systems.field_dynamic_system import FieldDynamicSystem
from fds.ensamble.ensamble_operator import EnsembleOperator

S = TypeVar("S", bound=State)

class Ensemble(ABC, Generic[S]):
    def __init__(
        self,
        systems: Dict[str, "FieldDynamicSystem[S]"],
        interaction_channels: Dict[str, "AffectingGroupsEvolution[S]"],
    ) -> None:
        self.systems = systems
        self.interaction_channels = interaction_channels
        self.channel_group_systems: Dict[str, AffectingGroupsEvolution] = {}
        self.generate_channel_affected_group_systems()
        self.ensamble_operator = EnsembleOperator(interaction_channels)

        self.multistep_field_built: bool = False
        self._fields_per_system: Dict[str, "Field[S]"] = {}


    # --------- grouping per channel (keeps your flow) ---------
    def generate_channel_affected_group_systems(self) -> None:
        """Populate `channel_group_systems` from each channel’s grouping logic + build CFDS backbone."""
        # Build/refresh the correlated backbone first
        for channel_id, affected_group in self.interaction_channels.items():
            affected_group.systems = self.systems
            affected_group.affected_groups_formed=False
            affected_group.affected_systems_formed=False
            affected_group.multi_step_field_formed=False
            affected_group.grouped_states_systems_formation()
            self.channel_group_systems[channel_id]=affected_group



    # --------- field generation hooks ---------
    def _generate_single_step_field(self,channel_group:Dict[str, AffectingGroupsEvolution]) -> None:
        """Set `self.multistep_field_built = True` and fill `self._fields_per_system`."""
        ...

    def single_step_field(self,channel_group:Dict[str, AffectingGroupsEvolution]):
        if channel_group is None:
            channel_group = self.channel_group_systems
        self._generate_single_step_field(channel_group)


    @abstractmethod
    def _generate_multi_step_field(self, current_step:int, steps: int = 1,**params) -> None:
        """Set `self.multistep_field_built = True` and fill `self._fields_per_system`."""
        ...

    def generate_field(self, steps: int=1) -> None:
        if not self.multistep_field_built:
            #print("building channel field")
            for k in range(1,steps+1):
                self._generate_multi_step_field(steps,k)
            self.multistep_field_built=True

    # --------- evolution hook ---------
    @abstractmethod
    def _evolve_system(self, steps: int = 1) -> None:
        ...

    def evolve(self,steps:int  =1):
        if not self.multistep_field_built:
            self.generate_field(steps)
        self._evolve_system(steps)
        self.generate_channel_affected_group_systems()
        self.multistep_field_built = False
    # --------- small utilities ---------
    def invalidate_fields(self) -> None:
        self.multistep_field_built = False
        self._fields_per_system.clear()

    def get_field_for(self, name: str) -> "Field[S]":
        if not self.multistep_field_built:
            raise RuntimeError("Fields not built yet; call generate_and_save_field(...) first.")
        return self._fields_per_system[name]

    def set_field_for(self, name: str, field: "Field[S]") -> None:
        self._fields_per_system[name] = field
