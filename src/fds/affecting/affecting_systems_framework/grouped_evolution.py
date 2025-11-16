from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Dict, Optional, List, Any

from numpy import integer

from fds import State, StatSpace, Field
from fds.affecting.affecting_systems_framework.affected_reachable import AffectedReachable
from fds.affecting.affecting_systems_framework.affected_reaching import AffectedReaching
from fds.affecting.affecting_systems_framework.affecting_fds import AffectingFDS
from fds.affecting.affecting_systems_framework.affecting_groups import InterAffectingGroups
from fds.affecting.affecting_systems_framework.affecting_state_space_mapping import AffectedSystemsMapping
from fds.affecting.affecting_systems_framework.grouped_operators import AffectedSystemsOperator
from fds.dynamic_systems.field_dynamic_system import FieldDynamicSystem
from fds.dynamic_systems.field_static_dynamic_system import FieldStaticDynamicSystem
from fds.dynamics.multi_step.multi_step_field import MultiStepField

S = TypeVar("S", bound=State)

class AffectingGroupsEvolution(ABC, Generic[S]):
    """
    Orchestrates: systems -> disjoint affected groups -> AffectingFDS per group -> evolution.
    """

    def __init__(
        self,
        systems: Dict[str, FieldDynamicSystem],
        group_generator: InterAffectingGroups,
        mapping: AffectedSystemsMapping =None,
        state_space: StatSpace = None,
        field: Field  = None,
        reachable: AffectedReachable[S] = None,
        transition_list: Optional[List[S]] = None,
        reaching: Optional[AffectedReaching] = None,
        multi_step_field: Optional[MultiStepField] = None,
        operator: Optional[AffectedSystemsOperator] = None,
        system_global_field: Optional[Field] = None,
        sync_members_from_s0: bool = False,  # if True and s0 provided, push s0 → members
        all_independent_systems: bool = False,
    ) -> None:
        self.systems = systems
        self.group_generator = group_generator

        # {gid -> {name -> FDS}}
        self.affecting_groups: Dict[int, Dict[str, FieldDynamicSystem]] = {}

        # {gid -> AffectingFDS}
        self.dict_of_groups_of_affecting_fds: Dict[int, AffectingFDS] = {}

        self.field: Field = field
        self.mapping = mapping
        self.state_space = state_space
        self.field = field
        self.reachable = reachable
        self.transition_list = []
        self.reaching=reaching
        self.multi_step_field=multi_step_field
        self.operator = operator
        self.system_global_field = system_global_field

        self.affected_groups_formed: bool = False
        self.affected_systems_formed: bool = False
        self.multi_step_field_formed : bool = False
        self.all_independent_systems = all_independent_systems
        self.generate_affecting_groups()
        self.grouped_states_systems_formation()

    # -------- grouping --------
    def generate_affecting_groups(self) -> Dict[int, Dict[str, FieldDynamicSystem]]:
        """Compute disjoint affected groups once (idempotent)."""
        if self.all_independent_systems:
            count:int=0
            for sys_id,systems in self.systems.items():
                temp_dict: Dict[str, FieldDynamicSystem] = {}
                temp_dict[sys_id] =  systems
                self.affecting_groups[count]=temp_dict
            self.affected_groups_formed = True
            return self.affecting_groups
        if not self.affected_groups_formed:
            self.group_generator.refresh_with_new_systems(self.systems)
            groups = self.group_generator.get_affected_groups_dict()
            # Expect groups as {gid -> {name -> FDS}}
            if not isinstance(groups, dict):
                raise TypeError("group_generator.get_affected_groups_dict() must return Dict[int, Dict[str, FDS]].")
            self.affecting_groups = groups
            self.affected_groups_formed = True
        return self.affecting_groups

    @abstractmethod
    def form_affected_system_from_dict(
        self,
        group: Dict[str, FieldStaticDynamicSystem]
    ) -> AffectingFDS:
        """
        Build a concrete AffectingFDS from a dict of member systems.
        Implement this in your subclass (wire mapping, reachable, operator, etc.).
        """
        raise NotImplementedError

    # -------- build AffectingFDS per group --------
    def grouped_states_systems_formation(self) -> Dict[int, AffectingFDS]:
        """Instantiate one AffectingFDS per disjoint group (idempotent)."""
        #print(self.affected_groups_formed)
        #print(self.affected_systems_formed)
        if not self.affected_groups_formed:
            #print("Forming group")
            self.generate_affecting_groups()

        if not self.affected_systems_formed:
            #print("Forming system")
            out: Dict[int, AffectingFDS] = {}
            #print("group: ",self.affecting_groups)
            for gid, members_dict in self.affecting_groups.items():
                affected_fds = self.form_affected_system_from_dict(members_dict)
                out[gid] = affected_fds
                #print("Forming system with group id ",gid)
            #print("Output fds: ",out)
            self.dict_of_groups_of_affecting_fds = out
            self.affected_groups_formed = True
            self.affected_systems_formed = True
        return self.dict_of_groups_of_affecting_fds

    # -------- evolution --------
    def evolve(self, steps: integer = 1, **params: Any) -> None:
        """Evolve each AffectingFDS for `steps` steps."""
        if not self.multi_step_field_formed:
           self.multi_step_field_generator(steps)
        for gid, fds in self.dict_of_groups_of_affecting_fds.items():
            fds.evolve_from_field()
        self.affected_groups_formed = False
        self.affected_systems_formed= False
        self.multi_step_field_formed= False
        self.generate_affecting_groups()
        self.grouped_states_systems_formation()


    def multi_step_field_generator(self,steps: integer = 1, **params: Any) -> None:
        if self.affected_systems_formed:
            for group_systems in self.dict_of_groups_of_affecting_fds:
                systems = self.dict_of_groups_of_affecting_fds[group_systems]
                systems.save_multi_step_field(steps, **params)
        else:
            self.grouped_states_systems_formation()
            for group_systems in self.dict_of_groups_of_affecting_fds.items():
                systems = self.dict_of_groups_of_affecting_fds[group_systems]
                systems.save_multi_step_field(steps, **params)
                self.affected_systems_formed = True
        self.multi_step_field_formed = True

    # -------- maintenance --------
    def refresh_groups(self) -> None:
        """Drop caches so groups and per-group systems can be rebuilt."""
        self.affecting_groups.clear()
        self.dict_of_groups_of_affecting_fds.clear()
        self.affected_groups_formed = False
        self.affected_systems_formed = False

    def get_states_of_systems(self):
        dict_of_systems_states : Dict[str,State] ={}
        for sys_id, systems in self.systems.items():
            dict_of_systems_states[sys_id]=systems.initial_state
        return dict_of_systems_states
