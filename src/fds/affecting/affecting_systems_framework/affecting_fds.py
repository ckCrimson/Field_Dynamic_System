from abc import ABC
from typing import TypeVar, Generic, Dict, Optional, List

from fds import State, StatSpace, Field
from fds.affecting.affecting_systems_framework.affected_reachable import AffectedReachable
from fds.affecting.affecting_systems_framework.affected_reaching import AffectedReaching
from fds.affecting.affecting_systems_framework.affecting_state_space_mapping import AffectedSystemsMapping
from fds.affecting.affecting_systems_framework.correlated_state import CorrelatedState
from fds.affecting.affecting_systems_framework.grouped_operators import AffectedSystemsOperator
from fds.dynamic_systems.field_dynamic_system import FieldDynamicSystem
from fds.dynamics.multi_step.multi_step_field import MultiStepField

S = TypeVar("S", bound=State)
class AffectingFDS(FieldDynamicSystem[S], ABC, Generic[S]):
    """
    Interaction-group FDS (generic). Owns a derived group state-space and evolves it.

    Parameters
    ----------
    members : Dict[name, FieldStaticDynamicSystem]
        Interacting systems that this group controls.
    mapping : AffectedSystemsMapping
        Bijective mapping between {name->state} and the group state S.
    state_space : StatSpace[S]
        The group state space (already constructed).
    field : Field[S]
        The primary group field (will be mutated over time).
    reachable : AffectedReachable[S]
        Group reachable; should already be bound to `members`.
    transition_list : Optional[List[S]]
        History list; if None, initialized with [s0].
    initial_state : Optional[S]
        If None, computed from current member states via `mapping.get_affected_state(...)`.
    reaching : Optional[AffectedReaching]
        Optional helper; may be None.
    multi_step_field : MultiStepField[S]
        Multi-step field generator (path-integral).
    operator : AffectedSystemsOperator[S]
        Operator that picks the next group state and writes back to members.
    system_global_field : Optional[Field[S]]
        Global shaping field Q; may be None if unused.
    """

    def __init__(
        self,
        members: Dict[str, FieldDynamicSystem] = {},
        mapping: AffectedSystemsMapping = None,
        state_space: StatSpace = None,
        field: Field  = None,
        reachable: AffectedReachable[S] = None,
        transition_list: Optional[List[S]] = None,
        initial_state: Optional[S] = None,
        reaching: Optional[AffectedReaching] = None,
        multi_step_field: Optional[MultiStepField] = None,
        operator: Optional[AffectedSystemsOperator] = None,
        system_global_field: Optional[Field] = None,
        build_from_reachable:bool=False,
        within_state_space:bool=False,
        sync_members_from_s0: bool = False  # if True and s0 provided, push s0 → members
    ) -> None:
        # ---- store & sanity checks ----
        if state_space is None:
            raise ValueError("AffectingFDS: `state_space` must not be None.")
        if field is None:
            raise ValueError("AffectingFDS: `field` must not be None.")
        if reachable is None:
            raise ValueError("AffectingFDS: `reachable` must not be None.")
        if multi_step_field is None:
            raise ValueError("AffectingFDS: `multi_step_field` must not be None.")
        if operator is None:
            raise ValueError("AffectingFDS: `operator` must not be None.")

        self.members = members
        self.mapping = mapping

        # ---- derive s0 if needed ----
        s0: S
        if initial_state is None:
            # pack current member states into a correlated collection then map to S
            current_dict = {name: sys.initial_state for name, sys in members.items()}
            corr = CorrelatedState(current_dict)  # your CorrelatedState ctor
            s0 = mapping.get_affected_state(corr)
        else:
            s0 = initial_state
            if sync_members_from_s0:
                # optional: ensure members agree with provided s0
                cs = operator.pick_state_from_set( mapping.get_system_state(s0) )
                for name, sys in members.items():
                    st = cs[name]
                    try:
                        sys.initial_state = st
                    finally:
                        if hasattr(sys, "state_space") and hasattr(sys.state_space, "set_state"):
                            sys.state_space.set_state(st)

        # ---- build transition history ----
        tlist = transition_list or [s0]

        # ---- hand off to the base class ----
        super().__init__(
            initial_state=s0,
            state_space=state_space,
            field=field,
            multi_step_field=multi_step_field,
            reachable=reachable,
            operator=operator,
            transition_list=tlist,
            reaching=reaching,
            system_global_field=system_global_field,
            build_from_reachable=build_from_reachable,
            within_state_space=within_state_space
        )

        # optional: light invariant in debug builds
        try:
            self._assert_consistency()
        except Exception:
            # swallow in production if you prefer; keep for debugging
            pass

    # ---------------- utilities ----------------
    def _assert_consistency(self) -> None:
        """Best-effort consistency check (no-op if spaces don’t expose get_state)."""
        try:
            assert self.state_space.get_state() == self.initial_state
        except Exception:
            # ok: some StatSpace impls may not track current
            pass
        # ensure operator knows our members/mapping if it exposes those attributes
        try:
            if hasattr(self.operator, "affecting_group"):
                assert self.operator.affecting_group is self.members
            if hasattr(self.operator, "mapping"):
                assert self.operator.mapping is self.mapping
        except Exception:
            pass


