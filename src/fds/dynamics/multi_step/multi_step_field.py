import copy
import random
from typing import TypeVar

from fds import State, Field, StatSpace, FieldValue
from fds.core.fds_field import ComposeField, TransformField
from fds.core.fds_state import Reachable
from fds.dynamics.single_step.single_step_field import SingleStepField

S = TypeVar('S', bound=State)

import copy as _copy
from typing import Iterable, Optional, Set, Generic



def rebuild_like_space(proto_space, states: Iterable[S], current: Optional[S]) -> "StatSpace[S]":
    """Prefer proto_space.build_like(); else generic DiscreteFiniteStatSpace; never deepcopy proto_space."""
    bl = getattr(proto_space, "build_like", None)
    if callable(bl):
        return bl(states, current=current)
    # Fallback: rebuild a generic finite space (adjust import to your project)
    from fds.core.fds_state.state_space import DiscreteFiniteStatSpace
    return DiscreteFiniteStatSpace(states=states, current=current)

def spawn_field_like(proto_field: "Field[S]", space_out) -> "Field[S]":
    """Avoid calling arbitrary __init__ shapes; never deepcopy live fields."""
    spawn = getattr(proto_field, "spawn", None)
    if callable(spawn):
        return spawn(space_out)
    from_space = getattr(type(proto_field), "from_space", None)
    if callable(from_space):
        return from_space(space_out)
    nf = _copy.copy(proto_field)          # shallow copy of config only
    nf.state_space = space_out
    if hasattr(nf, "set_empty_field"): nf.set_empty_field()
    if hasattr(nf, "set_zero_field"):  nf.set_zero_field()
    return nf

def broadcast_constant(proto_field: "Field[S]", space: "StatSpace[S]", value: "FieldValue") -> "Field[S]":
    """Create a field over `space` with the same `value` at every state (no deepcopy per state if immutable)."""
    out = spawn_field_like(proto_field, space)
    unit = out.get_unit_field()
    wrap = type(unit)
    # If FieldValue is immutable, reuse `value`; else clone once per write
    clone = getattr(value, "clone", None)
    fv = value if clone is None else clone()
    # ID fast path if present
    ids = getattr(space, "ids_view", None)
    sid2state = getattr(space, "get_state_by_id", None)
    if callable(ids) and callable(sid2state):
        arr = ids()
        for sid in arr:
            s = sid2state(int(sid))
            out.set_field(s, fv if clone is None else clone())
        return out
    # generic
    for s in space.get_all_states():
        out.set_field(s, fv if clone is None else clone())
    return out


class MultiStepField(Generic[S]):
    """
    Composite class for generating multi-step fields per the path-integral algorithm.
    """
    def __init__(
        self,
        reachable: Reachable[S],
        single_step: "SingleStepField[S]",
        intrinsic_composer: "ComposeField"=None,   # Z^P  (combine parent value with step field)
        extrinsic_composer: "ComposeField"=None,   # Z^t  (accumulate contributions from different parents)
        global_field_transform: "TransformField"=None,  # optional transform of global field each step
        single_step_transform: "TransformField"=None,   # transform for the single-step field
        multi_step_transform: "TransformField"=None     # transform for the multi-stepped field
    ):
        self.reachable = reachable
        self.single_step = single_step
        self.intrinsic_composer = intrinsic_composer
        self.extrinsic_composer = extrinsic_composer
        self.global_field_transform = global_field_transform
        self.single_step_transform = single_step_transform
        self.multi_step_transform = multi_step_transform

    # -------- helpers --------
    def _empty_like(self, proto_field: "Field[S]", space: "StatSpace[S]", *, zero: bool = True) -> "Field[S]":
        out = copy.deepcopy(proto_field)
        out.state_space = space
        out.set_empty_field()
        if zero:
            out.set_zero_field()
        return out
    def _broadcast_value_over_space(
        self,
        proto_field: "Field[S]",
        space: "StatSpace[S]",
        value: "FieldValue"
    ) -> "Field[S]":
        out = copy.deepcopy(proto_field)
        #out = proto_field.build_empty_from_state_space(type(proto_field),space,value)
        out.state_space = space
        out.set_empty_field()
        for s in space.get_all_states():
            out.set_field(s, copy.deepcopy(value))  # <<-- deep copy per state
        return out

    def build_next_reaching_state_space(
        self,
        initial_state: S,
        prev_reaching_space: "StatSpace[S]" = None,
        *,
        mode: str = "frontier",          # "frontier" (exactly l) or "within" (≤ l)
        positions_only: bool = False,    # collapse directions to one representative per (x,y)
    ) -> "StatSpace[S]":
        """
        Build the layer of states for the next step.

        - If prev_reaching_space is None:
            L^1 = Reachable(s0)
        - Else:
            L^l = ⋃_{s ∈ L^{l-1}} Reachable(s)

        mode="frontier" returns L^l (exactly l steps).
        mode="within"  returns ⋃_{k=0..l} L^k (cumulative "within l steps").

        NOTE: If your Reachable includes a stationary hop, L^l may intersect L^{l-1}.
              For “exactly l” semantics, keep stationary out of the frontier in Reachable.
        """
        # --- collect children (exactly-next layer) ---
        if prev_reaching_space is None:
            children = self.reachable.get_reachable(initial_state).get_all_states()
        else:
            children: set[S] = set()
            for s in prev_reaching_space.get_all_states():
                Rs = self.reachable.get_reachable(s)
                children.update(Rs.get_all_states())

        # --- choose the target set depending on the mode ---
        if mode == "within" and prev_reaching_space is not None:
            # union with all previously reachable (≤ l)
            target_states = set(prev_reaching_space.get_all_states())
            target_states.update(children)
        else:
            # exactly l (frontier)
            target_states = children

        # --- build a fresh space from those target states ---
        # Prefer to clone a compatible StatSpace; fall back to prev/Reachable's space.
        base_space = copy.deepcopy(prev_reaching_space) if prev_reaching_space is not None \
                     else copy.deepcopy(getattr(self.reachable, "space", None))
        if base_space is None:
            # as a last resort, start from a shallow copy of a space holding the initial state
            base_space = copy.deepcopy(self.reachable.get_reachable(initial_state))

        # If no targets, keep a single-state space at s0 (stable behavior)
        if not target_states:
            base_space.build_from_states({initial_state}, current=initial_state)
            return base_space
        cur = random.choice(list(target_states))
        # Pick a deterministic current drawn from the target se
        base_space.build_from_states(target_states, current=cur)
        return base_space


    # -------- main algorithm --------#
    def generate_multi_step_field_legacy(
        self,
        space: "StatSpace[S]",                  # ambient state space (for cloning/builds)
        start: S,                               # initial state s0
        L: int,                                 # number of steps
        single_transformed_field_proto: "Field[S]"=None,      # prototype for the accumulator/outputs
        single_step_transformed_field_proto: "Field[S]"=None, # prototype for per-parent single-step transform
        prev_field_input: "Field[S]"=None,                 # F^{0} (prob/weight over L^0={s0})
        global_field: "Field[S]"=None               # global/system field Q (can be transformed per step)

    ) -> "Field[S]":

        # L^0 = {s0}
        L_prev = copy.deepcopy(space)
        L_prev.build_from_states(states = {start}, current=start)

        # Ensure prev_field is defined on L^0 (delta-like by your spec)
        prev_field =  copy.deepcopy(prev_field_input)
        #prev_field.state_space = L_prev
        #prev_field = Field.build_empty_from_state_space(type(prev_field_input),L_prev,prev_field_input.get_unit_field())
        F_l =  copy.deepcopy(prev_field_input)
        #prev_field_input.set_zero_field()
        #prev_field.set_field(start,prev_field.get_unit_field())
        for step in range(1, L + 1):
            # 1) Build the next frontier L^l
           # print("In step: ",step)
            if step==1:
                L_cur = self.build_next_reaching_state_space(start)
            else :
                L_cur = self.build_next_reaching_state_space(start, L_prev)

            # 2) Prepare accumulator F^l over L^l (zero-initialized)
            #print("Number of states in current step ", len(L_cur.get_all_states()) )
            F_l.state_space = L_cur
            F_l.set_constant(single_transformed_field_proto.get_field(start))
            #print("Setting the constant field to output field")
            # 3) Optionally transform the global field once per step
            if self.global_field_transform is not None:
                global_field = self.global_field_transform.apply(global_field, copy.deepcopy(global_field))

            # 4) For each parent s in L^{l-1}

            for s in L_prev.get_all_states():
             #   print("Inside loop of previous states")
                # Reachable children from this parent
                R_s = self.reachable.get_reachable(s)  # space of s' reachable from s in 1 step
              #  print("Fetching the reachable")
                # Single-step field over R_s conditioned on s, then transform it
                raw_single = self.single_step.build_single_step_field(
                    s,              # current_state (parent)
                    global_field=global_field ,   # "global field" used inside the single-step kernel
                    system_field=copy.deepcopy(prev_field) # A reference to the system field which is to be returned
                )
                single_step_F = self.single_step_transform.apply(
                    raw_single,
                    copy.deepcopy(single_step_transformed_field_proto)
                )
               # print("Finished with single step: Number of states returned : ",len(single_step_F.state_space.get_all_states()))
                # Broadcast the parent's previous value F^{l-1}(s:s0) over R_s
                parent_value = prev_field.get_field(s)
                #print(f"Previous field value: {parent_value} at state {s}")
                parent_field_over_Rs = self._broadcast_value_over_space(
                    prev_field, R_s, parent_value
                )
                #print("per_parent: Number of states returned : ",len(parent_field_over_Rs.state_space.get_all_states()))
                # Intrinsic composition: Z^P[ single_step_F , parent_value ]

                per_parent_contrib = self.intrinsic_composer.apply(single_step_F, parent_field_over_Rs)
                #print("Intrinsic composer: Number of states returned : ",len(per_parent_contrib.state_space.get_all_states()))

                F_l = self.extrinsic_composer.apply(F_l, per_parent_contrib)
                #print("Extrinsic composer: Number of states returned : ",len(F_l.state_space.get_all_states()))


            # 5) Optional multi-step transform on F^l (e.g., normalization)
            #print("Doing the multi-step transform")
            prev_field = self.multi_step_transform.apply(F_l, copy.deepcopy(F_l))


            # 6) Advance the frontier
            #print("Finished with current step")
            L_prev = L_cur


        return prev_field
    def _choose_current_generic(self, states: "set[S]", fallback: "S") -> "S":
        """
        Pick a deterministic representative from `states` with no assumptions
        about state fields. Tries natural ordering; falls back to repr.
        """
        if not states:
            return fallback
        try:
            # If states implement a total order (__lt__), use it
            return min(states)
        except Exception:
            try:
                # Otherwise, use a stable lexical order by repr
                return min(states, key=lambda s: repr(s))
            except Exception:
                # Absolute fallback: arbitrary but deterministic enough for small sets
                return next(iter(states))

    def new_multi_step_field(self, start: State, space: "StatSpace[S]",steps,system_field: "Field[S]",global_field: "Field[S]",
                             single_step_field_proto: "Field[S]",
                             one_step_field_proto: "Field[S]",
                            final_step_field_proto: "Field[S]",
                             is_start_indicator: bool=True
                             ) -> "Field[S]":
        if is_start_indicator:
            system_field.set_zero_field()
            system_field.set_field(start, system_field.get_unit_field())
            space = StatSpace.build_from_states({start}, current=start)
        step_space =   StatSpace.build_from_states({start}, current=start)
        ids = space.ids_view()

        for _ in range(0,steps):
            for sid in ids:
                s=space.get_state_by_id(int(sid))
                single_step_field_at_s = self.single_step.build_single_step_field(s, global_field=global_field, system_field=system_field)
                step_space.union_state_space(single_step_field_at_s.state_space)



        pass

    def build_next_reaching_state_space_optimized(
            self,
            initial_state: S,
            prev_reaching_space: "StatSpace[S]" = None,
            *,
            mode: str = "frontier",  # "frontier" (exactly l) or "within" (≤ l)
    ) -> "StatSpace[S]":
        R = self.reachable

        # Collect children
        if prev_reaching_space is None:
            children = set(R.get_reachable(initial_state).get_all_states())
        else:
            children: Set[S] = set()
            sp = prev_reaching_space
            ids = getattr(sp, "ids_view", None)
            sid2state = getattr(sp, "get_state_by_id", None)
            if callable(ids) and callable(sid2state):
                for sid in ids():
                    s = sid2state(int(sid))
                    children.update(R.get_reachable(s).get_all_states())
            else:
                for s in sp.get_all_states():
                    children.update(R.get_reachable(s).get_all_states())

        # Target set: frontier or within
        if mode == "within" and prev_reaching_space is not None:
            target = set(prev_reaching_space.get_all_states())
            target.update(children)
        else:
            target = children

        if not target:
            # empty: keep {initial_state} to remain stable
            return rebuild_like_space(R.get_reachable(initial_state), {initial_state}, initial_state)

        # deterministic current
        try:
            cur = min(target)
        except Exception:
            cur = min(target, key=lambda x: repr(x))
        return rebuild_like_space(R.get_reachable(initial_state), target, cur)

    def generate_multi_step_field_iteration_optimized(
            self,
            space: "StatSpace[S]",  # ambient (type hint only)
            start: S,  # s0
            L: int,  # steps
            single_transformed_field_proto: "Field[S]" = None,  # proto for F^l (accumulator/output)
            single_step_transformed_field_proto: "Field[S]" = None,  # proto for per-parent single-step transform
            prev_field_input: "Field[S]" = None,  # mutable: allowed to modify
            global_field: "Field[S]" = None
    ) -> "Field[S]":

        # L^0 = {s0}
        L_prev = rebuild_like_space(space, {start}, start)

        # prev_field starts defined on L^0 (caller said mutation allowed)
        prev_field = prev_field_input
        prev_field.state_space = L_prev

        # initialize to delta at s0 if caller didn't set it yet
        if getattr(prev_field, "set_constant", None) is not None:
            prev_field.set_constant(prev_field.get_unit_field())
        else:
            # minimal: write unit at s0, zero elsewhere
            if hasattr(prev_field, "set_zero_field"): prev_field.set_zero_field()
            prev_field.set_field(start, prev_field.get_unit_field())

        # Accumulator prototype sanity
        if single_transformed_field_proto is None:
            single_transformed_field_proto = prev_field
        if single_step_transformed_field_proto is None:
            single_step_transformed_field_proto = prev_field

        # MAIN LOOP
        for step in range(1, L + 1):
            # 1) Next frontier
            L_cur = self.build_next_reaching_state_space(start, None if step == 1 else L_prev, mode="frontier")

            # 2) Prepare F^l over L^l (zero-initialized)
            F_l = spawn_field_like(single_transformed_field_proto, L_cur)
            if hasattr(F_l, "set_zero_field"): F_l.set_zero_field()

            # 3) Transform global field once per step (if configured)
            if self.global_field_transform is not None and global_field is not None:
                global_field = self.global_field_transform.apply(global_field, spawn_field_like(global_field,
                                                                                                global_field.state_space))

            # 4) For each parent s in L^{l-1}
            sp = L_prev
            ids = getattr(sp, "ids_view", None)
            sid2state = getattr(sp, "get_state_by_id", None)
            iter_states = (sid2state(int(i)) for i in ids()) if callable(ids) and callable(
                sid2state) else sp.get_all_states()

            for s in iter_states:
                # Reachable children R(s)
                R_s = self.reachable.get_reachable(s)

                # Single-step field over R_s (conditioned on s), then transform
                raw_single = self.single_step.build_single_step_field(
                    current_state=s,
                    system_field=spawn_field_like(prev_field, R_s),  # empty shell for type compatibility
                    global_field=global_field
                )
                single_step_F = self.single_step_transform.apply(
                    raw_single,
                    spawn_field_like(single_step_transformed_field_proto, raw_single.state_space)
                )

                # Broadcast the parent's previous value over R_s
                parent_value = prev_field.get_field(s)
                if parent_value is None:
                    continue  # nothing to contribute
                parent_over_Rs = broadcast_constant(prev_field, R_s, parent_value)

                # Intrinsic: Z^P  (combine per-child contribution with parent weight)
                per_parent = self.intrinsic_composer.apply(single_step_F, parent_over_Rs)

                # Extrinsic: Z^t  (accumulate contributions from all parents)
                F_l = self.extrinsic_composer.apply(F_l, per_parent)

            # 5) Optional multi-step transform on F^l
            prev_field = self.multi_step_transform.apply(F_l, spawn_field_like(F_l, F_l.state_space))

            # 6) Advance frontier
            L_prev = L_cur

        return prev_field

    def generate_multi_step_field(
            self,
            space: "StatSpace[S]",  # ambient/type anchor
            start: S,
            L: int,
            single_transformed_field_proto: "Field[S]" = None,
            single_step_transformed_field_proto: "Field[S]" = None,
            prev_field_input: "Field[S]" = None,  # may be mutated (as you allowed)
            global_field: "Field[S]" = None
    ) -> "Field[S]":
        """
        One-pass per step:
          - build single-step fields for each parent
          - accumulate per-child contributions in a dict
          - frontier = dict.keys()
          - materialize at the end of the step
        No deepcopies; uses spawn-based factories.
        """

        # ---- helpers (as in your optimized pipeline) ----
        def spawn_field_like(proto_field: "Field[S]", space_out) -> "Field[S]":
            spawn = getattr(proto_field, "spawn", None)
            if callable(spawn): return spawn(space_out)
            from_space = getattr(type(proto_field), "from_space", None)
            if callable(from_space): return from_space(space_out)
            import copy as _copy
            nf = _copy.copy(proto_field)
            nf.state_space = space_out
            if hasattr(nf, "set_empty_field"): nf.set_empty_field()
            if hasattr(nf, "set_zero_field"):  nf.set_zero_field()
            return nf

        def build_space(states_iter, proto_space, current=None):
            # prefer build_like if available to preserve concrete type
            bl = getattr(proto_space, "build_like", None)
            if callable(bl):
                return bl(states_iter, current=current)
            # generic finite fallback (adjust import if needed)
            from fds.core.fds_state.state_space import DiscreteFiniteStatSpace
            states_list = list(states_iter)
            cur = current if current in set(states_list) else (states_list[0] if states_list else None)
            return DiscreteFiniteStatSpace(states_list, current=cur)

        # ---- initialize L^0 and prev_field on L^0 ----
        L_prev = build_space([start], space, current=start)

        prev_field = prev_field_input
        prev_field.state_space = L_prev
        # ensure delta at s0 if not already set
        if hasattr(prev_field, "set_zero_field"): prev_field.set_zero_field()
        prev_field.set_field(start, prev_field.get_unit_field())

        # fill default prototypes
        if single_transformed_field_proto is None:
            single_transformed_field_proto = prev_field
        if single_step_transformed_field_proto is None:
            single_step_transformed_field_proto = prev_field

        # aliases for compositions and transforms
        intr = self.intrinsic_composer.composer.compose  # SingleFieldValue ⊗ SingleFieldValue
        extr = self.extrinsic_composer.composer.compose  # reducer for per-child accumulation

        for step in range(1, L + 1):
            # optional: transform global field once per step
            if self.global_field_transform is not None and global_field is not None:
                # supply a clean proto as output target
                global_field = self.global_field_transform.apply(
                    global_field, spawn_field_like(global_field, global_field.state_space)
                )

            # ---- per-step accumulator: child_state -> SingleFieldValue ----
            # we accumulate at SingleFieldValue level; wrap into FieldValue at the end
            acc: dict[S, "SingleFieldValue"] = {}

            sp = L_prev
            ids = getattr(sp, "ids_view", None)
            sid2state = getattr(sp, "get_state_by_id", None)
            parent_iter = (sid2state(int(i)) for i in ids()) if callable(ids) and callable(
                sid2state) else sp.get_all_states()

            # cache lookups
            get_parent_val = prev_field.get_field
            unit_out = single_transformed_field_proto.get_unit_field()
            wrap = type(unit_out)  # FieldValue constructor (data, norm, addition)

            for s in parent_iter:
                parent_fv = get_parent_val(s)
                if parent_fv is None:
                    continue
                parent_data = parent_fv.data  # SingleFieldValue

                # build raw single-step field over R(s)
                raw_single = self.single_step.build_single_step_field(
                    current_state=s,
                    system_field=spawn_field_like(prev_field, L_prev),  # ignored container; just matches type
                    global_field=global_field
                )
                # optional transform of single-step field
                single_step_F = self.single_step_transform.apply(
                    raw_single, spawn_field_like(single_step_transformed_field_proto, raw_single.state_space)
                )

                # iterate children in R(s) and accumulate
                rsp = single_step_F.state_space
                r_ids = getattr(rsp, "ids_view", None)
                r_sid2state = getattr(rsp, "get_state_by_id", None)

                # prefer ids for large spaces; fallback to states
                if callable(r_ids) and callable(r_sid2state):
                    for sid in r_ids():
                        t = r_sid2state(int(sid))
                        step_fv = single_step_F.get_field(t)
                        if step_fv is None:
                            continue
                        # intrinsic: combine child contribution with parent weight
                        contrib = intr(step_fv.data, parent_data)  # SingleFieldValue
                        # extrinsic: accumulate into child bucket
                        prev = acc.get(t)
                        acc[t] = contrib if prev is None else extr(prev, contrib)
                else:
                    for t in rsp.get_all_states():
                        step_fv = single_step_F.get_field(t)
                        if step_fv is None:
                            continue
                        contrib = intr(step_fv.data, parent_data)
                        prev = acc.get(t)
                        acc[t] = contrib if prev is None else extr(prev, contrib)

            # ---- materialize frontier & F^l from accumulator ----
            if not acc:
                # nothing reached → keep prev_field as-is and break early
                break

            # Build L^l directly from the accumulator keys
            # choose a deterministic current
            try:
                cur = min(acc.keys())
            except Exception:
                cur = min(acc.keys(), key=lambda x: repr(x))
            L_cur = build_space(acc.keys(), space, current=cur)

            # Write accumulated values into F_l
            F_l = spawn_field_like(single_transformed_field_proto, L_cur)
            if hasattr(F_l, "set_zero_field"): F_l.set_zero_field()

            for t, data in acc.items():
                F_l.set_field(t, wrap(data, unit_out.normTransform, unit_out.additionComposition))

            # multi-step transform to obtain F^{l} (becomes prev_field for next loop)
            prev_field = self.multi_step_transform.apply(F_l, spawn_field_like(F_l, F_l.state_space))

            # advance frontier
            L_prev = L_cur

        return prev_field




