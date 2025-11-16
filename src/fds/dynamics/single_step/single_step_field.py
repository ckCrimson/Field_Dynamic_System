from typing import Generic, Optional, TypeVar

from fds import State, FieldValue, Field
from fds.core.fds_field.single_field_value_composition import Composition
from fds.core.fds_state import Reachable
from fds.dynamics.single_step.kernel import Kernel

S = TypeVar('S', bound=State)
class SingleStepField(Generic[S]):
    """
    Builds a single-step field over the reachable subspace using a Kernel and an optional system field.
    Optionally applies a shaping Field Q (e.g., from Δ own-global field), defaulting to unit if None.
    """
    def __init__(
        self,
        kernel: Kernel[S]=None,
        reachable: "Reachable[S]"=None,
        kernelFieldComposition: Composition=None,
        postComposition: Optional[Composition] = None,  # optional: how to combine with Q
    ):
        self.kernel = kernel
        self.reachable = reachable
        self.kernelFieldComposition = kernelFieldComposition
        # if not provided, reuse the same composition operator for shaping
        self.postComposition = postComposition or kernelFieldComposition

    def build_single_step_field(
        self,
        current_state: S,
        system_field: Field[S],
        global_field: "Field[S]"=None,
        q_field: Optional["Field[S]"] = None,   # <-- NEW: optional shaping field
    ) -> "Field[S]":
        """
        Construct Field[S] over the reachable subspace:
          - R := Reachable(state_space, current_state)
          - new_field(s) = compose( kernel(s, current_state), system_field(s) )
          - if q_field is provided: new_field(s) = postCompose( new_field(s), q_field(s) )
        """
        # 1) reachable set/space
        reachable_space = self.reachable.get_reachable(current_state)
        states_iter = reachable_space.ids_view()

        # 2) initialize the output field over the same space, empty/unit by default
        new_field_class = type(system_field)
        new_field = new_field_class.build_empty_from_state_space(new_field_class,reachable_space,system_field.get_unit_field())
        #new_field.state_space.build_from_states(states_iter)
        #new_field.set_zero_field()
        #new_field.set_field(current_state,system_field.get_unit_field())
        # 3) per-state composition (kernel ⊗ system) and optional shaping (⊗ Q)
        for sid in states_iter:
            #print("H")
            s=reachable_space.get_state_by_id(int(sid))
            k_val = self.kernel.get_kernel_value(s, current_state)     # FieldValue / SingleFieldValue
            if global_field is not None:
                f_val = global_field.get_field(s)                      # FieldValue / SingleFieldValue
            else:
                f_val = system_field.get_unit_field()
            if f_val is not None:
                composed = self.kernelFieldComposition.compose(k_val.data, f_val.data)
            else:
                composed=k_val

            if q_field is not None:
                q_val = q_field.get_field(s)                           # FieldValue / SingleFieldValue
                composed = self.postComposition.compose(composed.data, q_val.data)
            composed_field_value = FieldValue(composed,system_field.get_unit_field().normTransform,system_field.get_unit_field().additionComposition )
            #print(f"Composed at {s} = {composed_field_value.data.get_data()}")
            new_field.set_field(s, composed_field_value)


        return new_field

    def build_single_step_field_new(
            self,
            current_state: S,
            system_field: "Field[S]",
            global_field: "Field[S]" = None,
            q_field: Optional["Field[S]"] = None,
    ) -> "Field[S]":
        """
        new_field(s) = compose( kernel(s, current_state), global_field(s) )
        If q_field is provided: new_field(s) = postCompose( new_field(s), q_field(s) )
        """
        import numpy as np
        reachable_space = self.reachable.get_reachable(current_state)

        # ---- spawn output field over reachable space (no ctor assumptions) ----
        def _spawn_field_like(proto_field: "Field[S]", space_out) -> "Field[S]":
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

        out = _spawn_field_like(system_field, reachable_space)
        if hasattr(out, "set_empty_field"): out.set_empty_field()
        if hasattr(out, "set_zero_field"):  out.set_zero_field()

        # ---- cache hot callables ----
        unit = system_field.get_unit_field()
        k_comp = self.kernelFieldComposition.compose
        post_comp = self.postComposition.compose
        k_get = self.kernel.get_kernel_value  # expects (state, current_state)

        gf_get = None if global_field is None else global_field.get_field
        qf_get = None if q_field is None else q_field.get_field
        out_set = out.set_field

        sp = reachable_space
        have_ids = callable(getattr(sp, "ids_view", None)) and callable(getattr(sp, "get_state_by_id", None))
        get_id = getattr(sp, "get_id", None)
        current_id = None
        if have_ids and callable(get_id):
            try:
                current_id = get_id(current_state)
            except Exception:
                current_id = None

        # Optional by-id kernel/fields
        k_get_by_id = getattr(self.kernel, "get_kernel_value_by_id", None)  # (sid, current_id)
        gf_get_by_id = None if global_field is None else getattr(global_field, "get_field_by_id", None)
        qf_get_by_id = None if q_field is None else getattr(q_field, "get_field_by_id", None)

        # ---- FAST PATH: iterate by ids ----
        if have_ids:
            ids = sp.ids_view()
            if ids.dtype != np.int32: ids = ids.astype(np.int32, copy=False)
            sid_to_state = sp.get_state_by_id

            use_k_by_id = callable(k_get_by_id) and (current_id is not None)
            use_gf_by_id = callable(gf_get_by_id) if gf_get_by_id is not None else False
            use_qf_by_id = callable(qf_get_by_id) if qf_get_by_id is not None else False

            for sid in ids:
                i = int(sid)

                if use_k_by_id:
                    k_val = k_get_by_id(i, current_id)  # <- (sid, current_id)
                else:
                    s = sid_to_state(i)
                    k_val = k_get(s, current_state)  # <- pass current_state

                # compose with global_field if present
                if global_field is not None:
                    if use_gf_by_id:
                        f_val = gf_get_by_id(i)
                    else:
                        s = sid_to_state(i) if 's' not in locals() else s
                        f_val = gf_get(s)
                    composed_sfv = k_comp(k_val.data, f_val.data) if f_val is not None else k_val.data
                else:
                    composed_sfv = k_val.data

                # optional shaping
                if q_field is not None:
                    if use_qf_by_id:
                        q_val = qf_get_by_id(i)
                    else:
                        s = sid_to_state(i) if 's' not in locals() else s
                        q_val = qf_get(s)
                    if q_val is not None:
                        composed_sfv = post_comp(composed_sfv, q_val.data)

                out_set(sid_to_state(i), type(unit)(composed_sfv, unit.normTransform, unit.additionComposition))
            return out

        # ---- FALLBACK: iterate over state objects ----
        for s in sp.get_all_states():
            k_val = k_get(s, current_state)  # <- pass current_state

            if global_field is not None:
                f_val = gf_get(s)
                composed_sfv = k_comp(k_val.data, f_val.data) if f_val is not None else k_val.data
            else:
                composed_sfv = k_val.data

            if q_field is not None:
                q_val = qf_get(s)
                if q_val is not None:
                    composed_sfv = post_comp(composed_sfv, q_val.data)

            out_set(s, type(unit)(composed_sfv, unit.normTransform, unit.additionComposition))

        return out
