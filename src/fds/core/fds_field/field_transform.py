from __future__ import annotations
from typing import TypeVar

from fds.core.fds_field.fields import Field
from fds.core.fds_field.single_field_value_transform import Transform  # assuming this name
from fds.core.fds_state.state import State

S = TypeVar("S", bound=State)


from typing import Generic, Optional, Iterable
import numpy as np
import copy as _copy


class TransformField(Generic[S]):
    """
    Applies a Transform (operating on SingleFieldValue) pointwise to a Field,
    producing a NEW Field. No in-place mutation. No shared instances.
    """
    __slots__ = ("transform",)

    def __init__(self, transform: Optional["Transform"] = None):
        self.transform = transform  # None => identity at SingleFieldValue level

    def apply(self, field_in: "Field[S]", field_out_proto: "Field[S]") -> "Field[S]":
        """
        Build a fresh field of the same 'kind' as field_out_proto over the SAME
        state space as field_in, and write transformed values. Inputs are not mutated.
        Finite spaces get a fast id-iteration path when available.
        """
        in_space = field_in.state_space

        # ---------- build the output space (same universe as input) ----------
        # Prefer a subclass/space factory; else generic finite rebuild.
        def _build_space_like(proto_space, src_space):
            bl = getattr(proto_space, "build_like", None)
            if callable(bl):
                # populate with exactly the states of src_space (order preserved if ids exist)
                if callable(getattr(src_space, "ids_view", None)) and callable(getattr(src_space, "get_state_by_id", None)):
                    ids = src_space.ids_view()
                    if ids.dtype != np.int32: ids = ids.astype(np.int32, copy=False)
                    states_iter = (src_space.get_state_by_id(int(i)) for i in ids)
                    return bl(states_iter, current=src_space.get_state())
                # fallback: generic iteration
                states = list(src_space.get_all_states())
                return bl(states, current=src_space.get_state())

            # Generic finite fallback without assuming constructor signature
            # Import your concrete finite space if you want to force that type here.
            try:
                from fds.core.fds_state.state_space import DiscreteFiniteStatSpace  # adjust import if needed
                if callable(getattr(src_space, "ids_view", None)) and callable(getattr(src_space, "get_state_by_id", None)):
                    ids = src_space.ids_view()
                    if ids.dtype != np.int32: ids = ids.astype(np.int32, copy=False)
                    states_iter = (src_space.get_state_by_id(int(i)) for i in ids)
                    return DiscreteFiniteStatSpace(states_iter, current=src_space.get_state())
                return DiscreteFiniteStatSpace(src_space.get_all_states(), current=src_space.get_state())
            except Exception:
                # Last resort: shallow copy space if your class supports it; otherwise return src_space (documented limitation)
                return src_space

        space_out = _build_space_like(field_out_proto.state_space, in_space)

        # ---------- create the output field without assuming ctor signature ----------
        out = self._spawn_field_like(field_out_proto, space_out)
        if hasattr(out, "set_empty_field"): out.set_empty_field()
        if hasattr(out, "set_zero_field"):  out.set_zero_field()

        # ---------- cache hot callables ----------
        out_set = out.set_field
        fin_get = field_in.get_field
        unit = out.get_unit_field()
        zero_out = out.get_zero_field  # callable; may build lazily

        # ---------- ID-FAST PATH (no Python sets) ----------
        if callable(getattr(in_space, "ids_view", None)) and callable(getattr(in_space, "get_state_by_id", None)):
            ids = in_space.ids_view()
            if ids.dtype != np.int32:
                ids = ids.astype(np.int32, copy=False)
            get_state_by_id = in_space.get_state_by_id

            if self.transform is None:
                # Identity: copy through only non-None entries; write zero otherwise
                for sid in ids:
                    s = get_state_by_id(int(sid))
                    fv = fin_get(s)
                    if fv is None:
                        out_set(s, zero_out())
                    else:
                        out_set(s, type(unit)(fv.data, unit.normTransform, unit.additionComposition))
                return out

            apply_sfv = self.transform.apply
            for sid in ids:
                s = get_state_by_id(int(sid))
                fv = fin_get(s)
                if fv is None:
                    out_set(s, zero_out())
                    continue
                sfv_out = apply_sfv(fv.data)
                out_set(s, type(unit)(sfv_out, unit.normTransform, unit.additionComposition))
            return out

        # ---------- GENERIC FALLBACK (set-of-states) ----------
        try:
            states: Iterable[S] = in_space.get_all_states()
        except AttributeError as e:
            raise TypeError("TransformField.apply expects a finite StateSpace with get_all_states().") from e

        if self.transform is None:
            for s in states:
                fv = fin_get(s)
                if fv is None:
                    out_set(s, zero_out())
                else:
                    out_set(s, type(unit)(fv.data, unit.normTransform, unit.additionComposition))
            return out

        apply_sfv = self.transform.apply
        for s in states:
            fv = fin_get(s)
            if fv is None:
                out_set(s, zero_out())
                continue
            sfv_out = apply_sfv(fv.data)
            out_set(s, type(unit)(sfv_out, unit.normTransform, unit.additionComposition))
        return out

    # ---------- field factory helper (robust) ----------
    @staticmethod
    def _spawn_field_like(proto_field: "Field[S]", space_out) -> "Field[S]":
        """
        Prefer factories; otherwise shallow-copy the prototype and rebind its space.
        Never call the subclass __init__ directly (constructor shapes vary).
        """
        spawn = getattr(proto_field, "spawn", None)
        if callable(spawn):
            return spawn(space_out)
        from_space = getattr(type(proto_field), "from_space", None)
        if callable(from_space):
            return from_space(space_out)
        nf = _copy.copy(proto_field)
        nf.state_space = space_out
        return nf

