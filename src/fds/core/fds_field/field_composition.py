from __future__ import annotations


from typing import Generic, Optional, TypeVar,  Iterable

from fds.core.fds_field.field_value import FieldValue
from fds.core.fds_field.fields import Field
from fds.core.fds_field.single_field_value_composition import Composition
from fds.core.fds_state.state import State
from fds.core.fds_state.state_space import DiscreteFiniteStatSpace, StatSpace

S = TypeVar("S", bound=State)

class ComposeField(Generic[S]):
    """
    Stateless composer: NEW field = compose(f1, f2) pointwise over the UNION of their state spaces.
    Missing entries are treated as zeros (configurable per-input and per-composition).
    """

    def __init__(
        self,
        composer: Composition =None,
        return_field_proto: Field[S] =None,
        zero_compose: Optional[FieldValue] = None,
        zero_field_1: Optional[FieldValue] = None,
        zero_field_2: Optional[FieldValue] = None,
    ):
        self.composer = composer
        self.return_field_proto = return_field_proto
        self.zero_compose = zero_compose or return_field_proto.get_zero_field()
        self.zero_field_1 = zero_field_1          # fallback is f1.get_zero_field()
        self.zero_field_2 = zero_field_2          # fallback is f2.get_zero_field()

    def apply(self, f1: "Field[S]", f2: "Field[S]") -> "Field[S]":
        """
        Compose f1 and f2 into a NEW field over union(f1.space, f2.space).
        Optimized to iterate by ids using f1's state_space as the reference universe.
        If mapping f2's states into f1 fails, falls back to generic set/State union.
        Does not mutate inputs or the prototype.
        """
        import numpy as np

        sp_ref = f1.state_space  # reference universe
        sp_other = f2.state_space
        proto_space = self.return_field_proto.state_space

        # cache hot callables
        get_zero_1 = (lambda: self.zero_field_1) if (self.zero_field_1 is not None) else f1.get_zero_field
        get_zero_2 = (lambda: self.zero_field_2) if (self.zero_field_2 is not None) else f2.get_zero_field
        zero_comp = self.zero_compose
        compose = self.composer.compose

        # ---------- try ID-first path with f1 as reference ----------
        try:
            # ids present in f1's space
            if callable(getattr(sp_ref, "ids_view", None)):
                ids1 = sp_ref.ids_view()
                if ids1.dtype != np.int32:
                    ids1 = ids1.astype(np.int32, copy=False)
            else:
                # build ids for f1 via mapping its own states (rare, but keeps us generic)
                states1 = list(sp_ref.get_all_states())
                ids1 = np.fromiter((sp_ref.get_id(s) for s in states1), dtype=np.int32, count=len(states1))

            # map f2's states into f1's id universe
            states2 = list(sp_other.get_all_states())
            ids2 = np.fromiter((sp_ref.get_id(s) for s in states2), dtype=np.int32, count=len(states2))

            # compute deterministic union of ids
            union_ids = np.union1d(ids1, ids2)

            # build output space ONCE from reference-space ids → states
            get_state_by_id = sp_ref.get_state_by_id
            ordered_states = [get_state_by_id(int(i)) for i in union_ids]

            # choose current
            cur_ref = sp_ref.get_state()
            current = cur_ref if sp_ref.contains(cur_ref) else (ordered_states[0] if ordered_states else None)

            space_out = self._build_space_like(proto_space, set(ordered_states), current)
            out = self._spawn_field_like(self.return_field_proto, space_out)
            out.set_empty_field()
            out.set_zero_field()
            unit = out.get_unit_field()

            f1_get = f1.get_field
            f2_get = f2.get_field
            out_set = out.set_field

            # single pass over ordered ids/states
            for s in ordered_states:
                a = f1_get(s)
                b = f2_get(s)

                if a is None and b is None:
                    out_set(s, zero_comp)
                    continue
                if a is None: a = get_zero_1()
                if b is None: b = get_zero_2()

                sfv_out = compose(a.data, b.data)
                fv_out = FieldValue(sfv_out, unit.normTransform, unit.additionComposition)
                out_set(s, fv_out)

            return out

        except KeyError:
            # Some state from f2 isn't in f1's universe → different universes. Fall back.
            pass
        except AttributeError:
            # Missing hooks on spaces; fall back.
            pass

        # ---------- generic fallback: set-of-State union (correct for any universes) ----------
        s1 = sp_ref.get_all_states()
        s2 = sp_other.get_all_states()
        union_states = set(s1) | set(s2)

        current = sp_ref.get_state() if sp_ref.contains(sp_ref.get_state()) else (
            next(iter(union_states)) if union_states else None)
        space_out = self._build_space_like(proto_space, union_states, current)
        out = self._spawn_field_like(self.return_field_proto, space_out)
        out.set_empty_field()
        out.set_zero_field()
        unit = out.get_unit_field()

        f1_get = f1.get_field
        f2_get = f2.get_field
        out_set = out.set_field

        for s in union_states:
            a = f1_get(s)
            b = f2_get(s)
            if a is None and b is None:
                out_set(s, zero_comp)
                continue
            if a is None: a = get_zero_1()
            if b is None: b = get_zero_2()
            sfv_out = compose(a.data, b.data)
            fv_out = FieldValue(sfv_out, unit.normTransform, unit.additionComposition)
            out_set(s, fv_out)

        return out

    # ---------------- helpers: no-input-mutation builders ----------------

    @staticmethod
    def _build_space_like(proto_space: StatSpace, states: Iterable[S], current: Optional[S]):
        # Prefer a subclass-provided build_like() if present
        bl = getattr(proto_space, "build_like", None)
        if callable(bl):
            return bl(states, current=current)
        # Generic finite fallback: avoid guessing subclass constructors
        return DiscreteFiniteStatSpace(states=states, current=current)

    @staticmethod
    def _spawn_field_like(proto_field: Field[S], space_out) -> Field[S]:
        """
        Create a NEW Field of the same concrete type as proto_field, bound to 'space_out'.
        Prefer a factory on Field (e.g., 'spawn(space)'); else shallow-construct and rebind.
        """
        if hasattr(proto_field, "spawn"):
            # type: ignore[attr-defined]
            return proto_field.spawn(space_out)

        # Fallback: reinstantiate and copy essential configuration.
        # Assumes Field can be constructed with (state_space=...) or set afterward.
        new_field: Field[S] = type(proto_field)(space_out,space_out.get_state())
        # If your Field requires ctor args, add a factory on Field and use that instead.
        new_field.state_space = space_out
        return new_field

    @staticmethod
    def _clone_field_value(fv: FieldValue) -> FieldValue:
        """
        If FieldValue is immutable, just return it (safe to reuse).
        If it's mutable, define/require a .clone() method; fallback to copying constructor if available.
        """
        if hasattr(fv, "clone"):
            # type: ignore[attr-defined]
            return fv.clone()
        # Fallback: construct a new FieldValue from its data/metadata if your API allows.
        try:
            return FieldValue(fv.data, fv.normTransform, fv.additionComposition)
        except Exception:
            # Last resort: use copy.deepcopy (rarely needed; avoided in fast paths)
            import copy as _copy
            return _copy.deepcopy(fv)

