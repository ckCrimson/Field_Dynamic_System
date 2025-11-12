from typing import ClassVar, Optional, TypeVar, Generic, Iterable
import numpy as np

from fds import FieldValue, Field
from fds.core.fds_field.single_field_value_composition import Composition
from fds.core.fds_state.state_space import DiscreteFiniteStatSpace

S = TypeVar("S")

class StatelessComposeField(Generic[S]):
    """
    Composer configured by class-level (static) constants.
    Subclasses set COMPOSER / ZERO_* and optionally override BUILD_OUT.
    """
    __slots__ = ()  # no instance dict; stateless

    # Static configuration (override in subclasses)
    COMPOSER: ClassVar["Composition"]              # required
    ZERO_COMPOSE: ClassVar[Optional["FieldValue"]] = None  # if None → treat as out.get_zero_field()
    ZERO_F1: ClassVar[Optional["FieldValue"]] = None       # if None → f1.get_zero_field()
    ZERO_F2: ClassVar[Optional["FieldValue"]] = None       # if None → f2.get_zero_field()

    @classmethod
    def _clone_fv(cls, fv: "FieldValue") -> "FieldValue":
        # If FieldValue is immutable, you can just return fv.
        # If it may be mutable, create a fresh wrapper around same data:
        try:
            return type(fv)(fv.data, fv.normTransform, fv.additionComposition)
        except Exception:
            import copy as _copy
            return _copy.deepcopy(fv)

    @classmethod
    def _build_space_like(cls, proto_space, states: Iterable[S], current: Optional[S]):
        # Prefer a subclass- or space-provided factory
        bl = getattr(proto_space, "build_like", None)
        if callable(bl):
            return bl(states, current=current)
        # Generic finite fallback (adjust import path to your project)
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
        new_field: Field[S] = type(proto_field)(space_out, space_out.get_state())
        # If your Field requires ctor args, add a factory on Field and use that instead.
        new_field.state_space = space_out
        return new_field

    @classmethod
    def apply(cls, f1: "Field[S]", f2: "Field[S]", *, return_field_proto: "Field[S]") -> "Field[S]":
        """
        Compose f1 and f2 into a NEW field over union(f1.space, f2.space),
        using class-level static policy (COMPOSER, ZERO_*).
        """
        sp1, sp2 = f1.state_space, f2.state_space
        proto_space = return_field_proto.state_space

        # ----- fast path if we can iterate by ids on a single universe -----
        same_object = (sp1 is sp2)
        have_ids = callable(getattr(sp1, "ids_view", None)) and callable(getattr(sp1, "get_state_by_id", None))

        get_zero_1 = (lambda: cls.ZERO_F1) if (cls.ZERO_F1 is not None) else f1.get_zero_field
        get_zero_2 = (lambda: cls.ZERO_F2) if (cls.ZERO_F2 is not None) else f2.get_zero_field

        zero_comp = cls.ZERO_COMPOSE
        compose = cls.COMPOSER.compose

        if same_object and have_ids:
            ids = sp1.ids_view()
            if ids.dtype != np.int32:
                ids = ids.astype(np.int32, copy=False)
            get_state_by_id = sp1.get_state_by_id
            ordered_states = [get_state_by_id(int(i)) for i in ids]

            cur = sp1.get_state() if sp1.contains(sp1.get_state()) else (ordered_states[0] if ordered_states else None)
            space_out = cls._build_space_like(proto_space, ordered_states, cur)
            out = cls._spawn_field_like(return_field_proto, space_out)
            if hasattr(out, "set_empty_field"): out.set_empty_field()
            if hasattr(out, "set_zero_field"):  out.set_zero_field()
            unit = out.get_unit_field()

            f1_get, f2_get, out_set = f1.get_field, f2.get_field, out.set_field

            for s in ordered_states:
                a = f1_get(s); b = f2_get(s)
                if a is None and b is None:
                    out_set(s, zero_comp if zero_comp is not None else cls._clone_fv(unit))  # safe default
                    continue
                if a is None: a = get_zero_1()
                if b is None: b = get_zero_2()
                sfv = compose(a.data, b.data)
                fv_out = type(unit)(sfv, unit.normTransform, unit.additionComposition)
                out_set(s, fv_out)
            return out

        # ----- generic fallback: set-of-States union -----
        uni = set(sp1.get_all_states()) | set(sp2.get_all_states())
        cur = sp1.get_state() if sp1.contains(sp1.get_state()) else (next(iter(uni)) if uni else None)
        space_out = cls._build_space_like(proto_space, uni, cur)
        out = cls._spawn_field_like(return_field_proto, space_out)
        if hasattr(out, "set_empty_field"): out.set_empty_field()
        if hasattr(out, "set_zero_field"):  out.set_zero_field()
        unit = out.get_unit_field()

        f1_get, f2_get, out_set = f1.get_field, f2.get_field, out.set_field
        for s in uni:
            a = f1_get(s); b = f2_get(s)
            if a is None and b is None:
                out_set(s, zero_comp if zero_comp is not None else cls._clone_fv(unit))
                continue
            if a is None: a = get_zero_1()
            if b is None: b = get_zero_2()
            sfv = compose(a.data, b.data)
            fv_out = type(unit)(sfv, unit.normTransform, unit.additionComposition)
            out_set(s, fv_out)
        return out
