import copy
from typing import Any, ClassVar, Optional
from matplotlib import pyplot as plt


from fds import FieldValue, Field, State
from fds.core.fds_field import ComposeField, TransformField
from fds.core.fds_field.field_function import FieldFunction, S
from fds.core.fds_field.single_field_value import SingleFieldValue
from fds.core.fds_field.single_field_value_composition import Composition, AdditionComposition
from fds.core.fds_field.single_field_value_transform import  NormTransform
from fds.core.fds_field.stateless_field_composition import StatelessComposeField
from one_dim_random_walker.core.states.state import IntegerState
from one_dim_random_walker.core.states.state_space import IntegerLine


# --------- Each Real Field with its operations defined -----------#
class RealSingleFieldValue(SingleFieldValue):
    value:  float

    def __add__(self, other):
        return RealSingleFieldValue(other.value+self.value)

    def __sub__(self, other):
        return RealSingleFieldValue(other.value-self.value)

    def __mul__(self, other):
        return RealSingleFieldValue(other.value*self.value)

    def __eq__(self, other):
        return other.value==self.value

class RealFieldValueAddition(AdditionComposition):

    def __init__(self):
        super().__init__()

    def compose(self, input1: RealSingleFieldValue, input2: RealSingleFieldValue) -> RealSingleFieldValue:
        return RealSingleFieldValue(input1.value + input2.value)

class RealFieldValueSubtraction(Composition):
    def __init__(self):
        super().__init__()

    def compose(self, input1: RealSingleFieldValue, input2: RealSingleFieldValue) -> RealSingleFieldValue:
        return RealSingleFieldValue(input1.value - input2.value)

class RealFieldValueMultiplication(Composition):
    def __init__(self):
        super().__init__()

    def compose(self, input1: RealSingleFieldValue, input2: RealSingleFieldValue) -> RealSingleFieldValue:
        return RealSingleFieldValue(input1.value * input2.value)

class RealFieldValueNorm(NormTransform):
    def __init__(self):
        super().__init__()

    def apply(self,input_field: RealSingleFieldValue) -> RealSingleFieldValue:
        return RealSingleFieldValue(abs(input_field.value))

class RealFieldValue(FieldValue):

    def get_unitary_field(self)->'RealFieldValue':
        return RealFieldValue(RealSingleFieldValue(1))

    def get_zero_field(self) -> 'RealFieldValue':
        return RealFieldValue(RealSingleFieldValue(0))

RealFieldValue.configure(RealFieldValueNorm(),RealFieldValueAddition())
# ------- Real Field Class --------------#

# RealField.py


class RealField(Field[IntegerState]):
    # class-level singletons (avoid re-allocating on every instance)
    _UNIT = RealFieldValue(RealSingleFieldValue(1))
    _ZERO = RealFieldValue(RealSingleFieldValue(0))

    # --- Primary constructor: takes a space. This is what internal code will use.
    def __init__(self, space: IntegerLine, current_state: Optional[IntegerState] = None,
                 field_function: Optional[FieldFunction] = None):
        super().__init__(state_space=space, unit_field=self._UNIT, field_function=field_function)
        self.zero_field = self._ZERO
        # ensure current is consistent


    # --- Ergonomic constructors for users (never used by composer/spawn) ---
    @classmethod
    def from_length(cls, real_line_length: int, current_state: IntegerState,
                    field_function: Optional[FieldFunction] = None) -> "RealField":
        space = IntegerLine(current_state=current_state,
                            left_limit=current_state.state-int(real_line_length),
                            right_limit=current_state.state+int(real_line_length))
        return cls(space=space, current_state=current_state, field_function=field_function)

    @classmethod
    def from_limits(cls, left_limit: int, right_limit: int, current_state: IntegerState,
                    field_function: Optional[FieldFunction] = None) -> "RealField":
        space = IntegerLine(current_state=current_state,
                            left_limit=int(left_limit),
                            right_limit=int(right_limit))
        return cls(space=space, current_state=current_state, field_function=field_function)

    # --- Factory used by ComposeField._spawn_field_like (duck-typed) ---
    def spawn(self, new_space: IntegerLine) -> "RealField":
        # create same kind of field on the new space; keep unit/zero via class-level singletons
        rf = RealField(space=new_space, current_state=new_space.get_state(), field_function=self.field_function)
        # reset storage if your Field base doesn’t do it in __init__
        if hasattr(rf, "set_empty_field"):
            rf.set_empty_field()
        if hasattr(rf, "set_zero_field"):
            rf.set_zero_field()
        return rf

    def plot_field(self):
        state_space_array = [s.state for s in self.state_space.get_all_states()]
        field_each_state = [self.get_field(s).data.value for s in self.state_space.get_all_states()]
        # print(state_space_array)
        # print(field_each_state)
        plt.scatter(state_space_array, field_each_state)
        plt.show()


# --------------- Real Field Addition --------------#

class RealFieldAddition(ComposeField):
    def __init__(self):
        composer=RealFieldValueAddition()
        return_field = RealField.from_length(1,IntegerState(0))
        zero_compose = RealFieldValue(RealSingleFieldValue(0))
        zero_field_1 = RealFieldValue(RealSingleFieldValue(0))
        zero_field_2 = RealFieldValue(RealSingleFieldValue(0))
        super().__init__(composer,return_field,zero_compose, zero_field_1, zero_field_2)

    def apply_legacy(self, f1: RealField, f2: RealField) -> Field[State | Any]:
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

                sfv_out = RealSingleFieldValue(a.data.get_data() + b.data.get_data())
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
            sfv_out = RealSingleFieldValue(a.data.get_data() + b.data.get_data())
            fv_out = FieldValue(sfv_out, unit.normTransform, unit.additionComposition)
            out_set(s, fv_out)

        return out

    def apply(self, f1: RealField, f2: RealField) -> Field[State | Any]:
        return super().apply(f1, f2)


# ----------- Real Field Multiplication -----------#
class RealFieldMultiplication(ComposeField):
    def __init__(self):
        composer = RealFieldValueMultiplication()
        return_field = RealField.from_length(1, IntegerState(0))
        zero_compose = RealFieldValue(RealSingleFieldValue(1))
        zero_field_1 = RealFieldValue(RealSingleFieldValue(1))
        zero_field_2 = RealFieldValue(RealSingleFieldValue(1))
        super().__init__(composer, return_field, zero_compose, zero_field_1, zero_field_2)

    def apply(self, f1: RealField, f2: RealField) -> Field[State | Any]:
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

                sfv_out = RealSingleFieldValue(a.data.get_data()*b.data.get_data())
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
            sfv_out = RealSingleFieldValue(a.data.get_data()*b.data.get_data())
            fv_out = FieldValue(sfv_out, unit.normTransform, unit.additionComposition)
            out_set(s, fv_out)

        return out

    def apply_legacy(self, f1: RealField, f2: RealField) ->Field[State | Any]:
        return super().apply(f1,f2)

#------------ State Less Field Addition -----------#

class RealFieldAdditionStateless(StatelessComposeField[S]):
    COMPOSER: ClassVar[Composition] = RealFieldValueAddition()
    ZERO_COMPOSE: ClassVar[Optional[FieldValue]] = RealFieldValue(RealSingleFieldValue(0))
    ZERO_F1: ClassVar[Optional[FieldValue]] = None
    ZERO_F2: ClassVar[Optional[FieldValue]] = None
    RETURN_FIELD_PROTO: ClassVar[Field[S]] = RealField.from_length(1, IntegerState(0))

    @classmethod
    def apply(cls, f1: Field[S], f2: Field[S]) -> Field[S]:
        return super().apply(f1, f2, return_field_proto=cls.RETURN_FIELD_PROTO)

# -------------- Real Field Norm Transform -------------#

class RealFieldNormTransform(TransformField[S]):
    def __init__(self):
        norm_transform= RealFieldValueNorm()
        super().__init__(norm_transform)

class RealFieldIdentityTransform(TransformField[S]):
    def __init__(self):
        super().__init__()

    def apply(self, f1: RealField, f2: RealField) -> RealField:
        return copy.deepcopy(f1)



