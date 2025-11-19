```mermaid

classDiagram
    %% ======================
    %% STATE MODULE
    %% ======================
    class State {
      <<core>>
      + configuration
    }

    class StateSpace {
      <<core>>
      + states : Set[State]
    }

    class StateSpaceMapping {
      <<optional>>
      + map(s : State) : Set[State]
    }

    class StateOperations {
      <<optional>>
      + op(a : State, b : State) : State
    }

    class Reachable {
      <<core>>
      + reachable_from(s0 : State, l : int) : StateSpace
    }

    class Reaching {
      <<optional>>
      + reaching_of(s : State) : Set[State]
    }

    class MultiStepReaching {
      <<core>>
      + reachable_l_steps(s0 : State, l : int) : StateSpace
    }

    StateSpace "1" o-- "*" State
    StateSpaceMapping --> StateSpace : source/target
    StateOperations --> State
    Reachable --> State
    Reachable --> StateSpace
    Reaching --> State
    Reaching --> StateSpace
    MultiStepReaching --> Reachable
    MultiStepReaching --> State
    MultiStepReaching --> StateSpace

    %% ======================
    %% FIELD MODULE
    %% ======================

    class SingleFieldValue {
      <<core>>
    }

    class SingleFieldValueComposition {
      <<core>>
      + compose(a : SingleFieldValue, b : SingleFieldValue) : SingleFieldValue
    }

    class SingleFieldValueTransform {
      <<core>>
      + transform(v : SingleFieldValue) : SingleFieldValue
    }

    class AdditionSingleFieldValue {
      <<core>>
      + add(a : SingleFieldValue, b : SingleFieldValue) : SingleFieldValue
    }

    class NormTransform {
      <<core>>
      + norm(v : SingleFieldValue) : float
    }

    class FieldValue {
      <<core>>
      + value : SingleFieldValue
      + add(...)
      + norm(...)
    }

    class FieldFunction {
      <<optional>>
      + f(s : State) : FieldValue
    }

    class Field {
      <<core>>
      + space : StateSpace
      + get(s : State) : FieldValue
    }

    class FieldTransform {
      <<core>>
      + transform(F : Field) : Field
    }

    class FieldComposition {
      <<core>>
      + compose(F1 : Field, F2 : Field) : Field
    }

    SingleFieldValueComposition --> SingleFieldValue
    SingleFieldValueTransform --> SingleFieldValue
    AdditionSingleFieldValue --|> SingleFieldValueComposition
    NormTransform --|> SingleFieldValueTransform

    FieldValue --> SingleFieldValue
    FieldValue --> AdditionSingleFieldValue
    FieldValue --> NormTransform

    FieldFunction --> State
    FieldFunction --> FieldValue

    Field --> FieldValue
    Field --> StateSpace
    FieldTransform --> Field
    FieldTransform --> SingleFieldValueTransform
    FieldComposition --> Field
    FieldComposition --> SingleFieldValueComposition
