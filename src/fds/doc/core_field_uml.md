```mermaid

classDiagram
    %% ======================
    %% FIELD MODULE (CORE)
    %% ======================

    class State {
      %% From state module
    }

    class StateSpace {
      %% From state module
    }

    class SingleFieldValue {
      <<core>>
      %% Atomic field value (real, complex, vector, etc.)
    }

    class SingleFieldValueComposition {
      <<core>>
      %% Internal composition
      +compose(a: SingleFieldValue, b: SingleFieldValue) SingleFieldValue
    }

    class SingleFieldValueTransform {
      <<core>>
      %% Transform on a single value
      +apply(v: SingleFieldValue) SingleFieldValue
    }

    class AdditionSingleFieldValue {
      <<concrete>>
      %% External accumulation (plus)
      +add(a: SingleFieldValue, b: SingleFieldValue) SingleFieldValue
    }

    class NormTransform {
      <<concrete>>
      %% Norm / contraction to R
      +norm(v: SingleFieldValue) float
    }

    class FieldValue {
      <<core>>
      %% Wraps value + operations
      -value: SingleFieldValue
    }

    class FieldFunction {
      <<optional>>
      %% Field as function on state space
      +__call__(s: State) FieldValue
    }

    class Field {
      <<core>>
      %% Field over a state space
      +get(s: State) FieldValue
      +supported_states() Set[State]
    }

    class FieldTransform {
      <<core>>
      %% Field -> Field on same space
      +apply(F: Field) Field
    }

    class FieldComposition {
      <<core>>
      %% Composition of two fields
      +compose(F1: Field, F2: Field) Field
    }

    %% Relationships

    AdditionSingleFieldValue --|> SingleFieldValueComposition
    NormTransform --|> SingleFieldValueTransform

    FieldValue --> SingleFieldValue

    Field "1" --> "1" StateSpace
    Field --> FieldValue

    FieldFunction --> State
    FieldFunction --> FieldValue

    FieldTransform --> Field
    FieldComposition --> Field
