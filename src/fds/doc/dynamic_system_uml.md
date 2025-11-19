```mermaid

classDiagram
    class State
    class StateSpace
    class Reachable
    class Reaching
    class MultiStepReaching
    class Field
    class MultiStepField
    class Operator

    class StaticDynamicSystem {
      <<optional>>
      + S : StateSpace
      + reachable : Reachable
      + reaching : Reaching [optional]
      + multistep_reaching : MultiStepReaching
    }

    class FieldStaticDynamicSystem {
      <<optional>>
      + field : Field
    }

    class FieldDynamicSystem {
      <<core>>
      + multistep_field : MultiStepField
      + operator : Operator
    }

    FieldStaticDynamicSystem --|> StaticDynamicSystem
    FieldDynamicSystem --|> FieldStaticDynamicSystem

    StaticDynamicSystem --> State
    StaticDynamicSystem --> StateSpace
    StaticDynamicSystem --> Reachable
    StaticDynamicSystem --> Reaching
    StaticDynamicSystem --> MultiStepReaching

    FieldStaticDynamicSystem --> Field
    FieldDynamicSystem --> MultiStepField
    FieldDynamicSystem --> Operator
