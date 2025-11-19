```mermaid

classDiagram
    class FieldDynamicSystem
    class AffectingGroupState
    class AffectingGroupStateSpace
    class AffectedSystemsMapping
    class Reachable
    class Reaching
    class MultiStepReaching
    class SingleStepField
    class MultiStepField
    class Operator

    %% Grouping module
    class IsAffecting {
      <<core>>
      + affects(A : FieldDynamicSystem, B : FieldDynamicSystem) : bool
    }

    class InterAffectingGroups {
      <<core>>
      + groups : List[List[FieldDynamicSystem]]
    }

    InterAffectingGroups --> IsAffecting
    InterAffectingGroups --> AffectedSystemsMapping

    %% Affecting reachable
    class AffectedReachable {
      <<core>>
    }

    class AffectedReaching {
      <<optional>>
    }

    class AffectingMultiStepReaching {
      <<core>>
    }

    AffectedReachable --|> Reachable
    AffectedReachable --> AffectingGroupStateSpace
    AffectedReachable --> AffectingGroupState

    AffectedReaching --|> Reaching
    AffectedReaching --> AffectingGroupStateSpace
    AffectedReaching --> AffectingGroupState

    AffectingMultiStepReaching --|> MultiStepReaching
    AffectingMultiStepReaching --> AffectedReachable
    AffectingMultiStepReaching --> AffectingGroupState

    %% Affecting dynamics
    class AffectingSingleStep {
      <<core>>
    }

    class AffectingMultiStepField {
      <<core>>
    }

    class AffectedSystemsOperator {
      <<core>>
    }

    AffectingSingleStep --|> SingleStepField
    AffectingMultiStepField --|> MultiStepField
    AffectedSystemsOperator --|> Operator

    %% Affecting FDS
    class AffectingFDS {
      <<core>>
    }

    AffectingFDS --|> FieldDynamicSystem
    AffectingFDS --> AffectingGroupState
    AffectingFDS --> AffectingGroupStateSpace
    AffectingFDS --> AffectedSystemsMapping
    AffectingFDS --> AffectedReachable
    AffectingFDS --> AffectedReaching
    AffectingFDS --> AffectingMultiStepReaching
    AffectingFDS --> AffectingMultiStepField
    AffectingFDS --> AffectedSystemsOperator

    %% Group evolution
    class AffectingGroupsEvolution {
      <<core>>
      + evolve(systems : List[FieldDynamicSystem])
    }

    AffectingGroupsEvolution --> InterAffectingGroups
    AffectingGroupsEvolution --> AffectingFDS
