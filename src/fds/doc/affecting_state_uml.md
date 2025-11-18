classDiagram
    class State
    class StateSpace
    class StateSpaceMapping
    class Reachable
    class Reaching
    class MultiStepReaching
    class FieldDynamicSystem

    class AffectingGroupState {
      <<optional>>
    }

    class AffectingGroupStateSpace {
      <<optional>>
    }

    class CorrelatedState {
      <<core>>
      + state_map : Map[SystemId, State]
    }

    class CorrelatedStateSpace {
      <<core>>
    }

    class AffectedSystemsMapping {
      <<core>>
      + map(corr : CorrelatedState) : AffectingGroupState
    }

    AffectingGroupState --|> State
    AffectingGroupStateSpace --|> StateSpace
    CorrelatedState --|> State
    CorrelatedStateSpace --|> StateSpace

    AffectedSystemsMapping --|> StateSpaceMapping
    AffectedSystemsMapping --> CorrelatedStateSpace
    AffectedSystemsMapping --> AffectingGroupStateSpace
