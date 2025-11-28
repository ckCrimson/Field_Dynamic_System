classDiagram
    %% ======================
    %% STATE MODULE (CORE)
    %% ======================

    class State {
      <<core>>
      %% Represents a single configuration
    }

    class StateSpace {
      <<core>>
      %% Set/type of all allowed states
      +contains(s: State) bool
      +neighbors(s: State) Set[State]
    }

    class StateSpaceMapping {
      <<optional>>
      %% Mapping between two state spaces
      +map(s: State) Set[State]
    }

    class StateOperations {
      <<optional>>
      %% Binary operations on states
      +op(a: State, b: State) State
    }

    class Reachable {
      <<core>>
      %% One-step reachable
      +V(s: State) Set[State]
    }

    class Reaching {
      <<optional>>
      %% Inverse of Reachable
      +R(s: State) Set[State]
    }

    class MultiStepReaching {
      <<core>>
      %% l-step reachable sets
      +L(s0: State, l: int) Set[State]
    }

    %% Relationships

    StateSpace "1" o-- "*" State

    Reachable --> State
    Reachable --> StateSpace

    Reaching --> State
    Reaching --> StateSpace

    MultiStepReaching --> Reachable
    MultiStepReaching --> State

    StateSpaceMapping --> StateSpace : source/target
    StateOperations --> State
