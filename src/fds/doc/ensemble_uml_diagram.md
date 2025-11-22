```mermaid
classDiagram
    class FieldDynamicSystem {
      <<system>>
    }

    class AffectingGroupsEvolution {
      <<affecting>>
      + evolve()
      + groups
    }

    class EnsembleOperator {
      <<ensemble>>
      + channels : List[AffectingGroupsEvolution]
      + stepAll()
      + collectResults()
    }

    class Ensemble {
      <<ensemble>>
      + systems : List[FieldDynamicSystem]
      + channels : List[AffectingGroupsEvolution]
      + operator : EnsembleOperator
      + run()
    }

    %% Relationships
    Ensemble "1" --> "1..*" FieldDynamicSystem : uses
    Ensemble "1" --> "1..*" AffectingGroupsEvolution : channels
    Ensemble "1" --> "1" EnsembleOperator : controls

    EnsembleOperator "1" --> "1..*" AffectingGroupsEvolution : operates on
