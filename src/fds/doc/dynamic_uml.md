classDiagram
    %% ======================
    %% CORE REFERENCES
    %% ======================
    class State
    class StateSpace
    class Reachable
    class Field
    class FieldTransform
    class FieldComposition

    %% ======================
    %% SINGLE STEP MODULE
    %% ======================
    class Kernel {
      <<core>>
      + eval(s : State, s' : State) : FieldValue
    }

    class HookingBias {
      <<optional>>
      + bias(F_global : Field, kernel_value : FieldValue) : FieldValue
    }

    class KernelFieldComposition {
      <<optional>>
      + compose(kernelField : Field, globalField : Field) : Field
    }

    class SingleStepField {
      <<core>>
      + single_step(s : State) : Field
    }

    Kernel --> Field
    Kernel --> State
    Kernel --> StateSpace

    HookingBias --> Field
    HookingBias --> FieldValue

    KernelFieldComposition --> Field

    SingleStepField --> Kernel
    SingleStepField --> Reachable
    SingleStepField --> HookingBias

    %% ======================
    %% MULTI STEP MODULE
    %% ======================
    class MultiStepField {
      <<core>>
      + compute(s0 : State, l : int) : Field
    }

    MultiStepField --> Reachable
    MultiStepField --> SingleStepField
    MultiStepField --> FieldComposition
    MultiStepField --> FieldTransform
