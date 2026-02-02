import pytest
import time
from dataclasses import dataclass

# --- IMPORTS ---
# Adjust these if your local path structure is slightly different
from src.field_dynamic_system.core import VectorState
from src.field_dynamic_system.operator.base import InteractionContext
from src.field_dynamic_system.operator.classical import ClassicalOperator


# --- 1. SYSTEM SETUP (Mocks the Orchestrator) ---

class SystemContainer:
    """
    The Mutable Box acting as the Orchestrator's memory.
    The Operator receives this to 'look inside'.
    """

    def __init__(self, initial_state: VectorState):
        self.value = initial_state


# --- 2. PHYSICS ENGINE (Separate from Operator) ---

def apply_physics(container: SystemContainer, context: InteractionContext):
    """
    The Physics Engine / Transition Function.

    Responsibility:
    1. Read Context (User Input).
    2. Calculate Delta.
    3. Mutate the System State.

    The Operator is NOT involved here.
    """
    current_state = container.value
    action_id = context.action_id

    # Logic: 0=Stay, 1=Up, 2=Down, 3=Left, 4=Right
    # We use raw tuples here assuming VectorState supports initialization from tuples
    delta_vals = (0, 0)
    if action_id == 1:
        delta_vals = (0, 1)
    elif action_id == 2:
        delta_vals = (0, -1)
    elif action_id == 3:
        delta_vals = (-1, 0)
    elif action_id == 4:
        delta_vals = (1, 0)

    # Create the Delta VectorState
    delta = VectorState(delta_vals)

    # Update the container in-place (Physics Step)
    # Assumes VectorState supports __add__
    container.value = current_state + delta


# --- 3. THE STRICT TEST SUITE ---

class TestStrictClassicalWorkflow:

    def setup_method(self):
        # 1. Init System at (0,0)
        start_state = VectorState((0, 0))
        self.system = SystemContainer(start_state)

        # 2. Init Operator (NO ARGUMENTS - Pure Observer)
        self.operator = ClassicalOperator()

    def test_workflow_separation(self):
        """
        TC1: Verifies the cycle: Input -> Physics(Update) -> Operator(Observe).
        """
        # A. Create Context (User presses UP)
        ctx = InteractionContext(action_id=1)

        # B. Physics Step (The System updates the state)
        # This confirms logic happens OUTSIDE the operator
        apply_physics(self.system, ctx)

        # C. Observation Step (The Operator reads the state)
        # The operator receives context per contract, but ignores it for physics
        observed_view = self.operator.observe(self.system, ctx)

        # Verify
        assert isinstance(observed_view, VectorState)
        assert observed_view.values == (0, 1)

    def test_trajectory_loop(self):
        """
        TC2: Simulates the Game Loop.
        Sequence: Right -> Right -> Up -> Left
        Expected Final: (1, 1)
        """
        actions = [4, 4, 1, 3]
        history = []

        print("\n--- Game Loop (Strict Architecture) ---")

        for i, action in enumerate(actions):
            # 1. Capture Input
            ctx = InteractionContext(action_id=action)

            # 2. Physics Update (Orchestrator Job)
            apply_physics(self.system, ctx)

            # 3. Render/Observe (Operator Job)
            view = self.operator.observe(self.system, ctx)
            history.append(view.values)

            print(f"Frame {i}: Input {action} -> Rendered {view}")

        # Final Verification
        # (0,0) -> (1,0) -> (2,0) -> (2,1) -> (1,1)
        assert history == [(1, 0), (2, 0), (2, 1), (1, 1)]

    def test_operator_latency(self):
        """
        TC3: Benchmark the Operator's 'Read' speed.
        Since it is just a pass-through, this should be sub-microsecond.
        """
        ctx = InteractionContext()

        start = time.perf_counter()
        for _ in range(100_000):
            # Pure observation, no physics overhead
            _ = self.operator.observe(self.system, ctx)
        end = time.perf_counter()

        avg_ns = ((end - start) * 1e9) / 100_000
        print(f"\nOperator Overhead: {avg_ns:.2f} nanoseconds")

        # Constraint: < 500ns (0.5 microseconds)
        assert avg_ns < 500, "Operator overhead is unexpectedly high!"