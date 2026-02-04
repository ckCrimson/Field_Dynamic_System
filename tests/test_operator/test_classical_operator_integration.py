import pytest
from dataclasses import dataclass

from src.field_dynamic_system.core.state import VectorState
from src.field_dynamic_system.operator.base import InteractionContext
from src.field_dynamic_system.operator.classical import ClassicalOperator


# --- 1. REAL VECTOR STATE (From your requirement) ---

# --- 2. TRANSITION FUNCTION (Pure Logic) ---
def mario_physics(state: VectorState, action_id: int) -> VectorState:
    # 0=Stay, 1=Up, 2=Down, 3=Left, 4=Right
    deltas = {
        0: (0, 0), 1: (0, 1), 2: (0, -1), 3: (-1, 0), 4: (1, 0)
    }
    move = VectorState(deltas.get(action_id, (0, 0)))
    return state + move


# --- 3. TEST SUITE ---
class TestClassicalOperator:

    def test_passive_observation(self):
        """TC1: If no logic provided, it just returns the input state."""
        # Setup
        state = VectorState((10, 10))
        op = ClassicalOperator()  # No transition function
        ctx = InteractionContext(action_id=1)

        # Execute
        observation = op.observe(state, ctx)

        # Verify (Identity)
        assert observation == state
        assert observation.values == (10, 10)

    def test_active_trajectory_generation(self):
        """TC2: If logic provided, it calculates the next step (Trajectory)."""
        # Setup
        start_state = VectorState((0, 0))
        op = ClassicalOperator(transition_fn=mario_physics)

        # Execute: Action 1 (Up)
        ctx = InteractionContext(action_id=1)
        next_state = op.observe(start_state, ctx)

        # Verify Result
        assert next_state.values == (0, 1)
        # Verify Immutability (Input didn't change)
        assert start_state.values == (0, 0)

    def test_game_loop_orchestration(self):
        """
        TC3: Verify the loop logic.
        The Operator calculates, the Loop updates.
        """
        # Initialize
        current_state = VectorState((0, 0))
        op = ClassicalOperator(transition_fn=mario_physics)

        inputs = [4, 4, 1]  # Right, Right, Up
        history = [current_state.values]

        print("\n--- Trajectory Generation ---")

        for action in inputs:
            ctx = InteractionContext(action_id=action)

            # 1. Operator gives the Trajectory (Next State)
            # It does NOT update 'current_state' automatically.
            next_state = op.observe(current_state, ctx)

            # 2. The Loop decides to accept the transition
            current_state = next_state

            history.append(current_state.values)
            print(f"Input {action} -> {current_state.values}")

        # Final Path: (0,0) -> (1,0) -> (2,0) -> (2,1)
        assert history == [(0, 0), (1, 0), (2, 0), (2, 1)]