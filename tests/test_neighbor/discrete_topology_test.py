import pytest
import jax.numpy as jnp
from typing import Any, List

# Import core components
from src.field_dynamic_system.core.state.discrete import AbstractDiscreteStateSpace
from src.field_dynamic_system.neighbor.discrete import DiscreteTopology




# --- 1. DEFINE THE USER'S CUSTOM TOPOLOGY ---
class ForwardSkipTopology(DiscreteTopology):
    """
    User Logic:
    From State i, you can go to:
    1. i + 1 (Step)
    2. i + 2 (Skip)
    Wrapping around modulus N.
    """

    def __init__(self, space, modulus: int):
        super().__init__(space)
        self.modulus = modulus

    def compute_neighbors(self, state: int) -> List[int]:
        next_val = (state + 1) % self.modulus
        skip_val = (state + 2) % self.modulus
        return [next_val, skip_val]


# --- 2. DEFINE THE TEST SUITE ---

@pytest.fixture
def branching_system():
    # 15 States (0-14)
    states = list(range(15))
    space = AbstractDiscreteStateSpace(states)
    topology = ForwardSkipTopology(space, modulus=15)
    return space, topology


def test_single_step_branching(branching_system):
    """
    Test 1 step: 0 -> {1, 2}
    """
    space, topology = branching_system

    # Check neighbors of 0
    next_space = topology.successor(0)

    # Should have exactly 2 states
    assert next_space.num_states == 2
    assert next_space.contains(1)
    assert next_space.contains(2)


def test_multi_step_expansion(branching_system):
    """
    Test 2 steps: Expansion Logic.

    Step 0: {0}
    Step 1: {1, 2}  (Neighbors of 0)
    Step 2: Neighbors({1, 2})
          = Neighbors(1) U Neighbors(2)
          = {2, 3}       U {3, 4}
          = {2, 3, 4}
    """
    space, topology = branching_system

    future_space = topology.multi_step_successor(0, steps=2)

    # Expect 3 unique states
    assert future_space.num_states == 3
    assert future_space.contains([2, 3, 4]).all()
    assert not future_space.contains(1)  # 1 is left behind
    assert not future_space.contains(5)  # 5 is too far


def test_wave_propagation(branching_system):
    """
    Test 5 steps.
    Pattern: Step N reaches range [N, 2N] (modulo 15).

    Step 5 Expectation:
    Min Reach: 0 + 1*5 = 5
    Max Reach: 0 + 2*5 = 10
    Set: {5, 6, 7, 8, 9, 10}
    """
    space, topology = branching_system

    future_space = topology.multi_step_successor(0, steps=5)

    # Size should be 6 (5 to 10 inclusive)
    assert future_space.num_states == 6

    # Verify bounds
    expected_states = [5, 6, 7, 8, 9, 10]
    assert future_space.contains(expected_states).all()

    # Verify outside
    assert not future_space.contains(4)
    assert not future_space.contains(11)


def test_full_saturation(branching_system):
    """
    Test large steps.
    Eventually, the "fast" wave wraps around and catches the "slow" wave,
    filling the entire ring.
    """
    space, topology = branching_system

    # After ~15 steps, the range is [15, 30].
    # Modulo 15, this covers [0, 0] -> Everything.
    future_space = topology.multi_step_successor(0, steps=14)

    # Should fill the entire space
    assert future_space.num_states == 15