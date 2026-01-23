import pytest
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from src.field_dynamic_system.core.state.discrete import VectorStateSpace
from src.field_dynamic_system.core import VectorState
from src.field_dynamic_system.neighbor.discrete import DiscreteTopology


# Import your topologies here...
# For this test, we mock a simple one or assume DeltaTopology exists
class LinearTopology(DiscreteTopology):
    # Simple line: i -> i+1
    def compute_neighbors(self, state):
        v = state.values[0]
        target = VectorState((v + 1.0,))
        return [target]


def test_raw_kernels():
    print("\n=== TEST: Raw Topology Kernels ===")

    # 1. Setup Space: [0, 1, 2, 3, 4]
    states = [VectorState((float(i),)) for i in range(5)]
    space = VectorStateSpace(states, dim=1)

    topo = LinearTopology(space)

    # 2. Test Successor (Raw)
    # Start at Index 1 (State "1.0")
    start_idx = jnp.array([1])
    succ_idx = topo.get_raw_successor(start_idx)

    # Expect: Index 2 (State "2.0")
    print(f"  Successor of 1: {succ_idx}")
    assert jnp.array_equal(succ_idx, jnp.array([2]))

    # 3. Test Multi-Step (Raw)
    # Start at Index 0, 3 steps -> Should reach Index 3
    multi_idx = topo.get_raw_multi_step_successor(jnp.array([0]), 3)

    print(f"  0 + 3 Steps: {multi_idx}")
    assert jnp.array_equal(multi_idx, jnp.array([3]))

    # 4. Test Predecessor (Raw)
    # Who can reach Index 4? Should be Index 3.
    pred_idx = topo.get_raw_predecessor(jnp.array([4]))

    print(f"  Predecessor of 4: {pred_idx}")
    assert jnp.array_equal(pred_idx, jnp.array([3]))

    print("✅ All Raw Kernels passed.")