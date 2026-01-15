import pytest
import jax.numpy as jnp
from src.field_dynamic_system.core.state.continous import HypersphereSpace, HypercubeSpace
from src.field_dynamic_system.neighbor.continuous import MetricTopology


# ==============================================================================
# 1. 1D TESTS (The Number Line)
# ==============================================================================

def test_1d_metric_topology_successor():
    """
    Scenario: 1D Line. Start at 10.0. Epsilon (step size) = 2.0.
    Successor should be range [8.0, 12.0].
    """
    # Universe is just a large box (conceptually the real line)
    universe = HypercubeSpace(low=jnp.array([-100.0]), high=jnp.array([100.0]))
    topology = MetricTopology(universe, epsilon=2.0)

    start_state = 10.0

    # --- Single Step ---
    next_space = topology.successor(start_state)

    assert isinstance(next_space, HypersphereSpace)
    assert next_space.dim == 1
    assert next_space.radius == 2.0
    assert next_space.center[0] == 10.0

    # Boundary Checks
    assert next_space.contains(12.0)  # Upper Bound
    assert next_space.contains(8.0)  # Lower Bound
    assert next_space.contains(10.0)  # Center
    assert not next_space.contains(12.1)  # Outside Upper
    assert not next_space.contains(7.9)  # Outside Lower


def test_1d_metric_topology_multistep():
    """
    Scenario: 1D Line. Start at 0.0. Epsilon = 1.5. Steps = 4.
    Expected Radius = 1.5 * 4 = 6.0.
    Range: [-6.0, 6.0]
    """
    universe = HypercubeSpace(low=jnp.array([-100.0]), high=jnp.array([100.0]))
    topology = MetricTopology(universe, epsilon=1.5)

    # --- Multi Step ---
    future_space = topology.multi_step_successor(0.0, steps=4)

    assert future_space.radius == 6.0

    # Verify bounds
    assert future_space.contains(6.0)
    assert future_space.contains(-6.0)
    assert future_space.contains(3.5)

    # Verify outside
    assert not future_space.contains(6.01)


# ==============================================================================
# 2. 5D TESTS (High Dimensional Hypersphere)
# ==============================================================================

def test_5d_metric_topology_successor():
    """
    Scenario: 5D Space. Start at Origin. Epsilon = 0.5.
    Successor is a 5D Ball with radius 0.5.
    """
    # Universe: 5D Box [-10, 10]
    universe = HypercubeSpace(low=jnp.full(5, -10.), high=jnp.full(5, 10.))
    topology = MetricTopology(universe, epsilon=0.5)

    start_state = jnp.zeros(5)  # [0, 0, 0, 0, 0]

    next_space = topology.successor(start_state)

    assert next_space.dim == 5
    assert next_space.radius == 0.5

    # Check point on axis 0 at distance 0.5
    p1 = jnp.array([0.5, 0., 0., 0., 0.])
    assert next_space.contains(p1)

    # Check point on axis 4 at distance 0.6 (Outside)
    p2 = jnp.array([0., 0., 0., 0., 0.6])
    assert not next_space.contains(p2)


def test_5d_metric_topology_multistep():
    """
    Scenario: 5D Space. Start at Origin. Epsilon = 1.0. Steps = 3.
    Expected Radius = 3.0.
    """
    universe = HypercubeSpace(low=jnp.full(5, -100.), high=jnp.full(5, 100.))
    topology = MetricTopology(universe, epsilon=1.0)

    future_space = topology.multi_step_successor(jnp.zeros(5), steps=3)

    assert future_space.radius == 3.0

    # 1. Test "Manhattan-ish" point that is actually inside Euclidean ball
    # Point [1, 1, 1, 1, 1]
    # Distance = sqrt(1^2 + ... + 1^2) = sqrt(5) ≈ 2.236
    # 2.236 < 3.0 -> Should be INSIDE
    mixed_point = jnp.ones(5)
    assert future_space.contains(mixed_point)

    # 2. Test point just outside
    # Point [3.0, 0.1, 0, 0, 0] -> Distance > 3.0
    outside_point = jnp.zeros(5).at[0].set(3.0).at[1].set(0.1)
    assert not future_space.contains(outside_point)

    # 3. Test Batch Query in 5D
    # [Inside, Outside]
    batch_points = jnp.array([
        [2.0, 0, 0, 0, 0],  # Dist 2
        [4.0, 0, 0, 0, 0]  # Dist 4
    ])
    results = future_space.contains(batch_points)
    assert jnp.array_equal(results, jnp.array([True, False]))