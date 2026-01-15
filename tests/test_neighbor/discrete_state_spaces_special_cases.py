import pytest
import jax.numpy as jnp
from typing import Any, Tuple

# --- Imports from your project structure ---
from src.field_dynamic_system.core.state.discrete import AbstractDiscreteStateSpace, VectorStateSpace, VectorState
from src.field_dynamic_system.neighbor.discrete import GraphTopology, DeltaTopology, MetricDiscreteTopology


# ==============================================================================
# 1. GRAPH TOPOLOGY TESTS (Explicit Edges)
# ==============================================================================

def test_graph_topology_directed():
    """
    Scenario: A -> B -> C. Directed.
    """
    # 1. Setup States
    states = ["CityA", "CityB", "CityC"]
    space = AbstractDiscreteStateSpace(states)

    # 2. Setup Edges
    edges = [("CityA", "CityB"), ("CityB", "CityC")]

    topo = GraphTopology(space, edges, directed=True)

    # 3. Test Connections
    # A should reach B
    succ_a = topo.successor("CityA")
    assert succ_a.contains("CityB")
    assert not succ_a.contains("CityC")  # Not directly connected

    # C should reach nothing (Dead end)
    succ_c = topo.successor("CityC")
    assert succ_c.num_states == 0


def test_graph_topology_undirected():
    """
    Scenario: A -- B. Undirected (Bi-directional).
    """
    states = ["A", "B"]
    space = AbstractDiscreteStateSpace(states)
    edges = [("A", "B")]

    topo = GraphTopology(space, edges, directed=False)

    # A -> B
    assert topo.successor("A").contains("B")
    # B -> A (Reverse automatically added)
    assert topo.successor("B").contains("A")


# ==============================================================================
# 2. DELTA TOPOLOGY TESTS (Grid / Relative Moves)
# ==============================================================================

@pytest.fixture
def grid_3x3():
    """
    Creates a 3x3 Grid of VectorStates: (0,0) to (2,2)
    """
    states = [VectorState((x, y)) for x in range(3) for y in range(3)]
    return VectorStateSpace(states, dim=2)


def test_delta_topology_cardinal_moves(grid_3x3):
    """
    Scenario: Robot can only move RIGHT (+1, 0) and UP (0, +1).
    """
    # Define Deltas (Right, Up)
    deltas = [VectorState((1.0, 0.0)), VectorState((0.0, 1.0))]

    topo = DeltaTopology(grid_3x3, deltas)

    # Case A: Center (1,1) -> Should reach (2,1) [Right] and (1,2) [Up]
    start = VectorState((1.0, 1.0))
    next_space = topo.successor(start)

    assert next_space.contains(VectorState((2.0, 1.0)))
    assert next_space.contains(VectorState((1.0, 2.0)))
    assert next_space.num_states == 2

    # Case B: Corner (2,2) -> Should reach NOWHERE (Out of bounds)
    corner = VectorState((2.0, 2.0))
    next_space_corner = topo.successor(corner)
    assert next_space_corner.num_states == 0


# ==============================================================================
# 3. METRIC TOPOLOGY TESTS (Distance & Rings)
# ==============================================================================

@pytest.fixture
def line_points():
    """
    Points on a 1D line: 0.0, 1.0, 2.0, 3.0, 4.0
    """
    states = [VectorState((x,)) for x in [0.0, 1.0, 2.0, 3.0, 4.0]]
    return VectorStateSpace(states, dim=1)


def test_metric_topology_standard(line_points):
    """
    Scenario: Connect everything within distance 1.5.
    (0) should connect to (1). (1) to (0, 2).
    """
    # Max Dist = 1.5
    topo = MetricDiscreteTopology(line_points, max_dist=1.5)

    # Check Neighbors of 1.0
    # Dist(1,0)=1.0 (OK), Dist(1,2)=1.0 (OK), Dist(1,3)=2.0 (Fail)
    succ = topo.successor(VectorState((1.0,)))

    assert succ.contains(VectorState((0.0,)))
    assert succ.contains(VectorState((2.0,)))
    assert not succ.contains(VectorState((3.0,)))


def test_metric_topology_ring(line_points):
    """
    Scenario: The "Donut" / Ring Topology.
    Connect states that are far, but not too far.
    Range: [1.5, 2.5]

    From 0.0:
    - 1.0 (Dist 1) -> Too Close (Fail)
    - 2.0 (Dist 2) -> Inside Ring (Pass)
    - 3.0 (Dist 3) -> Too Far (Fail)
    """
    topo = MetricDiscreteTopology(line_points, min_dist=1.5, max_dist=2.5)

    succ = topo.successor(VectorState((0.0,)))

    # Should ONLY contain 2.0
    assert succ.contains(VectorState((2.0,)))
    assert not succ.contains(VectorState((1.0,)))  # Too close
    assert not succ.contains(VectorState((3.0,)))  # Too far


def test_metric_topology_self_loops(line_points):
    """
    Scenario: Handling min_dist=0 (Self Loops).
    If min_dist > 0, a state should NOT be its own neighbor.
    """
    # Case A: Explicitly exclude self (min_dist > 0)
    topo_no_self = MetricDiscreteTopology(line_points, min_dist=0.1, max_dist=1.5)
    succ = topo_no_self.successor(VectorState((0.0,)))
    assert not succ.contains(VectorState((0.0,)))  # No self loop

    # Case B: Include self (min_dist = 0)
    # (Note: In strict neighbor definitions, self-loops are often excluded,
    # but mathematically d(x,x)=0 <= max_dist)
    topo_self = MetricDiscreteTopology(line_points, min_dist=0.0, max_dist=1.5)

    # Our implementation logic removes diagonal by default if min_dist > 0.
    # If min_dist=0, it keeps it. Let's verify behavior.
    succ_self = topo_self.successor(VectorState((0.0,)))
    assert succ_self.contains(VectorState((0.0,)))