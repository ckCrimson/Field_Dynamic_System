import pytest
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from src.field_dynamic_system.core import VectorState
from src.field_dynamic_system.core.state.discrete import VectorStateSpace, AbstractDiscreteStateSpace
from src.field_dynamic_system.neighbor.discrete import GraphTopology, DeltaTopology, MetricDiscreteTopology


# =========================================================================
# TEST 1: GraphTopology (Explicit Edges)
# Use Case: A directed cyclic graph 0->1->2->3->0
# =========================================================================
def test_graph_topology():
    print("\n=== TEST 1: GraphTopology (Explicit Edges) ===")

    # 1. Setup Space (4 Abstract States)
    states = ["A", "B", "C", "D"]
    space = AbstractDiscreteStateSpace(states)

    # 2. Define Edges (Directed Cycle + Shortcut A->C)
    edges = [
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "A"),
        ("A", "C")  # Shortcut
    ]

    topo = GraphTopology(space, edges, directed=True)

    # 3. Verify High-Level Logic (compute_neighbors)
    neighbors_A = topo.compute_neighbors("A")
    # Expected: B and C
    print(f"  Neighbors of A: {neighbors_A}")
    assert set(neighbors_A) == {"B", "C"}

    neighbors_C = topo.compute_neighbors("C")
    # Expected: D
    assert set(neighbors_C) == {"D"}

    # 4. Verify Low-Level Matrix (get_raw_successor)
    # A is index 0. Should reach indices for B(1) and C(2).
    raw_start = jnp.array([0])  # Index of A
    raw_succ = topo.get_raw_successor(raw_start)

    print(f"  Raw Successor indices of A(0): {raw_succ}")

    # Note: JAX returns indices sorted usually
    expected_indices = jnp.array([1, 2])
    assert jnp.array_equal(jnp.sort(raw_succ), expected_indices)

    # 5. Verify Multi-Step (Raw)
    # A -> (B, C) -> (C, D)
    # So 2 steps from A should reach {C, D} -> Indices {2, 3}
    raw_2step = topo.get_raw_multi_step_successor(raw_start, 2)
    print(f"  2 Steps from A(0): {raw_2step}")
    assert jnp.array_equal(jnp.sort(raw_2step), jnp.array([2, 3]))

    print("  ✅ GraphTopology Passed")


# =========================================================================
# TEST 2: DeltaTopology (Grid Movement)
# Use Case: 3x3 Grid, Moves: Up (0,1) and Right (1,0)
# =========================================================================
def test_delta_topology():
    print("\n=== TEST 2: DeltaTopology (Grid Deltas) ===")

    # 1. Setup 3x3 Grid (0,0 to 2,2)
    states = []
    for x in range(3):
        for y in range(3):
            states.append(VectorState((float(x), float(y))))

    # Note: VectorStateSpace order is preserved. Index 0=(0,0), Index 8=(2,2)
    space = VectorStateSpace(states, dim=2)

    # 2. Define Deltas (Up, Right)
    # Move +X and +Y
    deltas = [
        VectorState((1.0, 0.0)),  # Right
        VectorState((0.0, 1.0))  # Up
    ]

    topo = DeltaTopology(space, deltas)

    # 3. Verify Boundary Logic (Corner 2,2)
    # (2,2) + Right = (3,2) -> Invalid (Not in space)
    # (2,2) + Up    = (2,3) -> Invalid
    # Neighbors should be empty
    corner_state = VectorState((2.0, 2.0))
    neighbors_corner = topo.compute_neighbors(corner_state)
    print(f"  Neighbors of Corner (2,2): {neighbors_corner}")
    assert len(neighbors_corner) == 0

    # 4. Verify Center Logic (1,1)
    # (1,1) -> (2,1) [Valid] and (1,2) [Valid]
    center_state = VectorState((1.0, 1.0))
    neighbors_center = topo.compute_neighbors(center_state)

    # Convert to tuples for set comparison
    n_tuples = {tuple(s.values) for s in neighbors_center}
    expected = {(2.0, 1.0), (1.0, 2.0)}

    print(f"  Neighbors of Center (1,1): {n_tuples}")
    assert n_tuples == expected

    # 5. Verify Sparse Matrix Construction
    # Total states = 9.
    # (0,0) index 0 -> (1,0) index 3, (0,1) index 1.
    mat = topo.adjacency_matrix
    print(f"  Matrix Shape: {mat.shape} (Sparse BCOO)")

    # Verify raw successor of Index 0 ((0,0))
    raw_succ = topo.get_raw_successor(jnp.array([0]))
    # Neighbors of (0,0) are (0,1)[Index 1] and (1,0)[Index 3] (Assuming row-major generation)
    # Let's check indices in the space:
    idx_01 = space.get_index_of(VectorState((0.0, 1.0)))
    idx_10 = space.get_index_of(VectorState((1.0, 0.0)))

    print(f"  Raw Successor of (0,0): {raw_succ}")
    assert set(raw_succ.tolist()) == {idx_01, idx_10}

    print("  ✅ DeltaTopology Passed")


# =========================================================================
# TEST 3: MetricDiscreteTopology (Distance Threshold)
# Use Case: Points on a line. Connect if distance <= 1.5
# =========================================================================
def test_metric_topology():
    print("\n=== TEST 3: MetricDiscreteTopology (Distance) ===")

    # 1. Setup Points: 0.0, 1.0, 2.0, 5.0
    # 0,1,2 are chained. 5 is isolated.
    points = [0.0, 1.0, 2.0, 5.0]
    states = [VectorState((p,)) for p in points]
    space = VectorStateSpace(states, dim=1)

    # 2. Define Metric: Max Dist = 1.5, Min Dist = 0.1 (exclude self)
    topo = MetricDiscreteTopology(space, max_dist=1.5, min_dist=0.1)

    # 3. Check Connectivity
    # State 1.0 should connect to 0.0 (dist 1) and 2.0 (dist 1)
    # It should NOT connect to 5.0 (dist 4)
    neighbors_1 = topo.compute_neighbors(VectorState((1.0,)))
    vals_1 = sorted([s.values[0] for s in neighbors_1])

    print(f"  Neighbors of 1.0: {vals_1}")
    assert vals_1 == [0.0, 2.0]

    # 4. Check Isolation
    neighbors_5 = topo.compute_neighbors(VectorState((5.0,)))
    print(f"  Neighbors of 5.0: {neighbors_5}")
    assert len(neighbors_5) == 0

    # 5. Verify Matrix
    # Index 3 is state 5.0. Raw successor should be empty.
    raw_succ_5 = topo.get_raw_successor(jnp.array([3]))
    assert raw_succ_5.size == 0

    print("  ✅ MetricDiscreteTopology Passed")


if __name__ == "__main__":
    test_graph_topology()
    test_delta_topology()
    test_metric_topology()