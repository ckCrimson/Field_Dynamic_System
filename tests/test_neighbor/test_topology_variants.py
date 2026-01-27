import pytest
import jax.numpy as jnp
from jax import config
import numpy as np

config.update("jax_enable_x64", True)

from src.field_dynamic_system.core import VectorState
from src.field_dynamic_system.core.state.discrete import VectorStateSpace, AbstractDiscreteStateSpace
from src.field_dynamic_system.neighbor.discrete import GraphTopology, DeltaTopology


def test_graph_topology():
    print("\n=== TEST 1: GraphTopology (Explicit Edges) ===")

    states = ["A", "B", "C", "D"]
    space = AbstractDiscreteStateSpace(states)

    edges = [
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "A"),
        ("A", "C")
    ]

    topo = GraphTopology(space, edges)

    # 3. Verify High-Level Logic
    neighbors_A = topo.compute_neighbors("A")
    assert set(neighbors_A) == {"B", "C"}

    # 4. Verify Raw Successor (Discovery Track)
    # FIX: Pass RAW STATE "A", not index 0.
    raw_start = ["A"]
    raw_succ = topo.get_raw_successor(raw_start)

    print(f"  Raw Successor of A: {raw_succ}")
    assert set(raw_succ) == {"B", "C"}

    # 5. Verify Multi-Step (Raw)
    raw_2step = topo.get_raw_multi_step_successor(raw_start, 2)
    print(f"  2 Steps from A: {raw_2step}")
    assert set(raw_2step) == {"C", "D"}

    print("  ✅ GraphTopology Passed")


def test_delta_topology():
    print("\n=== TEST 2: DeltaTopology (Grid Deltas) ===")

    states = []
    for x in range(3):
        for y in range(3):
            states.append(VectorState((float(x), float(y))))

    space = VectorStateSpace(states, dim=2)

    deltas = [
        VectorState((1.0, 0.0)),
        VectorState((0.0, 1.0))
    ]

    topo = DeltaTopology(space, deltas)

    # 3. Verify Boundary Logic
    corner_state = VectorState((2.0, 2.0))
    neighbors_corner = topo.compute_neighbors(corner_state)
    print(f"  Neighbors of Corner (2,2): {neighbors_corner}")
    assert len(neighbors_corner) == 0

    # 4. Verify Center Logic
    center_state = VectorState((1.0, 1.0))
    neighbors_center = topo.compute_neighbors(center_state)

    # FIX: Handle both raw tuples (optimized return) and VectorState objects
    n_tuples = set()
    for s in neighbors_center:
        if hasattr(s, 'values'):
            n_tuples.add(tuple(s.values))
        else:
            n_tuples.add(tuple(s))

    expected = {(2.0, 1.0), (1.0, 2.0)}
    print(f"  Neighbors of Center (1,1): {n_tuples}")

    # Fuzzy compare for floats
    assert len(n_tuples) == len(expected)
    for t in n_tuples:
        assert any(np.allclose(t, e) for e in expected)

    print("  ✅ DeltaTopology Passed")


if __name__ == "__main__":
    test_graph_topology()
    test_delta_topology()