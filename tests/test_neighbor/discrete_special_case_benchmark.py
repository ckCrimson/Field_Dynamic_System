import time
import pytest
import jax.numpy as jnp
import random
from typing import List

from src.field_dynamic_system.core.state.discrete import AbstractDiscreteStateSpace, VectorStateSpace, VectorState
from src.field_dynamic_system.neighbor.discrete import DiscreteTopology
from src.field_dynamic_system.neighbor.discrete import GraphTopology, DeltaTopology, MetricDiscreteTopology


# ==============================================================================
# 0. BASELINE IMPLEMENTATIONS (To simulate "Generic User Logic")
# ==============================================================================

class GenericGraphTopo(DiscreteTopology):
    def __init__(self, space, edges, directed=True):
        super().__init__(space)
        # Convert edge list to dict for slightly faster lookups than pure list
        self.adj = {}
        for u, v in edges:
            if u not in self.adj: self.adj[u] = []
            self.adj[u].append(v)
            if not directed:
                if v not in self.adj: self.adj[v] = []
                self.adj[v].append(u)

    def compute_neighbors(self, state):
        return self.adj.get(state, [])


class GenericDeltaTopo(DiscreteTopology):
    def __init__(self, space, deltas):
        super().__init__(space)
        self.deltas = deltas

    def compute_neighbors(self, state):
        neighbors = []
        for d in self.deltas:
            # Try/Catch overhead + Python loop overhead
            try:
                candidate = state + d
                if self.discrete_space.get_index_of(candidate) != -1:
                    neighbors.append(candidate)
            except:
                pass
        return neighbors


class GenericMetricTopo(DiscreteTopology):
    def __init__(self, space, max_dist):
        super().__init__(space)
        self.max_dist = max_dist

    def compute_neighbors(self, state):
        # The classic O(N) loop per state -> O(N^2) total
        nbrs = []
        q = jnp.array(state.values)
        for s in self.discrete_space.states:
            d = float(jnp.linalg.norm(q - jnp.array(s.values)))
            if 0 < d <= self.max_dist:
                nbrs.append(s)
        return nbrs


# ==============================================================================
# BENCHMARK SUITE
# ==============================================================================

@pytest.mark.benchmark
def test_benchmark_graph_topology(capsys):
    """
    Scenario: Random Graph with 2,000 Nodes and 10,000 Edges.
    """
    print("\n\n--- BENCHMARK 1: GRAPH TOPOLOGY (N=2,000, E=10,000) ---")

    # 1. Setup Data
    states = [f"Node_{i}" for i in range(2000)]
    space = AbstractDiscreteStateSpace(states)

    edges = []
    for _ in range(10000):
        u = random.choice(states)
        v = random.choice(states)
        edges.append((u, v))

    # 2. Run BASELINE (Generic)
    base_topo = GenericGraphTopo(space, edges)
    t0 = time.time()
    _ = base_topo.multi_step_successor(states[0], steps=50).states
    t_base = time.time() - t0
    print(f"Generic Implementation: {t_base:.4f}s")

    # 3. Run SPECIALIZED (GraphTopology)
    # Re-create space to ensure no caching affects results
    space_spec = AbstractDiscreteStateSpace(states)
    spec_topo = GraphTopology(space_spec, edges)
    t0 = time.time()
    _ = spec_topo.multi_step_successor(states[0], steps=50).states
    t_spec = time.time() - t0
    print(f"Special Implementation: {t_spec:.4f}s")

    speedup = t_base / t_spec
    print(f"Speedup: {speedup:.1f}x")
    assert t_spec < t_base


@pytest.mark.benchmark
def test_benchmark_delta_topology(capsys):
    """
    Scenario: 50x50 Grid (2,500 States). 4-Connected.
    """
    print("\n\n--- BENCHMARK 2: DELTA TOPOLOGY (Grid 50x50 = 2,500 States) ---")

    states = [VectorState((x, y)) for x in range(50) for y in range(50)]
    space = VectorStateSpace(states, dim=2)
    deltas = [VectorState((1, 0)), VectorState((-1, 0)), VectorState((0, 1)), VectorState((0, -1))]

    # 2. Run BASELINE
    base_topo = GenericDeltaTopo(space, deltas)
    t0 = time.time()
    _ = base_topo.multi_step_successor(states[1250], steps=75).states
    t_base = time.time() - t0
    print(f"Generic Implementation: {t_base:.4f}s")

    # 3. Run SPECIALIZED
    # Need fresh space to clear internal caches
    space_spec = VectorStateSpace(states, dim=2)
    spec_topo = DeltaTopology(space_spec, deltas)
    t0 = time.time()
    _ = spec_topo.multi_step_successor(states[1250], steps=75).states
    t_spec = time.time() - t0
    print(f"Special Implementation: {t_spec:.4f}s")

    speedup = t_base / t_spec
    print(f"Speedup: {speedup:.1f}x")
    assert t_spec < t_base


@pytest.mark.benchmark
def test_benchmark_metric_topology(capsys):
    """
    Scenario: 500 Points on a line. Connect neighbors < 1.5.
    (Reducing N slightly because Generic O(N^2) is excruciatingly slow)
    """
    print("\n\n--- BENCHMARK 3: METRIC TOPOLOGY (N=500, Pairwise Check) ---")

    states = [VectorState((float(i),)) for i in range(500)]
    space = VectorStateSpace(states, dim=1)

    # 2. Run BASELINE (Python Loop O(N^2))
    base_topo = GenericMetricTopo(space, max_dist=1.5)
    t0 = time.time()
    _ = base_topo.multi_step_successor(states[250], steps=50).states
    t_base = time.time() - t0
    print(f"Generic Implementation: {t_base:.4f}s")

    # 3. Run SPECIALIZED (JAX Matrix Broadcasting)
    space_spec = VectorStateSpace(states, dim=1)
    spec_topo = MetricDiscreteTopology(space_spec, max_dist=1.5)
    t0 = time.time()
    _ = spec_topo.multi_step_successor(states[250], steps=50).states
    t_spec = time.time() - t0
    print(f"Special Implementation: {t_spec:.4f}s")

    speedup = t_base / t_spec
    print(f"Speedup: {speedup:.1f}x")
    assert t_spec < t_base