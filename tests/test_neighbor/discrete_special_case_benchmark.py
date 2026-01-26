import pytest
import numpy as np
import scipy.sparse as sp
import jax.numpy as jnp
from typing import List

# --- IMPORTS ---
from src.field_dynamic_system.core.state.discrete import VectorStateSpace, VectorState
from src.field_dynamic_system.neighbor.discrete import GraphTopology


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def linear_chain_system():
    """
    Creates a simple 3-node chain: 0 -> 1 -> 2
    """
    # 1. Define States (Fixed Order)
    raw_states = [VectorState((0,)), VectorState((1,)), VectorState((2,))]
    space = VectorStateSpace(raw_states, dim=1)

    # 2. Define Edges (Indices)
    # 0->1, 1->2
    edges = [(0, 1), (1, 2)]

    topology = GraphTopology(space, edges=edges)
    return space, topology


@pytest.fixture
def dense_matrix_system():
    """
    Creates a 2-node cycle: 0 <-> 1 using a Numpy Matrix.
    """
    raw_states = [VectorState((0,)), VectorState((1,))]
    space = VectorStateSpace(raw_states, dim=1)

    # Adjacency Matrix:
    # [[0, 1],  (Node 0 connects to 1)
    #  [1, 0]]  (Node 1 connects to 0)
    matrix = np.array([[0, 1], [1, 0]], dtype=np.float32)

    topology = GraphTopology(space, adjacency_matrix=matrix)
    return space, topology


# =========================================================
# TESTS: EDGE LIST CONSTRUCTION
# =========================================================

def test_edges_immediate_successor(linear_chain_system):
    """ Verify direct neighbor lookup works (0->1). """
    space, topology = linear_chain_system

    # State 0 -> Should find State 1
    state_0 = space.states[0]  # VectorState((0,))
    state_1 = space.states[1]  # VectorState((1,))

    successors = topology.successor(state_0)

    assert successors.num_states == 1
    assert successors.contains(state_1)


def test_edges_dead_end(linear_chain_system):
    """ Verify State 2 has no outgoing edges. """
    space, topology = linear_chain_system

    state_2 = space.states[2]
    successors = topology.successor(state_2)

    assert successors.num_states == 0


def test_edges_multi_step_propagation(linear_chain_system):
    """
    Verify JAX Physics Propagation:
    Step 0: {0}
    Step 1: {1}
    Step 2: {2}
    """
    space, topology = linear_chain_system
    state_0 = space.states[0]

    # 2 Steps from 0 should reach 2
    future_space = topology.multi_step_successor(state_0, steps=2)

    assert future_space.num_states == 1
    assert future_space.contains(space.states[2])


# =========================================================
# TESTS: MATRIX CONSTRUCTION
# =========================================================

def test_matrix_cycle(dense_matrix_system):
    """ Verify matrix-based topology handles cycles correctly. """
    space, topology = dense_matrix_system
    state_0 = space.states[0]

    # Step 1: 0 -> 1
    step1 = topology.multi_step_successor(state_0, steps=1)
    assert step1.contains(space.states[1])

    # Step 2: 0 -> 1 -> 0 (Back home)
    step2 = topology.multi_step_successor(state_0, steps=2)
    assert step2.contains(space.states[0])


def test_jax_matrix_shape(dense_matrix_system):
    """ Verify the internal JAX matrix is built with correct Physics layout. """
    _, topology = dense_matrix_system

    jax_matrix = topology.adjacency_matrix

    # Physics Layout: [Target, Source]
    # For 0<->1, it is symmetric, so still [[0,1],[1,0]]
    dense = jax_matrix.todense()

    assert dense.shape == (2, 2)
    assert dense[0, 1] == 1.0  # Col 1 (Src) -> Row 0 (Tgt)
    assert dense[1, 0] == 1.0  # Col 0 (Src) -> Row 1 (Tgt)


# =========================================================
# TESTS: SCIPY SPARSE INTEGRATION
# =========================================================

def test_sparse_matrix_input():
    """ Verify we can pass a Scipy CSR/LIL matrix directly. """
    N = 100
    states = [VectorState((i,)) for i in range(N)]
    space = VectorStateSpace(states, dim=1)

    # Create a random sparse matrix
    # e.g., a simple line 0->1->2...
    row = np.arange(0, N - 1)
    col = np.arange(1, N)
    data = np.ones(N - 1)
    sparse_matrix = sp.coo_matrix((data, (row, col)), shape=(N, N))

    topology = GraphTopology(space, adjacency_matrix=sparse_matrix)

    # Test connection 50 -> 51
    state_50 = states[50]
    next_space = topology.successor(state_50)

    assert next_space.contains(states[51])
    assert not next_space.contains(states[52])