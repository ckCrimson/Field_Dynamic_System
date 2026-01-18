import time
import pytest
import jax.numpy as jnp
from src.field_dynamic_system.core.state import VectorState, VectorStateSpace
from src.field_dynamic_system.core.state.discrete import IndexedVectorStateSpace


# --- Fixtures ---
@pytest.fixture
def grid_space():
    """
    Creates a small 10x10 grid (100 points) for functional testing.
    Indexed on Axis 0 (x-axis) only.
    """
    vectors = [VectorState((float(x), float(y))) for x in range(10) for y in range(10)]
    return IndexedVectorStateSpace(vectors, dim=2, indexed_axes=(0,))


# --- Functional Tests ---

def test_initialization_builds_index(grid_space):
    """Verify that the internal hash map is populated correctly."""
    # We indexed Axis 0. There should be 10 unique keys (0.0 to 9.0).

    # OLD NAME: grid_space._index_map
    # NEW NAME: grid_space._axis_index_map

    # Check that Axis 0 exists in the map
    assert 0 in grid_space._axis_index_map

    # Check that it found the keys (0.0, 1.0, etc.)
    axis_0_lookup = grid_space._axis_index_map[0]
    assert len(axis_0_lookup) == 10


def test_search_by_index_hit(grid_space):
    """Test searching on an indexed axis."""
    # Find all points where x = 5.0
    result = grid_space.search_by_index(axis_idx=0, value=5.0)

    assert result.num_states == 10
    # Verify all returned vectors actually have x=5.0
    matrix = result.get_matrix()
    assert jnp.all(matrix[:, 0] == 5.0)


def test_search_by_index_miss(grid_space):
    """Test searching for a value that doesn't exist."""
    result = grid_space.search_by_index(axis_idx=0, value=99.0)
    assert result.num_states == 0


def test_search_fallback_to_standard(grid_space):
    """
    Test searching on Axis 1 (which is NOT indexed).
    It should fall back to the slow O(N) filter but still work.
    """
    # Find all points where y = 3.0
    result = grid_space.search_by_index(axis_idx=1, value=3.0)

    assert result.num_states == 10
    matrix = result.get_matrix()
    assert jnp.all(matrix[:, 1] == 3.0)


def test_select_index_compound(grid_space):
    """Test compound search: x=5 AND y=3."""
    # x=5 is indexed (fast), y=3 is manual (slow)
    result = grid_space.select_index(axes=[0, 1], values=[5.0, 3.0])

    # Should only be one point: (5.0, 3.0)
    assert result.num_states == 1
    state = result.get_matrix()[0]
    assert jnp.array_equal(state, jnp.array([5.0, 3.0]))


# --- Performance Benchmark ---

def test_benchmark_indexed_vs_naive(capsys):
    """
    Compare O(1) Indexed Search vs O(N) Naive Scan.
    Target: >100x Speedup on large datasets.
    """
    N = 100_000
    print(f"\n\n--- Index Benchmark (N={N}) ---")

    # 1. Setup Data: Large 1D line
    raw_vectors = [VectorState((float(i), 0.0)) for i in range(N)]

    # 2. Setup Spaces
    print("Building Standard Space...")
    standard_space = VectorStateSpace(raw_vectors, dim=2)

    print("Building Indexed Space...")
    t0 = time.time()
    indexed_space = IndexedVectorStateSpace(raw_vectors, dim=2, indexed_axes=(0,))
    build_time = time.time() - t0
    print(f"Index Build Time: {build_time:.4f}s")

    # 3. Define Query
    target_val = 5000.0  # Middle of the pack

    # 4. Measure Naive (Standard)
    start = time.time()
    for _ in range(100):
        _ = standard_space.filter_by_index(0, target_val)
    naive_dur = (time.time() - start) / 100

    # 5. Measure Indexed (Optimized)
    start = time.time()
    for _ in range(100):
        _ = indexed_space.search_by_index(0, target_val)
    indexed_dur = (time.time() - start) / 100

    # 6. Report
    print(f"Naive Time:   {naive_dur:.6f}s")
    print(f"Indexed Time: {indexed_dur:.6f}s")

    speedup = naive_dur / indexed_dur if indexed_dur > 0 else 0
    print(f"🚀 Speedup:     {speedup:.1f}x")

    # Assertion: It must be significantly faster
    # (Allowing some margin, but typically this is 1000x+)
    assert speedup > 10.0