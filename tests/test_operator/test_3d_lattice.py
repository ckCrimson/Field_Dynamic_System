import time
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass

# Core System Imports
from src.field_dynamic_system.core.state.state import State
# Import the Lazy Space specifically
from src.field_dynamic_system.core.state.discrete import LazyDiscreteStateSpace
from src.field_dynamic_system.core.field.mappings import FieldMapper
from src.field_dynamic_system.core.field.algebra import RealFieldAlgebra
from src.field_dynamic_system.operator.base import InteractionContext
from src.field_dynamic_system.operator.field import FieldBasedOperator, Strategies


@dataclass(frozen=True)
class LatticeState(State):
    """
    Lightweight state container.
    Only instantiated when the Operator identifies a specific winner.
    """
    x: int
    y: int
    z: int

    def __repr__(self):
        return f"({self.x}, {self.y}, {self.z})"


class TestLatticePerformance:

    def test_3d_lattice_performance(self):
        print("\n--- Starting High-Performance 3D Lattice Benchmark ---")

        # PARAMETERS
        # Range: -100 to 100 -> 201 points per dim -> ~8.1 million states
        start, end = -100, 100

        # STEP 1: Generate Coordinates & Values (Math World)
        t0 = time.time()

        range_vals = np.arange(start, end + 1)
        # 'ij' indexing: x varies dim 0, y dim 1, z dim 2
        X, Y, Z = np.meshgrid(range_vals, range_vals, range_vals, indexing='ij')

        # 1. Coordinate Matrix (N, 3)
        # Stack into a single matrix of shape (8120601, 3)
        # This is the "Raw Data" our Lazy Space will hold
        coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

        # 2. Field Values (N,)
        # Function: x^2 + y^2 - z^2
        # Expected Max: x=±100, y=±100, z=0 => 20,000
        values = (X ** 2 + Y ** 2 - Z ** 2).ravel()

        num_states = coords.shape[0]
        t1 = time.time()
        print(f"1. Math Generation: {t1 - t0:.4f}s | States: {num_states:,}")

        # STEP 2: Lazy System Initialization (Topology World)
        # We DO NOT loop here. We just pass the matrix pointer.
        print("2. System Initialization (Lazy Mode)...")

        # A. Create Space (0s overhead, just stores the matrix)
        # We tell it: "Use 'LatticeState' class to view rows when needed"
        space = LazyDiscreteStateSpace(coords, LatticeState)

        # B. Create Mapper
        algebra = RealFieldAlgebra()
        mapper = FieldMapper(space, algebra)

        # C. Set Field Values (Fast Path)
        # We map Indices [0..N] -> Values [v0..vN]
        # This skips all object lookups.
        indices = np.arange(num_states)
        mapper.set_raw_values(indices, values)

        t2 = time.time()
        print(f"   Total Init Time: {t2 - t1:.4f}s (vs ~67s previously)")

        # STEP 3: Execution (The Physics)
        # Find the state with the maximum value
        op = FieldBasedOperator(selection_strategy=Strategies.argmax)
        ctx = InteractionContext(rng_key=jax.random.PRNGKey(0))

        start_obs = time.time()
        best_state = op.observe(mapper, ctx)
        end_obs = time.time()

        print(f"3. Operator Execution: {end_obs - start_obs:.6f}s")
        print(f"   Result State: {best_state}")

        # STEP 4: Verification
        expected_val = 100 ** 2 + 100 ** 2 - 0

        # Verify correctness
        assert abs(best_state.x) == 100
        assert abs(best_state.y) == 100
        assert best_state.z == 0

        print(f"   Verification: PASS (State matches optimal value {expected_val})")


if __name__ == "__main__":
    t = TestLatticePerformance()
    t.test_3d_lattice_performance()