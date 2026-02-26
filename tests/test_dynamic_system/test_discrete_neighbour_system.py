import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse
from typing import Any, List, Tuple, NamedTuple, Dict, Optional
from dataclasses import dataclass

from src.field_dynamic_system.clock.window_clock import WindowedInternalClock
from src.field_dynamic_system.core import VectorState
from src.field_dynamic_system.neighbor import DiscreteTopology
from src.field_dynamic_system.operator import InteractionContext, ClassicalOperator
from src.field_dynamic_system.systems.dynamic.topology import DiscreteTopologyDynamicSystem
from src.field_dynamic_system.systems.static.topology import DiscreteStaticTopologySystem


# ==========================================
# 1. THE REAL TOPOLOGY IMPLEMENTATION
# ==========================================

class DistanceTopology(DiscreteTopology):
    """
    A purely mathematical topology defining 1-step neighbors on a 2D grid.
    No mock state space required; it just needs bounds to build the matrix.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._matrix = self._build_adjacency_matrix()
        super().__init__()

    def _build_adjacency_matrix(self) -> sparse.BCOO:
        N = self.width * self.height
        rows, cols = [], []

        print(f"⚡ Building Adjacency Matrix for {self.width}x{self.height} grid...")
        for x in range(self.width):
            for y in range(self.height):
                idx = x * self.width + y

                # Manhattan distance = 1 candidates
                candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

                for cx, cy in candidates:
                    if 0 <= cx < self.width and 0 <= cy < self.height:
                        neighbor_idx = cx * self.width + cy
                        rows.append(idx)
                        cols.append(neighbor_idx)

        indices = jnp.array(np.column_stack((rows, cols)))
        values = jnp.ones(len(rows), dtype=jnp.float32)
        return sparse.BCOO((values, indices), shape=(N, N))

    def get_raw_data(self) -> sparse.BCOO:
        """Required by StaticTopologySystem to extract the initial matrix."""
        return self._matrix

    def get_raw_rewired_matrix(self, current_state_raw, current_matrix):
        """Satisfies dynamic interface, though grid is static here."""
        return current_matrix


# ==========================================
# 2. THE MULTI-STEP OPERATOR
# ==========================================

@jax.jit
def jax_multi_step_jump(current_idx: int, matrix: sparse.BCOO, steps: int, key: jnp.ndarray):
    N = matrix.shape[0]
    state_vec = jnp.zeros((N,), dtype=jnp.float32).at[current_idx].set(1.0)

    def body(i, val): return matrix @ val

    reachability = jax.lax.fori_loop(0, steps, body, state_vec)
    valid_mask = reachability > 0
    logits = jnp.where(valid_mask, reachability, -1e9)
    return jax.random.categorical(key, logits)


class MultiStepRandomOperator(ClassicalOperator):
    def __init__(self, grid_width: int):
        # The operator needs to know the width to flatten/unflatten coordinates
        self.grid_width = grid_width
        super().__init__()

    def observe(self, system_state: jnp.ndarray, context: InteractionContext) -> jnp.ndarray:
        matrix = context.global_params['topology_data']
        steps = context.global_params.get('steps', 1)
        key = context.rng_key

        # 1. TRANSLATE: (x, y) -> 1D Matrix Index
        x, y = system_state[0], system_state[1]
        current_idx = jnp.int32(x * self.grid_width + y)

        # 2. EXECUTE BARE-METAL PHYSICS
        new_idx = jax_multi_step_jump(current_idx, matrix, steps, key)

        # 3. TRANSLATE: 1D Matrix Index -> (x, y)
        new_x = new_idx // self.grid_width
        new_y = new_idx % self.grid_width

        return jnp.array([new_x, new_y], dtype=jnp.float32)


# ==========================================
# 3. INTEGRATION TEST
# ==========================================

class TestDiscreteWalker:
    def test_random_walk_sequence(self):
        print("\n--- 2D RANDOM WALKER TEST (INTEGRATED) ---")

        # 1. Setup Pure Topology & Start State
        width, height = 5, 5
        topo = DistanceTopology(width=width, height=height)
        start_state = VectorState((2, 2))

        # 2. Setup System
        operator = MultiStepRandomOperator(grid_width=width)
        clock = WindowedInternalClock()
        static_sys = DiscreteStaticTopologySystem(start_state, topo)
        system = DiscreteTopologyDynamicSystem(static_sys, operator, clock)

        key = jax.random.PRNGKey(0)
        simulated_inputs = [1, 2, 3]

        for steps in simulated_inputs:
            # Depending on OOP getters, this might return the VectorState object or raw array
            # Let's assume it returns the raw array based on system._current_raw_state
            current_raw = system.get_raw_data()
            print(f"\n📍 CURRENT RAW STATE: {current_raw} (Tick: {clock.current_tick})")
            print(f" > User input simulated: Jump {steps} steps")

            key, subkey = jax.random.split(key)

            system.apply_operator({
                'rng_key': subkey,
                'global_params': {'steps': steps}
            })

            new_raw = system.get_raw_data()
            assert 0 <= new_raw[0] < width
            assert 0 <= new_raw[1] < height

        print("\n✅ Test sequence completed successfully.")


if __name__ == "__main__":
    tester = TestDiscreteWalker()
    tester.test_random_walk_sequence()