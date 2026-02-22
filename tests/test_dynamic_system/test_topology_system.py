import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

from src.field_dynamic_system.clock.window_clock import WindowedInternalClock
from src.field_dynamic_system.core import VectorState
from src.field_dynamic_system.core.state.continous import ContinuousStateSpace
from src.field_dynamic_system.neighbor.continuous import ContinuousTopology
from src.field_dynamic_system.operator import InteractionContext
from src.field_dynamic_system.systems.dynamic.topology import ContinuousTopologyDynamicSystem
from src.field_dynamic_system.systems.static.topology import ContinuousStaticTopologySystem


# ==========================================
# 1. CORE FRAMEWORK MOCKS (The Architecture)
# ==========================================




class LinearGrowthTopology(ContinuousTopology):
    def __init__(self, step_size: float):
        super().__init__(state_space=ContinuousStateSpace)
        self.step_size = step_size

    def get_raw_successor(self, state_raw: jnp.ndarray) -> tuple:
        return (state_raw - self.step_size, state_raw + self.step_size)

    def successor(self, state: Any) -> 'StateSpace':
        pass

@jax.jit
def jax_random_uniform(state: jnp.ndarray, bounds: tuple, prng_key: jnp.ndarray) -> jnp.ndarray:
    lows, highs = bounds
    return jax.random.uniform(prng_key, shape=state.shape, minval=lows, maxval=highs)


class RNGOperator:
    def observe(self, state: jnp.ndarray, context: InteractionContext) -> jnp.ndarray:
        bounds = context.global_params['topology_data']
        key = context.rng_key
        return jax_random_uniform(state, bounds, key)


# ==========================================
# 2. THE PYTEST SUITE
# ==========================================

class TestContinuousRandomWalker:
    """Pytest suite to validate the 1D Walker OOP Path."""

    def test_random_walker_execution(self):
        """Tests that the dynamic system correctly extracts, rewires, and updates."""
        num_walkers = 5
        iterations = 5

        # 1. Initialize Components
        initial_state = VectorState(np.zeros(num_walkers))
        topology = LinearGrowthTopology(step_size=1.0)
        operator = RNGOperator()
        clock = WindowedInternalClock()

        static_sys = ContinuousStaticTopologySystem(initial_state, topology)
        dyn_sys = ContinuousTopologyDynamicSystem(
            topology_system=static_sys,
            operator=operator,
            clock=clock,
            is_dynamic=True
        )

        # 2. Check Initial State
        raw_start = dyn_sys.get_raw_data()
        assert jnp.all(raw_start == 0.0), "Walkers should start at 0.0"

        # Initial bounds should be exactly -1.0 to 1.0
        lows, highs = dyn_sys._current_topology_data
        assert jnp.all(lows == -1.0)
        assert jnp.all(highs == 1.0)

        # 3. Execute Loop
        key = jax.random.PRNGKey(42)
        previous_state = raw_start

        for i in range(iterations):
            key, subkey = jax.random.split(key)
            dyn_sys.apply_operator({'rng_key': subkey})

            current_state = dyn_sys.get_raw_data()

            # Assert they moved!
            assert not jnp.all(current_state == previous_state)

            # Assert the topology bounded them correctly (they couldn't move more than 1.0 unit per tick)
            distance_moved = jnp.abs(current_state - previous_state)
            assert jnp.all(distance_moved <= 1.0), "Walker teleported past topological boundaries!"

            previous_state = current_state

        # 4. Check Clock
        assert dyn_sys._clock.current_tick == iterations, "Clock did not tick correctly"