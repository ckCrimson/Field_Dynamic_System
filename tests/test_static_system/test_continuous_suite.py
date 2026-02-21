import sys
from typing import Any

from src.field_dynamic_system.generator.generator_interfaces import IContinuousFieldGenerator
from src.field_dynamic_system.generator.utility.gaussian import GaussianWavePacketGenerator

sys.dont_write_bytecode = True

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# --- Core Interfaces & Mappings ---
from src.field_dynamic_system.core.field.mappings import ContinuousFieldMapper
from src.field_dynamic_system.core.field.algebra import IFieldAlgebra
from src.field_dynamic_system.systems.static.field import AbstractStaticFieldGeneratorSystem
from src.field_dynamic_system.core.state.discrete import VectorStateSpace

# =========================================================
# 1. THE ADAPTER INTERFACE & ORCHESTRATOR
# =========================================================
# (Include your IContinuousFieldGenerator and ContinuousStaticFieldGeneratorSystem here
# if they aren't already saved in your core files)



class ContinuousStaticFieldGeneratorSystem(AbstractStaticFieldGeneratorSystem):
    def __init__(self, generator: IContinuousFieldGenerator, field_algebra: IFieldAlgebra, state_system: Any,
                 is_raw_mode: bool = False):
        super().__init__(generator, field_algebra, is_raw_mode)
        self.state_system = state_system
        if self.current_raw_data is None:
            self.clear_field()

    def generate_raw_field(self, steps: int, **kwargs) -> Any:
        return self.generator.generate_continuous_step(self.current_raw_data, steps, **kwargs)

    def clear_field(self) -> None:
        self.set_field(self.generator.get_initial_parameters())

    def get_state_space(self) -> Any: return self.state_system

    def get_raw_state_space(self) -> Any: return None

    def _wrap_raw_to_mapper(self, raw_params: Any) -> ContinuousFieldMapper:
        base_func = self.generator.get_base_function()

        def bound_bg_func(coords):
            return base_func(coords, raw_params)

        mapper = ContinuousFieldMapper(self.get_state_space(), self.algebra, bg_func=bound_bg_func)
        mapper.parameters = raw_params
        return mapper

    def _sync_raw_from_mapper(self) -> None:
        self.current_raw_data = self.current_mapper.parameters


# =========================================================
# 2. SIMPLE FLOAT ALGEBRA (For Real-Valued Waves)
# =========================================================
class FloatFieldAlgebra(IFieldAlgebra):
    @property
    def dtype(self): return jnp.float32

    def get_zero(self, shape=()): return jnp.zeros(shape, dtype=self.dtype)

    def get_one(self, shape=()): return jnp.ones(shape, dtype=self.dtype)

    def cast(self, values): return jnp.asarray(values, dtype=self.dtype)




# =========================================================
# 4. VISUALIZATION ENGINE
# =========================================================
def plot_continuous_field(mapper: ContinuousFieldMapper, title: str):
    print(f"-> Sampling Continuous Space for {title}...")

    # 1. Create a 100x100 physical mesh grid from -10 to +10
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)

    # 2. Flatten grid into a list of (X, Y) coordinates
    coords = np.stack([X.ravel(), Y.ravel()], axis=-1)

    # 3. FAST JAX EVALUATION: Ask the mapper for the value at every coordinate instantly!
    # Because of our bound closure, this automatically uses the latest wave parameters.
    Z_flat = mapper.get_raw_batch(jnp.array(coords))
    Z = np.array(Z_flat).reshape(100, 100)

    # 4. Plot the Intensity Graph
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Z, levels=50, cmap='inferno')
    plt.colorbar(contour, label='Wave Amplitude')

    # Mark the origin <0,0>
    plt.scatter([0], [0], color='cyan', marker='x', s=100, label='Origin <0,0>')

    # Extract current center parameter from the mapper
    curr_center = mapper.parameters["center"]
    plt.scatter([curr_center[0]], [curr_center[1]], color='lime', marker='o', s=50, label='Wave Center')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


# =========================================================
# 5. THE TEST SUITE
# =========================================================
def test_continuous_gaussian_evolution():
    print("\n" + "=" * 50)
    print(" 🚀 INITIALIZING CONTINUOUS GAUSSIAN ENGINE ")
    print("=" * 50)

    # 1. State Space (2D Vector Space)
    state_space = VectorStateSpace(vectors=[], dim=2)
    algebra = FloatFieldAlgebra()

    # 2. Initialize the Generator starting at origin <0, 0> moving Up-Right
    generator = GaussianWavePacketGenerator(
        initial_center=(0.0, 0.0),
        velocity=(1.5, 0.8),  # Moves faster in X than Y
        spread=1.5
    )

    # 3. Boot the Orchestrator
    system = ContinuousStaticFieldGeneratorSystem(
        generator=generator,
        field_algebra=algebra,
        state_system=state_space  # Supplying state space as the mock state_system
    )

    # 4. Plot Initial State (Time = 0)
    initial_mapper = system.get_field()
    plot_continuous_field(initial_mapper, "Time T=0 (Initial State at <0,0>)")

    # 5. Evolve the Wave! (50 steps at dt=0.1)
    print("\n-> Evolving wave through continuous space...")
    system.save_generated_field(steps=50, dt=0.1)

    # 6. Plot Final State (Time = 50)
    final_mapper = system.get_field()
    plot_continuous_field(final_mapper, "Time T=50 (Wave Translated)")


if __name__ == "__main__":
    test_continuous_gaussian_evolution()
    plt.show()  # Keep windows open at the end