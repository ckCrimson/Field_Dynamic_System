import sys

# Force Python to ignore cached files
sys.dont_write_bytecode = True

import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# --- Core Imports ---
from src.field_dynamic_system.core import FieldTransform
from src.field_dynamic_system.core.field.compositions import AdditionComposition, MultiplicationComposition
from src.field_dynamic_system.core.field.algebra import ComplexFieldAlgebra
from src.field_dynamic_system.neighbor.discrete import DiscreteTopology
from src.field_dynamic_system.generator.field_genetators import GenericMarkovianDiscreteFieldGenerator
from src.field_dynamic_system.generator.kernel import AbstractTransitionKernel

# --- OOP System Imports ---
from src.field_dynamic_system.systems.static.topology import DiscreteStaticTopologySystem
from src.field_dynamic_system.systems.static.state import DiscreteStaticStateSystem
from src.field_dynamic_system.systems.static.field import DiscreteStaticFieldGeneratorSystem
from src.field_dynamic_system.core.state.discrete import VectorStateSpace, VectorState


# ==========================================
# 1. TOPOLOGIES (8-Fold & Hexagonal)
# ==========================================
class Grid8Topology(DiscreteTopology):
    def __init__(self): super().__init__(state_space=None)

    def compute_neighbors(self, state_val):
        x, y = state_val if isinstance(state_val, tuple) else state_val.values
        return [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),
                (x + 1, y + 1), (x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1)]


class HexagonalTopology(DiscreteTopology):
    def __init__(self): super().__init__(state_space=None)

    def compute_neighbors(self, state_val):
        q, r = state_val if isinstance(state_val, tuple) else state_val.values
        return [(q, r), (q + 1, r), (q, r + 1), (q - 1, r + 1),
                (q - 1, r), (q, r - 1), (q + 1, r - 1)]


# ==========================================
# 2. PHYSICS (Schrödinger Kernel & L2 Norm)
# ==========================================
class DiscreteSchrodingerKernel(AbstractTransitionKernel):
    def __init__(self, degree: int, alpha: float = 0.05):
        self.degree = degree
        self.alpha = alpha

    def compute_raw_batch(self, edge_indices, context_mapper=None):
        is_self = edge_indices[:, 0] == edge_indices[:, 1]
        self_weight = 1.0 - (self.degree * 1j * self.alpha)
        neighbor_weight = 1j * self.alpha
        weights = jnp.where(is_self, self_weight, neighbor_weight)
        return weights.reshape(-1, 1).astype(jnp.complex64)


class QuantumL2NormTransform(FieldTransform):
    def __call__(self, raw_data: jnp.ndarray) -> jnp.ndarray:
        norm_factor = jnp.sqrt(jnp.sum(jnp.abs(raw_data) ** 2) + 1e-10)
        return raw_data / norm_factor


# ==========================================
# 3. VISUALIZATION & BENCHMARKING
# ==========================================
def visualize_quantum_state(coordinates, final_field_array, is_hex=False, title_prefix=""):
    print(f"\n-> Rendering {title_prefix} Wavefront...")
    coords = np.array(coordinates)

    if is_hex:
        x = np.sqrt(3) * coords[:, 0] + (np.sqrt(3) / 2) * coords[:, 1]
        y = (3.0 / 2.0) * coords[:, 1]
    else:
        x, y = coords[:, 0], coords[:, 1]

    np_field = np.array(final_field_array).flatten()
    magnitudes = np.abs(np_field)
    phases = np.angle(np_field)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"{title_prefix} Quantum Diffraction (N={len(magnitudes)})", fontsize=16, fontweight='bold')

    mag_max = np.max(magnitudes)
    wave_exists = magnitudes > (mag_max * 1e-4)

    mags_masked = np.where(wave_exists, magnitudes, np.nan)
    phases_masked = np.where(wave_exists, phases, np.nan)

    marker_type = 'h' if is_hex else 's'

    sc1 = ax1.scatter(x, y, c=mags_masked, cmap='plasma', s=40, marker=marker_type,
                      norm=colors.LogNorm(vmin=mag_max * 1e-4, vmax=mag_max * 0.1), edgecolors='none')
    fig.colorbar(sc1, ax=ax1, label='Probability Amplitude |z| (Log)')
    ax1.axis('equal');
    ax1.grid(True, linestyle='--', alpha=0.2)
    ax1.set_facecolor('#050510')

    sc2 = ax2.scatter(x, y, c=phases_masked, cmap='hsv', s=40, marker=marker_type,
                      vmin=-np.pi, vmax=np.pi, edgecolors='none')
    fig.colorbar(sc2, ax=ax2, label='Phase arg(z) (Radians)')
    ax2.axis('equal');
    ax2.grid(True, linestyle='--', alpha=0.2)

    plt.tight_layout()
    plt.show(block=False)  # Non-blocking so the benchmark keeps running!
    plt.pause(0.1)


def plot_benchmarks(results):
    print("\n-> Generating Benchmark Report...")
    names = [r['name'] for r in results]
    times = [r['time'] for r in results]
    nodes = [r['nodes'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, times, color=['#4C72B0', '#55A868'])

    ax.set_ylabel('Execution Time (Seconds)')
    ax.set_title('JAX Quantum Engine Performance Benchmark', fontweight='bold')

    # Add Node Count labels on top of bars
    for bar, n in zip(bars, nodes):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + (max(times) * 0.02),
                f'N={n:,}\nSteps=120', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()  # Final blocking show


# ==========================================
# 4. THE MASTER ORCHESTRATOR RUNNER
# ==========================================
def execute_quantum_experiment(topology_class, degree, is_hex, steps=120, name="Square Grid"):
    print(f"\n{'=' * 60}")
    print(f"⚛️  INITIALIZING EXPERIMENT: {name} Double-Slit")
    print(f"{'=' * 60}")

    print("-> Expanding Spatial Manifold...")
    topology = topology_class()
    topology_system = DiscreteStaticTopologySystem.from_raw_data(
        raw_initial_state=[(0, 0)],
        topology=topology
    )
    topology_system.create_multi_step_states_space(steps=steps)

    state_system = DiscreteStaticStateSystem.from_raw_data(
        raw_initial_state=topology_system.get_raw_state_space(),
        state_space=VectorStateSpace(vectors=[], dim=2),
        state_class=VectorState
    )

    print("-> Constructing Physical Barrier (Double-Slit)...")
    space = state_system.get_state_space()
    num_nodes = len(space._idx_to_state)
    mask = np.ones((num_nodes, 1), dtype=np.complex64)

    wall_axis = 5
    slit_1, slit_2 = 12, -12

    for i, state in enumerate(space._idx_to_state):
        c1, c2 = state.values
        if int(round(c1)) == wall_axis and int(round(c2)) not in [slit_1, slit_2]:
            mask[i] = 0.0 + 0.0j

    jax_mask = jnp.array(mask)

    print("-> Fusing Math Pipeline...")
    generator = GenericMarkovianDiscreteFieldGenerator(
        topology=None,
        step_composer=AdditionComposition(),
        chain_composer=MultiplicationComposition(),
        global_composer=MultiplicationComposition(),
        kernel=DiscreteSchrodingerKernel(degree=degree),
        extrinsic_transform=QuantumL2NormTransform()
    )

    print("-> Booting Static System Orchestrator...")
    system = DiscreteStaticFieldGeneratorSystem.from_systems(
        generator=generator,
        field_algebra=ComplexFieldAlgebra(),
        state_system=state_system,
        topology_system=topology_system
    )

    # ---> THE TYPE FIX <---
    # Force the starting array to be purely complex64 so JAX fori_loop compiles
    complex_impulse = jnp.zeros((num_nodes, 1), dtype=jnp.complex64).at[0].set(1.0 + 0.0j)
    system.set_field(complex_impulse)

    print(f"-> Executing Quantum Evolution ({steps} Steps)...")

    # Force JAX compilation step first to get an accurate benchmark of the pure math!
    system.save_generated_field(steps=1, raw_global_field=jax_mask)
    system.set_field(complex_impulse)  # Reset

    # ---> THE TRUE BENCHMARK <---
    t0 = time.time()
    system.save_generated_field(steps=steps, raw_global_field=jax_mask)
    final_field = system.get_raw_fields()

    # ASYNC SYNC: Force Python to wait for the GPU/CPU to finish computing
    final_field.block_until_ready()

    t1 = time.time()
    exec_time = t1 - t0
    print(f"   Evolution Complete! Compute Time: {exec_time:.4f}s")

    visualize_quantum_state(system.get_raw_state_space(), final_field, is_hex=is_hex, title_prefix=name)

    return {
        'name': name,
        'nodes': num_nodes,
        'time': exec_time
    }


# ==========================================
# 5. EXECUTION & PYTEST WRAPPERS
# ==========================================
def test_8_fold_moore_quantum_walk():
    execute_quantum_experiment(Grid8Topology, degree=8, is_hex=False, steps=120, name="8-Fold Moore")


def test_6_fold_hexagonal_quantum_walk():
    execute_quantum_experiment(HexagonalTopology, degree=6, is_hex=True, steps=120, name="6-Fold Hexagonal")


if __name__ == "__main__":
    print("\n🚀 STARTING FULL QUANTUM BENCHMARK SUITE...")

    results = []

    # Run the tests and collect the timing data
    # res_sq = execute_quantum_experiment(Grid8Topology, degree=8, is_hex=False, steps=120, name="8-Fold Moore")
    # results.append(res_sq)

    res_hx = execute_quantum_experiment(HexagonalTopology, degree=6, is_hex=True, steps=120, name="6-Fold Hexagonal")
    results.append(res_hx)

    # Plot the final performance bar chart
    plot_benchmarks(results)