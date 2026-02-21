import sys

from tests.test_system.test_complex_2d_walker import GenericMarkovianDiscreteFieldGenerator, DiscreteSchrodingerKernel

# Force Python to ignore cached .pyc files so it reads your latest Generator fixes!
sys.dont_write_bytecode = True

import time
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# --- Core Imports ---
from src.field_dynamic_system.core import FieldTransform
from src.field_dynamic_system.core.field.compositions import AdditionComposition, MultiplicationComposition
from src.field_dynamic_system.neighbor.discrete import DiscreteTopology
from src.field_dynamic_system.generator.kernel import AbstractTransitionKernel


# ==========================================
# 1. KERNEL & TOPOLOGY
# ==========================================
class ComplexGrid8Kernel(AbstractTransitionKernel):
    def compute_raw_batch(self, edge_indices, context_mapper=None):
        is_self = edge_indices[:, 0] == edge_indices[:, 1]
        mags = jnp.where(is_self, 0.5, 0.5 / 8.0)
        phases = jnp.where(is_self, 0.0, jnp.pi / 4.0)
        return (mags * jnp.exp(1j * phases)).reshape(-1, 1).astype(jnp.complex64)


class ComplexNormTransform(FieldTransform):
    def __call__(self, raw_data: jnp.ndarray) -> jnp.ndarray:
        return raw_data / (jnp.sum(jnp.abs(raw_data)) + 1e-10)


class Grid8Topology(DiscreteTopology):
    def __init__(self): super().__init__(state_space=None)

    def compute_neighbors(self, state_val):
        x, y = state_val if isinstance(state_val, tuple) else state_val.values
        return [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1), (x + 1, y + 1), (x - 1, y - 1)]


# ==========================================
# 2. VISUALIZATION
# ==========================================
def visualize_2d_complex_field(coordinates, final_field_array):
    print("\n-> Visualizing 2D Complex Wave (Logarithmic Exposure)...")
    coords = np.array(coordinates)
    x, y = coords[:, 0], coords[:, 1]
    np_field = np.array(final_field_array).flatten()
    magnitudes = np.abs(np_field)
    phases = np.angle(np_field)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"Double-Slit Diffraction (N={len(magnitudes)})", fontsize=16)

    mag_max = np.max(magnitudes)

    sc1 = ax1.scatter(x, y, c=magnitudes, cmap='plasma', s=40, marker='s',
                      norm=colors.LogNorm(vmin=mag_max * 1e-5, vmax=mag_max * 0.1),
                      edgecolors='none')
    fig.colorbar(sc1, ax=ax1, label='Magnitude |z| (Log Scale)')
    ax1.set_title('Probability Amplitude')
    ax1.axis('equal');
    ax1.grid(True, linestyle='--', alpha=0.3)

    wave_exists = magnitudes > (mag_max * 1e-5)
    phases_masked = np.where(wave_exists, phases, np.nan)

    sc2 = ax2.scatter(x, y, c=phases_masked, cmap='hsv', s=40, marker='s',
                      vmin=-np.pi, vmax=np.pi, edgecolors='none')
    fig.colorbar(sc2, ax=ax2, label='Phase arg(z) (Radians)')
    ax2.set_title('Wave Phase (Cleaned)')
    ax2.axis('equal');
    ax2.grid(True, linestyle='--', alpha=0.3)

    # --- DRAW THE WALL ---
    wall_x, slit_1_y, slit_2_y = 5, 2, -2
    wall_coords_x, wall_coords_y = [], []
    for cx, cy in coords:
        if cx == wall_x and cy != slit_1_y and cy != slit_2_y:
            wall_coords_x.append(cx)
            wall_coords_y.append(cy)

    ax1.scatter(wall_coords_x, wall_coords_y, color='cyan', s=20, marker='x', label='Absorbing Wall')
    ax2.scatter(wall_coords_x, wall_coords_y, color='black', s=20, marker='x')
    ax1.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


# ==========================================
# 3. DIRECT GENERATOR TEST
# ==========================================
def test_raw_global_field_generator():
    print("\n==================================================")
    print("⚛️  RAW GENERATOR TEST (Strict Global Field)")
    print("==================================================")

    # FIXED: Increased steps so the wave has time to actually pass through the slits
    STEPS = 40

    print("-> Expanding Raw Topology...")
    topology = Grid8Topology()
    topology.expand_frontier([(0, 0)], depth=STEPS)

    num_nodes = len(topology._id_to_raw)
    adj_matrix = topology.adjacency_matrix

    kernel = ComplexGrid8Kernel()
    weights = kernel.compute_raw_batch(adj_matrix.indices).flatten()

    raw_topology_tuple = (
        adj_matrix.indices[:, 0],
        adj_matrix.indices[:, 1],
        weights,
        num_nodes
    )

    raw_initial_field = jnp.zeros((num_nodes, 1), dtype=jnp.complex64).at[0].set(1.0)

    print("-> Constructing Raw Global Field...")
    mask = np.ones((num_nodes, 1), dtype=np.complex64)
    wall_x, slit_1_y, slit_2_y = 5, 2, -2

    for i, coords in enumerate(topology._id_to_raw):
        x, y = coords
        if x == wall_x and y != slit_1_y and y != slit_2_y:
            mask[i] = 0.0 + 0.0j

    raw_global_field = jnp.array(mask)
    print(f"   [DIAGNOSTIC] Global Field Wall Nodes: {np.sum(mask == 0.0)}")

    print("-> Initializing Generator...")
    generator = GenericMarkovianDiscreteFieldGenerator(
        topology=None,
        kernel=DiscreteSchrodingerKernel(),
        step_composer=AdditionComposition(),
        chain_composer=MultiplicationComposition(),
        global_composer=MultiplicationComposition(),
        extrinsic_transform=ComplexNormTransform()
    )

    print(f"-> Firing generate_raw_multi_step for {STEPS} steps...")
    t0 = time.time()

    final_field = generator.generate_raw_multi_step(
        raw_field=raw_initial_field,
        raw_topology=raw_topology_tuple,
        steps=STEPS,
        raw_global_field=raw_global_field
    )

    print(f"   Execution Time: {time.time() - t0:.4f}s")

    visualize_2d_complex_field(topology._id_to_raw, final_field)


if __name__ == "__main__":
    test_raw_global_field_generator()