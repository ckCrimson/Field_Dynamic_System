import time
import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Sequence
import matplotlib.pyplot as plt

from src.field_dynamic_system.core import FieldTransform
from src.field_dynamic_system.core.field.compositions import AdditionComposition, MultiplicationComposition
from src.field_dynamic_system.neighbor.discrete import DiscreteTopology
from src.field_dynamic_system.generator.field_genetators import GenericMarkovianDiscreteFieldGenerator
from src.field_dynamic_system.generator.kernel import AbstractTransitionKernel


# ==========================================
# 1. COMPLEX ALGEBRA COMPONENTS
# ==========================================

class ComplexPhaseKernel(AbstractTransitionKernel):
    """
    Moves mass like diffusion, but rotates the phase of the wave when it travels.
    This creates Quantum-like interference patterns.
    """

    def compute_raw_batch(self, edge_indices, context_mapper=None):
        is_self = edge_indices[:, 0] == edge_indices[:, 1]

        # Magnitude (Amplitude): 50% stay, 50% spread
        mags = jnp.where(is_self, 0.5, 0.5 / 6.0)

        # Phase Shift: 0 radians if staying, Pi/4 radians if moving
        phases = jnp.where(is_self, 0.0, jnp.pi / 4.0)

        # Euler's Formula: z = r * e^(i * theta)
        complex_weights = mags * jnp.exp(1j * phases)

        # MUST cast to complex64 for JAX
        return complex_weights.reshape(-1, 1).astype(jnp.complex64)


class ComplexNormTransform(FieldTransform):
    """
    Normalizes the field so the sum of all MAGNITUDES equals 1.0.
    Preserves the phase (angle) of the complex vectors.
    """

    def __call__(self, raw_data: jnp.ndarray) -> jnp.ndarray:
        # 1. Get the real magnitude (absolute value) of each complex number
        magnitudes = jnp.abs(raw_data)

        # 2. Sum them to find the total mass of the system
        total_mass = jnp.sum(magnitudes)

        # 3. Divide the complex vector by the real scalar.
        # This scales the amplitude but leaves the phase perfectly intact.
        return raw_data / (total_mass + 1e-10)


# ==========================================
# 2. TOPOLOGY
# ==========================================

class HexGridTopology(DiscreteTopology):
    def __init__(self):
        super().__init__(state_space=None)

    def compute_neighbors(self, state_val: Any) -> Sequence[Any]:
        if hasattr(state_val, 'tolist'):
            state_val = tuple(state_val.tolist())
        elif isinstance(state_val, list):
            state_val = tuple(state_val)

        x, y = state_val
        return [
            (x, y),  # The Self-Loop!
            (x + 1, y), (x - 1, y),
            (x, y + 1), (x - 1, y + 1),
            (x + 1, y - 1), (x, y - 1)
        ]


# ==========================================
# 3. VISUALIZATION (DUAL GRAPH)
# ==========================================

def visualize_complex_field(topology, final_field_array):
    print("\n-> Phase 5: Visualizing Complex Field...")
    coords = np.array(topology._id_to_raw)
    x = coords[:, 0]
    y = coords[:, 1]

    # Extract Real Metrics from the Complex JAX Array
    np_field = np.array(final_field_array).flatten()
    magnitudes = np.abs(np_field)  # Length of the vector
    phases = np.angle(np_field)  # Angle from -Pi to Pi

    # Create a side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"Complex Wave Dynamics on Hexagonal Lattice (N={len(magnitudes)})", fontsize=16)

    # GRAPH 1: MAGNITUDE (The "Mass")
    sc1 = ax1.scatter(x, y, c=magnitudes, cmap='plasma', s=90, edgecolors='none')
    fig.colorbar(sc1, ax=ax1, label='Magnitude |z|')
    ax1.set_title('Field Magnitude (Interference Pattern)')
    ax1.axis('equal')
    ax1.grid(True, linestyle='--', alpha=0.3)

    # GRAPH 2: PHASE (The "Angle")
    # We use 'hsv' colormap because phase is cyclical (Pi and -Pi are the same color)
    sc2 = ax2.scatter(x, y, c=phases, cmap='hsv', s=90, edgecolors='none', vmin=-np.pi, vmax=np.pi)
    fig.colorbar(sc2, ax=ax2, label='Phase arg(z) (Radians)')
    ax2.set_title('Field Phase (Wavefronts)')
    ax2.axis('equal')
    ax2.grid(True, linestyle='--', alpha=0.3)

    print("✨ Plot generated. Displaying...")
    plt.tight_layout()
    plt.show()


# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def test_complex_data_flow():
    print("\n==================================================")
    print("🌊 QUANTUM DATA FLOW: Complex Hex Grid")
    print("==================================================")

    STEPS = 120  # Enough steps to see interference bands form

    print(f"-> Phase 1: Bare-Metal Discovery (Depth={STEPS})...")
    topology = HexGridTopology()
    topology.expand_frontier([(0, 0)], depth=STEPS)
    num_nodes = len(topology._id_to_raw)

    print("-> Phase 2: Compiling Complex Matrix...")
    adj_matrix = topology.adjacency_matrix

    # We use the ComplexPhaseKernel to generate complex weights!
    kernel = ComplexPhaseKernel()
    complex_weights = kernel.compute_raw_batch(adj_matrix.indices)

    raw_topo_tuple = (
        adj_matrix.indices[:, 0],
        adj_matrix.indices[:, 1],
        complex_weights.flatten(),
        num_nodes
    )

    print("-> Phase 3: Initializing Complex Field...")
    # CRITICAL: Field must be initialized as complex64!
    raw_field = jnp.zeros((num_nodes, 1), dtype=jnp.complex64)
    # Origin gets a pure real magnitude of 1.0 + 0.0i
    raw_field = raw_field.at[0].set(1.0 + 0j)

    print("-> Phase 4: Executing Complex JAX Physics...")
    gen = GenericMarkovianDiscreteFieldGenerator(
        topology=None,
        kernel=ComplexPhaseKernel(),
        step_composer=AdditionComposition(),  # Performs Complex Addition!
        chain_composer=MultiplicationComposition(),  # Performs Complex Multiplication!
        extrinsic_transform=ComplexNormTransform()  # Normalizes based on Magnitude
    )

    t0 = time.time()
    final_raw_field = gen.generate_raw_multi_step(
        raw_field=raw_field,
        raw_topology=raw_topo_tuple,
        steps=STEPS
    ).block_until_ready()

    print(f"   Physics Time:  {time.time() - t0:.4f}s")

    # Verify Mass
    final_mass = float(jnp.sum(jnp.abs(final_raw_field)))
    print(f"   Final Magnitude Sum: {final_mass:.4f}")

    visualize_complex_field(topology, final_raw_field)


if __name__ == "__main__":
    test_complex_data_flow()