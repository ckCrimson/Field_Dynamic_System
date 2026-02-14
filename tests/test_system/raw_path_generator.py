import time
import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Sequence

from src.field_dynamic_system.core import FieldTransform
from src.field_dynamic_system.core.field import NormTransform, MultiplicationComposition
from src.field_dynamic_system.neighbor.discrete import DiscreteTopology
from src.field_dynamic_system.generator.field_genetators import GenericMarkovianDiscreteFieldGenerator
from src.field_dynamic_system.core.field.compositions import AdditionComposition
from src.field_dynamic_system.generator.kernel import AbstractTransitionKernel

import matplotlib.pyplot as plt


# 1. THE DUMMY KERNEL
class DummyKernel(AbstractTransitionKernel):
    def compute_raw_batch(self, edge_indices, context_mapper=None):
        return jnp.ones((edge_indices.shape[0], 1), dtype=jnp.float32)


class DiffusionKernel(AbstractTransitionKernel):
    def compute_raw_batch(self, edge_indices, context_mapper=None):
        is_self = edge_indices[:, 0] == edge_indices[:, 1]
        weights = jnp.where(is_self, 0.5, 0.5 / 6.0)
        return weights.reshape(-1, 1)


class NormTransforms(FieldTransform):
    def __call__(self, raw_data: jnp.ndarray) -> jnp.ndarray:
        total_mass = jnp.sum(raw_data)
        return raw_data / (total_mass + 1e-10)


# 2. THE RAW TOPOLOGY
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
            (x + 1, y), (x - 1, y),
            (x, y + 1), (x - 1, y + 1),
            (x + 1, y - 1), (x, y - 1)
        ]


def test_bare_metal_data_flow():
    print("\n==================================================")
    print("🔬 BARE-METAL DATA FLOW: Hex Grid Raw Path")
    print("==================================================")

    STEPS = 50  # Changed to 3 to match the math we discussed

    print(f"-> Phase 1: Bare-Metal Discovery (Depth={STEPS})...")
    topology = HexGridTopology()
    initial_tuple = (0, 0)
    topology.expand_frontier([initial_tuple], depth=STEPS)

    num_nodes = len(topology._id_to_raw)
    print(f"   States Discovered: {num_nodes} pure tuples.")

    print("-> Phase 2: Compiling Hardware Matrix...")
    adj_matrix = topology.adjacency_matrix
    raw_topo_tuple = (
        adj_matrix.indices[:, 0],
        adj_matrix.indices[:, 1],
        adj_matrix.data,
        num_nodes
    )
    print(f"   Edges Compiled: {len(adj_matrix.data)}")

    print("-> Phase 3: Initializing Raw Field...")
    raw_field = jnp.zeros((num_nodes, 1), dtype=jnp.float32).at[0].set(1.0)

    print("-> Phase 4: Executing JAX Physics...")
    gen = GenericMarkovianDiscreteFieldGenerator(
        topology=None,
        kernel=DummyKernel(),
        step_composer=AdditionComposition(),
        chain_composer=MultiplicationComposition(),
        intrinsic_transform=None,  # Removed to prevent double-normalization scaling issues per step
        extrinsic_transform=NormTransforms()
    )

    t0_run = time.time()
    final_raw_field = gen.generate_raw_multi_step(
        raw_field=raw_field,
        raw_topology=raw_topo_tuple,
        steps=STEPS
    ).block_until_ready()

    t_run = time.time() - t0_run
    print(f"   Physics Time:  {t_run:.4f}s")

    final_mass = float(jnp.sum(final_raw_field))

    # # --- NEW PHASE: PRINTING THE PDF ---
    # print("\n==================================================")
    # print(f"📊 EXACT PROBABILITY DISTRIBUTION (t={STEPS})")
    # print("==================================================")
    #
    # magnitudes = np.array(final_raw_field).flatten()
    # state_probs = []
    #
    # for i, state in enumerate(topology._id_to_raw):
    #     prob = float(magnitudes[i])
    #     if prob > 1e-6:  # Filter out empty states
    #         state_probs.append((state, prob))
    #
    # # Sort by probability (descending), then by coordinates
    # state_probs.sort(key=lambda x: (-x[1], x[0]))
    #
    # for state, prob in state_probs:
    #     print(f"State: {state} | Probability: {prob:.5f}")

    print("\n==================================================")
    print(f"Final Mass (Conservation): {final_mass:.4f}")
    print("==================================================")

    assert 0.99 < final_mass < 1.01, "Mass Leak!"
    #visualize_hex_field(topology, final_raw_field)


def visualize_hex_field(topology, final_field_array):
    print("\n-> Phase 5: Visualizing Results...")
    coords = np.array(topology._id_to_raw)
    x = coords[:, 0]
    y = coords[:, 1]
    magnitudes = np.array(final_field_array).flatten()

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x, y, c=magnitudes, cmap='plasma', s=200, edgecolors='none')

    plt.colorbar(scatter, label='Field Intensity (Probability)')
    plt.title(f"Field Diffusion on Hexagonal Lattice (N={len(magnitudes)})")
    plt.xlabel("Axial X")
    plt.ylabel("Axial Y")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.axis('equal')

    print("✨ Plot generated. Displaying...")
    plt.show()


if __name__ == "__main__":
    test_bare_metal_data_flow()