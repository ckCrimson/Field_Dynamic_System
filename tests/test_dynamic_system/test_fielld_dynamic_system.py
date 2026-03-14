import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Any

from src.field_dynamic_system.core.field import MultiplicationComposition
from src.field_dynamic_system.generator import GenericMarkovianDiscreteFieldGenerator
from src.field_dynamic_system.operator import InteractionContext
from src.field_dynamic_system.clock.window_clock import HistoryClock
from src.field_dynamic_system.systems.dynamic.field import DiscreteFieldDynamicSystem
from tests.test_core_field.test_performance_diffusion import AdditionComposition
from tests.test_static_system.double_slit import ComplexGrid8Kernel, Grid8Topology
from tests.test_static_system.test_complex_2d_walker import DiscreteSchrodingerKernel, ComplexNormTransform


# ==========================================
# 1. VISUALIZATION HELPERS
# ==========================================

def plot_complex_field(raw_field: jnp.ndarray, topology: Any, tick: int):
    """Plots the probability distribution of the field."""
    amplitudes = np.abs(raw_field.flatten())

    coords = np.array(topology._id_to_raw)
    x, y = coords[:, 0], coords[:, 1]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, c=amplitudes, cmap='magma', s=100, marker='s')
    plt.colorbar(scatter, label="Probability Amplitude / Intensity")
    plt.title(f"Field State at Tick {tick} (Pre-Collapse)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_trajectory(history_ids: list, topology: Any):
    """Plots the entity's path through the graph."""
    if len(history_ids) < 2:
        print("Not enough history to plot trajectory.")
        return

    coords = np.array([topology._id_to_raw[node_id] for node_id in history_ids])
    x, y = coords[:, 0], coords[:, 1]

    plt.figure(figsize=(10, 8))
    plt.plot(x, y, linestyle='--', color='gray', alpha=0.5)

    u, v = np.diff(x), np.diff(y)
    plt.quiver(x[:-1], y[:-1], u, v, angles='xy', scale_units='xy', scale=1,
               color='cyan', width=0.005, headwidth=4, zorder=3)

    plt.scatter(x[0], y[0], c='green', s=150, zorder=5, edgecolors='black', label='Start')
    plt.scatter(x[-1], y[-1], c='red', s=150, zorder=5, edgecolors='black', label='End')

    plt.title("Quantum Random Walker Trajectory")
    plt.legend()
    plt.grid(True)
    plt.show()


# ==========================================
# 2. PHYSICS WRAPPERS
# ==========================================

def raw_quantum_collapse_operator(state_raw: int, context: InteractionContext) -> int:
    """
    The Operator: Observes the complex field probabilities and collapses
    the entity into a single node ID.
    """
    field_probs = context.global_params['field_data']
    key = context.rng_key

    probs = jnp.abs(field_probs).flatten()
    valid_mask = probs > 1e-8
    logits = jnp.where(valid_mask, jnp.log(probs + 1e-12), -1e9)

    next_node_id = jax.random.categorical(key, logits)
    return next_node_id


# def run_quantum_benchmark():
#     # ==========================================
#     # 3. SETUP
#     # ==========================================
#     STEPS = 40
#     print("-> Expanding Raw Topology...")
#     topology = Grid8Topology()
#     topology.expand_frontier([(0, 0)], depth=STEPS)
#
#     num_nodes = len(topology._id_to_raw)
#     adj_matrix = topology.adjacency_matrix
#
#     kernel = ComplexGrid8Kernel()
#     weights = kernel.compute_raw_batch(adj_matrix.indices).flatten()
#
#     raw_topology_tuple = (
#         adj_matrix.indices[:, 0],
#         adj_matrix.indices[:, 1],
#         weights,
#         num_nodes
#     )
#
#     raw_initial_field = jnp.zeros((num_nodes, 1), dtype=jnp.complex64).at[0].set(1.0 + 0.0j)
#
#     print("-> Constructing Raw Global Field (Double Slit)...")
#     mask = np.ones((num_nodes, 1), dtype=np.complex64)
#     wall_x, slit_1_y, slit_2_y = 5, 2, -2
#
#     for i, coords in enumerate(topology._id_to_raw):
#         x, y = coords
#         if x == wall_x and y != slit_1_y and y != slit_2_y:
#             mask[i] = 0.0 + 0.0j
#
#     raw_global_field = jnp.array(mask)
#     print(f"   [DIAGNOSTIC] Global Field Wall Nodes: {np.sum(mask == 0.0)}")
#
#     print("-> Initializing Generator...")
#     generator = GenericMarkovianDiscreteFieldGenerator(
#         topology=None,
#         kernel=DiscreteSchrodingerKernel(),
#         step_composer=AdditionComposition(),
#         chain_composer=MultiplicationComposition(),
#         global_composer=MultiplicationComposition(),
#         extrinsic_transform=ComplexNormTransform()
#     )
#
#     def raw_field_update_wrapper(raw_field, raw_topo, steps):
#         return generator.generate_raw_multi_step(
#             raw_field=raw_field,
#             raw_topology=raw_topo,
#             steps=steps,
#             raw_global_field=raw_global_field
#         )
#
#     # ==========================================
#     # 4. SYSTEM INITIALIZATION
#     # ==========================================
#     clock = HistoryClock()
#     start_node_id = topology._raw_to_id[(0, 0)]
#
#     system = DiscreteFieldDynamicSystem.from_raw_data(
#         raw_initial_state=start_node_id,
#         raw_field_data=raw_initial_field,
#         raw_topology_data=raw_topology_tuple,
#         raw_entity_operator_fn=raw_quantum_collapse_operator,
#         clock=clock,
#         is_dynamic_field=True,
#         raw_field_update_fn=raw_field_update_wrapper
#     )
#
#     key = jax.random.PRNGKey(42)
#     trajectory_history = [start_node_id]
#
#     # ==========================================
#     # 5. PRODUCTION EXECUTION LOOP
#     # ==========================================
#     while True:
#         curr_id = system.get_raw_data()
#         curr_coords = topology._id_to_raw[curr_id]
#         print(f"\n📍 ENTITY COORD: {curr_coords} | Tick: {clock.current_tick}")
#
#         try:
#             user_input = input("Enter field evolution steps (0 to exit): ")
#             steps = int(user_input)
#         except ValueError:
#             continue
#
#         if steps == 0:
#             break
#
#         print(f"   ⚙️  Evolving Complex Field for {steps} steps...")
#         key, subkey = jax.random.split(key)
#
#         context_kwargs = {
#             'rng_key': subkey,
#             'global_params': {'steps': steps}
#         }
#
#         # --- PRE-COLLAPSE VISUALIZATION (Optional Debugging) ---
#         # To view the field before observation, we manually run the update step
#         # and extract the tensor before the core system runs its full transaction.
#         system.update_field(context_kwargs)
#         plot_complex_field(system.get_raw_field_data(), topology, clock.current_tick)
#
#         # --- THE PRODUCTION PHYSICS TRANSACTION ---
#         # This handles the entity observation and the subsequent quantum wave collapse.
#         # Note: Since we manually called update_field above to plot it, the field will technically
#         # evolve by `steps` twice in this specific debug loop. In true production without the plot,
#         # you would just call apply_operator directly.
#         system.apply_operator(context_kwargs)
#
#         # Track history
#         trajectory_history.append(system.get_raw_data())
#
#     print("\n✅ Simulation Ended. Plotting final trajectory...")
#     plot_trajectory(trajectory_history, topology)


def run_quantum_benchmark():
    STEPS = 40
    start_coords = (0, 0)

    # 1. DEFINE THE WORLD (OOP)
    topology = Grid8Topology()

    generator = GenericMarkovianDiscreteFieldGenerator(
        topology=None,  # Compiler handles raw topology injection
        kernel=DiscreteSchrodingerKernel(),
        step_composer=AdditionComposition(),
        chain_composer=MultiplicationComposition(),
        global_composer=MultiplicationComposition(),
        extrinsic_transform=ComplexNormTransform()
    )

    # 2. DEFINE THE DOUBLE SLIT MASK
    def double_slit_builder(coords_list, num_nodes):
        mask = np.ones((num_nodes, 1), dtype=np.complex64)
        wall_x, slit_1_y, slit_2_y = 8, 6, -6
        for i, (x, y) in enumerate(coords_list):
            if x == wall_x and y != slit_1_y and y != slit_2_y:
                mask[i] = 0.0 + 0.0j
        return jnp.array(mask)

    # 3. COMPILE THE BARE-METAL SYSTEM
    clock = HistoryClock()
    system = DiscreteFieldDynamicSystem.compile_bare_metal(
        topology=topology,
        generator=generator,
        start_state=start_coords,
        entity_operator_fn=raw_quantum_collapse_operator,
        clock=clock,
        expansion_depth=STEPS,
        global_field_builder=double_slit_builder
    )

    key = jax.random.PRNGKey(42)
    trajectory_history = [system.get_raw_data()]

    # 4. EXECUTE
    while True:
        curr_id = system.get_raw_data()
        print(f"\n📍 ENTITY COORD: {topology._id_to_raw[curr_id]} | Tick: {clock.current_tick}")

        try:
            steps = int(input("Enter field evolution steps (0 to exit): "))
            if steps == 0: break
        except ValueError:
            continue

        key, subkey = jax.random.split(key)
        context_kwargs = {'rng_key': subkey, 'global_params': {'steps': steps}}

        # Visualize uncollapsed wave
        system.update_field(context_kwargs)
        plot_complex_field(system.get_raw_field_data(), topology, clock.current_tick)

        # Execute observation & collapse
        system.apply_operator(context_kwargs)
        trajectory_history.append(system.get_raw_data())

    plot_trajectory(trajectory_history, topology)

if __name__ == "__main__":
    run_quantum_benchmark()