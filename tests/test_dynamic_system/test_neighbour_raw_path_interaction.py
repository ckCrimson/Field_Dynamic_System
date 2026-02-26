import functools
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from typing import Any, Dict, Optional, NamedTuple
from dataclasses import dataclass

from src.field_dynamic_system.clock.window_clock import WindowedInternalClock, HistoryClock
from src.field_dynamic_system.operator import InteractionContext
from src.field_dynamic_system.systems.dynamic.topology import DiscreteTopologyDynamicSystem


# ==========================================
# 0. CORE DEFINITIONS
# ==========================================






# ==========================================
# 1. UNBOUNDED BFS TOPOLOGY & PHYSICS
# ==========================================

# We tell JAX to treat 'steps' as a static integer so it can dynamically size the grid
@functools.partial(jax.jit, static_argnums=(1,))
def jax_bfs_reachability(current_pos: jnp.ndarray, steps: int, key: jnp.ndarray):
    """
    1. Creates a local dynamic grid large enough to hold the frontier.
    2. Uses 2D convolution (shift-and-add) to expand the BFS frontier.
    3. Builds the probability state space and samples a final destination.
    """
    if steps == 0:
        return current_pos, 1  # Return current pos and 1 reachable state

    # Create the bounded sub-matrix for this specific jump
    grid_size = 2 * steps + 1
    center = steps

    # Initialize the local state space (1.0 at the center)
    grid = jnp.zeros((grid_size, grid_size), dtype=jnp.float32)
    grid = grid.at[center, center].set(1.0)

    # BFS Expansion Loop (Mathematically identical to matrix multiplication)
    def expand_frontier(i, g):
        up = jnp.roll(g, shift=-1, axis=0).at[-1, :].set(0)
        down = jnp.roll(g, shift=1, axis=0).at[0, :].set(0)
        left = jnp.roll(g, shift=-1, axis=1).at[:, -1].set(0)
        right = jnp.roll(g, shift=1, axis=1).at[:, 0].set(0)
        return up + down + left + right

    # Execute the BFS spread
    reachability_grid = jax.lax.fori_loop(0, steps, expand_frontier, grid)

    # Flatten the state space to evaluate valid frontier states
    flat_grid = reachability_grid.flatten()
    valid_mask = flat_grid > 0
    total_reachable_states = jnp.sum(valid_mask)

    # Mask out unreachable states with -inf for categorical sampling
    logits = jnp.where(valid_mask, flat_grid, -1e9)

    # The Operator chooses a state from the generated frontier
    sampled_flat_idx = jax.random.categorical(key, logits)

    # Convert local index back to absolute global coordinates
    local_y = sampled_flat_idx // grid_size
    local_x = sampled_flat_idx % grid_size

    offset_x = local_x - center
    offset_y = local_y - center

    final_pos = current_pos + jnp.array([offset_x, offset_y], dtype=jnp.int32)

    return final_pos, total_reachable_states



# ==========================================
# 2. DYNAMIC SYSTEM
# ==========================================


# ==========================================
# 3. PLOTTING LOGIC
# ==========================================

def plot_trajectory(history_matrix):
    if len(history_matrix) < 2:

        print("Not enough data to plot.")
        return

    x = history_matrix[:, 0]
    y = history_matrix[:, 1]

    plt.figure(figsize=(10, 8))
    plt.plot(x, y, linestyle='--', color='gray', alpha=0.5, label='Trajectory')

    u = np.diff(x)
    v = np.diff(y)
    pos_x = x[:-1] + u / 2
    pos_y = y[:-1] + v / 2
    norm = np.sqrt(u ** 2 + v ** 2)

    mask = norm > 0
    plt.quiver(pos_x[mask], pos_y[mask], u[mask], v[mask],
               angles='xy', scale_units='xy', scale=1,
               color='blue', width=0.003, headwidth=4, label='Transition')

    plt.scatter(x[0], y[0], c='green', s=150, zorder=5, edgecolors='black', label='Start')
    plt.scatter(x[-1], y[-1], c='red', s=150, zorder=5, edgecolors='black', label='End')

    for i in range(len(x)):
        plt.text(x[i], y[i], str(i), fontsize=8, color='black', ha='right', va='bottom')

    plt.title("True BFS Multi-Step Reachability Walk")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.legend()
    plt.axis('equal')

    filename = "bfs_random_walk.png"
    plt.savefig(filename)
    print(f"\n✅ Plot saved as '{filename}'")
    plt.show()


# ==========================================
# 4. EXECUTION LOOP
# ==========================================

def raw_bfs_physics_handler(state_raw: Any, context: InteractionContext) -> Any:
    steps = context.global_params.get('steps', 1)
    key = context.rng_key

    # Call the JIT function and extract final position and metric
    final_pos, reachable_count = jax_bfs_reachability(state_raw, steps, key)

    # Print the metric directly from the Python wrapper!
    print(f"   🔍 Explored Frontier: Found {reachable_count} unique accessible states.")

    return final_pos


# ==========================================
# 4. EXECUTION LOOP
# ==========================================

def run_bfs_benchmark():
    print("--- UNBOUNDED WALKER: TRUE BFS FRONTIER EXPANSION ---")

    raw_start_pos = jnp.array((0, 0), dtype=jnp.int32)
    clock = HistoryClock()

    system = DiscreteTopologyDynamicSystem.from_raw_data(
        raw_initial_state=raw_start_pos,
        raw_topology_data=None,
        raw_operator_fn=raw_bfs_physics_handler,
        clock=clock
    )

    key = jax.random.PRNGKey(42)

    # Track the trajectory locally
    trajectory_history = []

    while True:
        curr_pos = system.get_raw_data()

        # Log the state at the start of every tick
        trajectory_history.append(np.array(curr_pos))

        print(f"\n📍 COORD: ({int(curr_pos[0])}, {int(curr_pos[1])}) | Tick: {clock.current_tick}")

        try:
            user_input = input("Enter steps to generate BFS frontier (0 to plot & exit): ")
            steps = int(user_input)
        except ValueError:
            continue

        if steps == 0:
            break

        print(f"   ⚙️  Compiling/Executing Local Convolution Matrix for {steps} steps...")
        key, subkey = jax.random.split(key)

        # The physics handler will now handle printing the frontier metric
        system.apply_operator({
            'rng_key': subkey,
            'global_params': {'steps': steps}
        })

    print("\nPreparing trajectory data...")
    # Convert our tracked list into the matrix format plot_trajectory expects
    history_matrix = np.array(trajectory_history)
    plot_trajectory(history_matrix)


if __name__ == "__main__":
    run_bfs_benchmark()