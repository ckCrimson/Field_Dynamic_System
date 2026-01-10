import time
import jax
import jax.numpy as jnp
from src.field_dynamic_system.core.state import VectorState
from src.field_dynamic_system.core.state.discrete import VectorStateSpace

# 50,000 Entities
N_ENTITIES = 50_000


def benchmark_simulation_fps():
    print(f"\n--- Simulation Loop Benchmark (N={N_ENTITIES}) ---")

    # 1. Initialize Space (One-time cost: ~7s)
    # We create the space ONCE at the start of the game
    print("Initializing State Space (The 'Loading Screen')...")
    raw_data = jax.random.uniform(jax.random.PRNGKey(0), (N_ENTITIES, 3))
    # Trick: We can inject the matrix directly to skip the slow list loop for this test
    # (In real usage, you pay the 7s load time once)
    initial_matrix = raw_data

    # 2. Define the Physics Engine (The "Update" function)
    # Move every entity slightly: pos = pos + velocity
    @jax.jit
    def update_step(current_matrix):
        velocity = jnp.array([0.01, 0.0, 0.0])  # Move X axis
        # Operations like this are instant on GPU
        return current_matrix + velocity

    # Warmup (Compile the kernel)
    state = initial_matrix
    state = update_step(state).block_until_ready()

    # 3. The Game Loop
    print("Starting Game Loop (1000 Frames)...")
    start = time.time()

    FRAMES = 1000
    for _ in range(FRAMES):
        # We pass the RAW MATRIX, not the Python Object
        state = update_step(state)
        # We assume the result stays on GPU (no .tolist())

    # Force synchronization at the end to measure true time
    state.block_until_ready()

    total_time = time.time() - start
    fps = FRAMES / total_time

    print(f"Total Time:      {total_time:.4f}s")
    print(f"⚡ True FPS:     {fps:.1f} FPS")
    print(f"Time per Frame:  {total_time / FRAMES * 1000:.4f} ms")


if __name__ == "__main__":
    benchmark_simulation_fps()