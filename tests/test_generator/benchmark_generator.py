import time
import jax
import jax.numpy as jnp
import numpy as np

# --- IMPORTS (Your Actual System) ---
from src.field_dynamic_system.core.field.algebra import VectorFieldAlgebra
from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper
from src.field_dynamic_system.core.state.discrete import VectorStateSpace, VectorState
from src.field_dynamic_system.neighbor.discrete import DiscreteTopology
from src.field_dynamic_system.generator.kernel import AbstractTransitionKernel
from src.field_dynamic_system.generator.field_genetators import DiscreteFieldGenerator
from src.field_dynamic_system.core.field.compositions import MultiplicationComposition
from src.field_dynamic_system.core.field.transform import FieldTransform


# ========================================================
# 1. REAL IMPLEMENTATIONS (No Virtual Shortcuts)
# ========================================================

class StandardDeltaTopology(DiscreteTopology):
    """
    Your standard topology using Real VectorState objects.
    """

    def __init__(self, state_space, deltas):
        super().__init__(state_space)
        self.deltas = deltas

    def compute_neighbors(self, state_val):
        # Extract tuple from VectorState object
        raw = state_val.values if hasattr(state_val, 'values') else state_val

        neighbors = []
        base = np.array(raw)
        for d in self.deltas:
            res = base + np.array(d)
            # Return raw tuple; Base class handles object wrapping/lookup
            neighbors.append(tuple(res.tolist()))
        return neighbors


class BroadcastingKernel(AbstractTransitionKernel):
    def compute_raw_batch(self, edge_indices, context_mapper):
        # Send 1.0 down every edge
        return jnp.ones((edge_indices.shape[0], 1), dtype=jnp.float32)


class IdentityTransform(FieldTransform):
    def __call__(self, vector: jnp.ndarray) -> jnp.ndarray:
        return vector  # No normalization, let values grow to verify branching

    @property
    def is_linear_map(self): return True


# ========================================================
# 2. THE AUTHENTIC BENCHMARK
# ========================================================

def run_authentic_benchmark():
    # CONFIGURATION
    STEPS = 50
    # 5-way branching expands by max 2 units per step.
    # Max reach = 50 * 2 = 100.
    # We define the Universe from -100 to +100.
    MIN_VAL, MAX_VAL = -100, 100
    N_STATES = (MAX_VAL - MIN_VAL) + 1  # 201 States

    print(f"\n==================================================")
    print(f"🚀 BENCHMARK: Real Architecture (State 0 -> Outward)")
    print(f"   Universe: [{MIN_VAL}, {MAX_VAL}] ({N_STATES} States)")
    print(f"   Steps:    {STEPS}")
    print(f"==================================================")

    # 1. SETUP REAL SPACE
    # We create the specific objects required for the walk.
    print("-> [1/5] Creating VectorStateSpace...")
    real_states = [VectorState((i,)) for i in range(MIN_VAL, MAX_VAL + 1)]
    space = VectorStateSpace(real_states, dim=1)

    # 2. SETUP TOPOLOGY
    print("-> [2/5] Building Topology (5-Way Branching)...")
    # [-2, -1, 0, 1, 2]
    deltas = [(-2,), (-1,), (0,), (1,), (2,)]
    topology = StandardDeltaTopology(space, deltas)

    # 3. SETUP FIELD (Start at 0)
    print("-> [3/5] Initializing Field (State 0 = 1.0)...")
    algebra = VectorFieldAlgebra(dim=1)

    # Mapper needs to know the universe size to allocate the GPU buffer
    mapper = DiscreteFieldMapper(space, algebra)
    mapper.sync_size(N_STATES)

    # We construct the initial pulse.
    # We find the index of State(0) dynamically.
    start_node = VectorState((0,))
    start_idx = space.get_index_of(start_node)

    # Apply Initial Condition: Field is 1.0 at State 0, 0.0 elsewhere.
    init_vec = jnp.zeros((N_STATES, 1)).at[start_idx].set(1.0)
    mapper.apply_vector(init_vec)

    # Global Field (Identity)
    global_mapper = DiscreteFieldMapper(space, algebra)
    global_mapper.sync_size(N_STATES)
    global_mapper.apply_vector(jnp.ones((N_STATES, 1)))

    # 4. PHYSICS ENGINE
    physics_engine = DiscreteFieldGenerator(
        topology=topology,
        kernel=BroadcastingKernel(),
        intrinsic_transform=IdentityTransform(),
        extrinsic_transform=IdentityTransform(),
        intrinsic_composer=MultiplicationComposition(),
        chain_composer=MultiplicationComposition(),
        global_field_mapper=global_mapper
    )

    # 5. EXECUTION
    print(f"-> [4/5] Executing {STEPS} Steps (JAX Compiled)...")
    t0 = time.time()

    # This call includes: JIT Compile + Execution + Device Transfer
    result_mapper = physics_engine.generate_multi_step(mapper, steps=STEPS, global_mapper=global_mapper)
    res_vec = result_mapper.raw_buffer.block_until_ready()

    duration = time.time() - t0

    # 7. VISUALIZATION (Corrected for Scrambled States)
    print("\n-> [6/6] Generating Plot...")
    try:
        import matplotlib.pyplot as plt

        # 1. Extract (Coordinate, Value) pairs
        data_points = []
        for i, state_obj in enumerate(space.states):
            # Extract the integer coordinate (e.g., -5) from VectorState(( -5, ))
            coord = state_obj.values[0]

            # Get the field intensity at this state's index
            intensity = float(res_vec[i][0])

            data_points.append((coord, intensity))

        # 2. Sort by Coordinate (Restore Order)
        data_points.sort(key=lambda x: x[0])

        # 3. Unzip for Plotting
        x_sorted, y_sorted = zip(*data_points)

        # 4. Plot
        plt.figure(figsize=(10, 6))
        plt.plot(x_sorted, y_sorted, label="Field Intensity (Log Scale)")
        plt.yscale("log")

        plt.title(f"5-Way Branching Propagation (Step {STEPS})")
        plt.xlabel("State Coordinate (Sorted)")
        plt.ylabel("Intensity")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.axvline(0, color='r', linestyle='--', label="Start (0)")
        plt.legend()

        plt.show()
        print("✅ Plot generated (Sorted).")

    except ImportError:
        print("⚠️ Matplotlib not found.")


if __name__ == "__main__":
    run_authentic_benchmark()