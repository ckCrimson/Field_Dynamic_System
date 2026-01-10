"""
Verification of JAX Pytree Registration for Discrete Spaces.
Ensures that multiple instances do not trigger recompilation.
"""
import time
import jax
import jax.numpy as jnp
from src.field_dynamic_system.core.state import VectorState, AbstractState
from src.field_dynamic_system.core.state.discrete import VectorStateSpace, AbstractDiscreteStateSpace


def test_abstract_pytree_traffic_light():
    print("\n--- 1. Abstract Space (Traffic Light) Pytree Test ---")

    # 1. Define States
    red = AbstractState("Red", {})
    green = AbstractState("Green", {})
    yellow = AbstractState("Yellow", {})
    blue = AbstractState("Blue", {})  # Special emergency light

    # 2. Create DIFFERENT instances (simulating different entities)
    # Light A: Standard {Red, Green, Yellow}
    light_A = AbstractDiscreteStateSpace({red, green, yellow})

    # Light B: Identical logic to A, but a NEW object in memory
    light_B = AbstractDiscreteStateSpace({red, green, yellow})

    # Light C: Different logic {Red, Blue} (Emergency Mode)
    light_C = AbstractDiscreteStateSpace({red, blue})

    # 3. Define JIT Function
    # We want to see if JAX compiles this ONCE or THREE times.
    @jax.jit
    def check_is_red(space):
        # We pass an ID (e.g., 0) and see if it's valid
        # This forces JAX to look at the space structure
        # Note: Abstract logic is mostly static, so JAX traces the *values*
        return space.contains(jnp.array([0]))

    # --- Run & Measure ---

    print("1. Compiling for Light A...")
    start = time.time()
    _ = check_is_red(light_A).block_until_ready()
    t_A = time.time() - start
    print(f"   Time: {t_A:.4f}s (Compilation + Run)")

    print("2. Running for Light B (Same Logic, New Object)...")
    start = time.time()
    _ = check_is_red(light_B).block_until_ready()
    t_B = time.time() - start
    print(f"   Time: {t_B:.4f}s")

    print("3. Running for Light C (Different Logic)...")
    start = time.time()
    _ = check_is_red(light_C).block_until_ready()
    t_C = time.time() - start
    print(f"   Time: {t_C:.4f}s")

    # Verification
    # t_B should be INSTANT compared to t_A
    if t_B < t_A * 0.1:
        print("✅ SUCCESS: Light B reused Light A's compilation!")
    else:
        print("❌ FAILURE: Light B triggered recompilation.")


def test_vector_pytree_grid():
    print("\n--- 2. Vector Space (Grid) Pytree Test ---")

    # 1. Setup Vectors
    up = VectorState((0, 1))
    down = VectorState((0, -1))
    left = VectorState((-1, 0))
    right = VectorState((1, 0))

    # 2. Create 3 Spaces
    # Grid 1: Vertical Only
    grid_vert = VectorStateSpace([up, down], dim=2)

    # Grid 2: Horizontal Only (Different Matrix, Same Structure)
    grid_horiz = VectorStateSpace([left, right], dim=2)

    # Grid 3: Vertical Again (New Object)
    grid_vert_2 = VectorStateSpace([up, down], dim=2)

    # 3. Define JIT Function
    @jax.jit
    def check_origin(space):
        # Check if (0,0) is in the set (It shouldn't be, but it runs the kernel)
        origin = jnp.array([0.0, 0.0])
        return space.contains(origin)

    # --- Run & Measure ---

    print("1. Compiling for Grid Vertical...")
    start = time.time()
    _ = check_origin(grid_vert).block_until_ready()
    t_1 = time.time() - start
    print(f"   Time: {t_1:.4f}s")

    print("2. Running for Grid Horizontal (Different Data, Same Shape)...")
    # CRITICAL TEST: JAX should reuse the kernel because the SHAPE (2,2) matches
    # even though the numbers are different!
    start = time.time()
    _ = check_origin(grid_horiz).block_until_ready()
    t_2 = time.time() - start
    print(f"   Time: {t_2:.4f}s")

    print("3. Running for Grid Vertical (New Object)...")
    start = time.time()
    _ = check_origin(grid_vert_2).block_until_ready()
    t_3 = time.time() - start
    print(f"   Time: {t_3:.4f}s")

    # Verification
    if t_2 < t_1 * 0.1:
        print("✅ SUCCESS: Grid Horizontal reused compilation (Matrix Data Swapped)!")
    else:
        print(f"⚠️ NOTE: t_2 ({t_2:.4f}s) might be slower if JAX treats data change as meaningful.")

    if t_3 < t_1 * 0.1:
        print("✅ SUCCESS: New Vertical Grid reused compilation!")
    else:
        print("❌ FAILURE: New Object triggered recompilation.")


if __name__ == "__main__":
    test_abstract_pytree_traffic_light()
    test_vector_pytree_grid()