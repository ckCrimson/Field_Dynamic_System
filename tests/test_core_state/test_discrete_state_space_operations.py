"""
Test for the 'Map' functionality in Discrete Spaces.
"""
import jax.numpy as jnp
from src.field_dynamic_system.core.state import VectorState, AbstractState
from src.field_dynamic_system.core.state.discrete import VectorStateSpace, AbstractDiscreteStateSpace


def test_vector_map_optimization():
    print("\n--- Testing Vector Map (Batched Optimization) ---")

    # 1. Setup Space
    dirs = [
        VectorState((0, 1)),  # Up
        VectorState((0, -1)),  # Down
        VectorState((-1, 0)),  # Left
        VectorState((1, 0))   # Right
    ]
    space = VectorStateSpace(dirs, dim=2)

    # 2. Define the Operation
    # FIX: Must handle both Raw Arrays (Optimization) and Objects (Fallback)
    def calculate_magnitude(state):
        # Case A: Optimized Path (state is JAX Array from vmap)
        if hasattr(state, 'shape'):
            vec = state
        # Case B: Fallback Path (state is VectorState Object)
        else:
            # Must convert tuple -> array manually for JAX
            vec = jnp.array(state.values)

        return jnp.linalg.norm(vec, axis=-1)

    # 3. Run Map
    results = space.map(calculate_magnitude)

    print(f"Results: {results}")

    # 4. Assertions
    assert jnp.allclose(results, 1.0)
    assert results.shape == (4,)
    print("✅ Vector Map Success (Batching Worked)")


def test_abstract_map_flexibility():
    print("\n--- Testing Abstract Map (Python Flexibility) ---")

    # 1. Setup Space
    rock = AbstractState("Rock", {"density": 5.0})
    paper = AbstractState("Paper", {"density": 0.5})
    space = AbstractDiscreteStateSpace({rock, paper})

    # 2. Define Operation
    def get_density_description(state: AbstractState):
        d = state.properties["density"]
        return f"{state.name}: {'Heavy' if d > 1 else 'Light'}"

    # 3. Run Map
    results = space.map(get_density_description)

    print(f"Results: {results}")

    # 4. Assertions
    # Note: Order is not guaranteed in sets, so we check membership
    assert "Rock: Heavy" in results
    assert "Paper: Light" in results
    print("✅ Abstract Map Success")


if __name__ == "__main__":
    test_vector_map_optimization()
    test_abstract_map_flexibility()