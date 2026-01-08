"""
Functional Test for Discrete State Spaces.
Verifies: AbstractDiscreteStateSpace, VectorStateSpace, Union, Intersection.
"""
import jax.numpy as jnp
from src.field_dynamic_system.core.state.state import VectorState, AbstractState
from src.field_dynamic_system.core.state.discrete import AbstractDiscreteStateSpace, VectorStateSpace


def test_abstract_logic():
    print("\n--- 1. Testing Abstract Discrete Logic (Rock/Paper/Scissors) ---")

    # 1. Setup
    rock = AbstractState("Rock", {})
    paper = AbstractState("Paper", {})
    scissors = AbstractState("Scissors", {})

    space_A = AbstractDiscreteStateSpace({rock, paper})  # {R, P}
    space_B = AbstractDiscreteStateSpace({paper, scissors})  # {P, S}

    # 2. Test UNION ({R, P} U {P, S} = {R, P, S})
    union_space = space_A.union(space_B)
    assert union_space.contains(rock)
    assert union_space.contains(scissors)
    assert union_space._num_states == 3
    print("✅ Union Logic Passed")

    # 3. Test INTERSECTION ({R, P} n {P, S} = {P})
    inter_space = space_A.intersection(space_B)
    assert inter_space.contains(paper)
    assert not inter_space.contains(rock)
    assert inter_space._num_states == 1
    print("✅ Intersection Logic Passed")

    # 4. Test BATCH Contains (List Input)
    # Check [Rock, Scissors] against space_A ({R, P})
    batch = [rock, scissors]
    mask = space_A.contains(batch)  # Should be [True, False]

    assert jnp.array_equal(mask, jnp.array([True, False]))
    print("✅ Batch List Check Passed")

    # 5. Test JAX ID Contains (Hot Path)
    # space_A has {Rock, Paper}. Rock is ID 1 (alphabetical order: Paper=0, Rock=1)
    # We check ID 0 (Paper -> True), ID 1 (Rock -> True), ID 5 (Invalid -> False)
    ids = jnp.array([0, 1, 5], dtype=jnp.int32)
    mask_jax = space_A.contains(ids)

    assert jnp.array_equal(mask_jax, jnp.array([True, True, False]))
    print("✅ JAX ID Check Passed")


def test_vector_logic():
    print("\n--- 2. Testing Vector Discrete Logic (Directions) ---")

    # 1. Setup
    up = VectorState((0., 1.))
    down = VectorState((0., -1.))
    left = VectorState((-1., 0.))

    # Space Vertical = {Up, Down}
    space_vert = VectorStateSpace([up, down], dim=2)

    # 2. Test Contains (Exact Object)
    assert space_vert.contains(up)
    assert not space_vert.contains(left)

    # 3. Test Contains (JAX Array / Broadcasting)
    # Input: [Up, Left, Down] -> Expect [True, False, True]
    input_vecs = jnp.array([
        [0., 1.],  # Up
        [-1., 0.],  # Left
        [0., -1.]  # Down
    ])

    mask = space_vert.contains(input_vecs)
    assert jnp.array_equal(mask, jnp.array([True, False, True]))
    print("✅ Vector Broadcasting Passed")

    # 4. Test Set Operations
    space_single = VectorStateSpace([left], dim=2)

    # Union: {Up, Down} U {Left}
    space_all = space_vert.union(space_single)
    assert space_all.contains(left)
    assert space_all.contains(up)

    print("✅ Vector Set Logic Passed")


if __name__ == "__main__":
    test_abstract_logic()
    test_vector_logic()