"""
Functional Test for Core State Module.
Verifies: VectorState, AbstractState, VectorEncoding, BitMaskingEncoding.
"""
import jax.numpy as jnp
from src.field_dynamic_system.core.state.encoding import VectorEncoding, BitMaskingEncoding
from src.field_dynamic_system.core.state.encoding import VectorState , AbstractState


def test_vector_pipeline():
    print("\n--- Testing Vector Pipeline ---")

    # 1. Creation
    original = VectorState(values=(1.0, 0.5, -0.5))
    print(f"Original: {original}")

    # 2. Encoding
    encoder = VectorEncoding(dim=3)
    encoded = encoder.encode(original)

    print(f"Encoded Shape: {encoded.shape}")
    print(f"Encoded Data: {encoded}")

    # 3. Decoding
    decoded = encoder.decode(encoded)
    print(f"Decoded: {decoded}")

    # 4. Assertions
    assert isinstance(encoded, jnp.ndarray)
    assert encoded.dtype == jnp.float32
    assert decoded == original
    print("✅ Vector Pipeline Success")


def test_discrete_pipeline():
    print("\n--- Testing Discrete (Abstract) Pipeline ---")

    # 1. Define the Universe of States
    rock = AbstractState(name="Rock", properties={"solid": True})
    water = AbstractState(name="Water", properties={"fluid": True})
    fire = AbstractState(name="Fire", properties={"hot": True})

    universe = [rock, water, fire]

    # 2. Setup Encoder (The Registry)
    encoder = BitMaskingEncoding(universe)

    # 3. Test Encoding 'Water'
    target = water
    encoded = encoder.encode(target)

    print(f"Target: {target.name}")
    print(f"Encoded ID: {encoded} (Expected: 1)")

    # 4. Test Decoding
    decoded = encoder.decode(encoded)
    print(f"Decoded: {decoded.name}")

    # 5. Assertions
    # Extract the scalar value using .item()
    assert int(encoded.item()) == 1
    assert decoded == target
    print("✅ Discrete Pipeline Success")


if __name__ == "__main__":
    test_vector_pipeline()
    test_discrete_pipeline()