import jax.numpy as jnp
import pytest
from src.field_dynamic_system.core.state import VectorState, AbstractState
from src.field_dynamic_system.core.state.discrete import VectorStateSpace, AbstractDiscreteStateSpace
from src.field_dynamic_system.core.state.transformation import DiscreteStateTransformation


def test_vector_to_vector_rotation():
    """Test standard geometric transformation."""
    # 1. Setup
    v1 = VectorState((1.0, 0.0))
    v2 = VectorState((2.0, 0.0))
    space = VectorStateSpace([v1, v2], dim=2)

    # 2. Operation: Rotate 90 degrees
    def op_rotate(state):
        # Handle both VectorState object and raw JAX array
        vals = state.values if hasattr(state, 'values') else state
        vals = jnp.array(vals)

        # FIX: Use [..., index] to get columns safely for both 1D and 2D arrays
        x = vals[..., 0]
        y = vals[..., 1]

        # Returns (..., 2)
        return jnp.stack([-y, x], axis=-1)

    # 3. Transform
    transformer = DiscreteStateTransformation(op_rotate, target_class=VectorStateSpace)
    new_space = transformer.transform(space)

    # 4. Verify
    assert isinstance(new_space, VectorStateSpace)
    assert new_space.num_states == 2

    # Check if (0, 1) exists (Rotation of 1,0)
    target = jnp.array([0.0, 1.0])

    # Note: Use allclose for float comparisons to avoid -0.0 vs 0.0 issues
    # But contains() usually handles exact matches.
    # If this fails on precision, we might need a fuzzy check,
    # but for simple integers (1.0, 0.0) it should be fine.
    assert new_space.contains(target)
def test_abstract_to_vector_embedding():
    """Test converting concepts to vectors (Embedding)."""
    # 1. Setup: Traffic Light
    # FIX: Add properties={}
    red = AbstractState("Red", {})
    green = AbstractState("Green", {})
    space = AbstractDiscreteStateSpace({red, green})

    # ... rest of test ...

def test_vector_to_abstract_labeling():
    """Test clustering vectors into named regions."""
    # 1. Setup: Points at 0 and 10
    s1 = VectorState((0.0,))
    s2 = VectorState((10.0,))
    space = VectorStateSpace([s1, s2], dim=1)

    # 2. Operation: Label based on value
    def op_label(state):
        val = state.values[0]
        return "Far" if val > 5 else "Near"

    # 3. Transform asking for AbstractSpace
    transformer = DiscreteStateTransformation(op_label, target_class=AbstractDiscreteStateSpace)
    new_space = transformer.transform(space)

    # 4. Verify
    assert isinstance(new_space, AbstractDiscreteStateSpace)
    names = {s.name for s in new_space.allowed_states}
    assert "Far" in names
    assert "Near" in names