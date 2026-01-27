import jax.numpy as jnp
import pytest
from src.field_dynamic_system.core.state import VectorState, AbstractState
from src.field_dynamic_system.core.state.discrete import VectorStateSpace, AbstractDiscreteStateSpace
from src.field_dynamic_system.core.state.transformation import (
    DiscreteStateTransformation,
    VectorStateTransformation
)


def test_vector_to_vector_rotation():
    """Test standard geometric transformation."""
    # 1. Setup
    v1 = VectorState((1.0, 0.0))
    v2 = VectorState((2.0, 0.0))
    space = VectorStateSpace([v1, v2], dim=2)

    # 2. Operation: Rotate 90 degrees
    def op_rotate(state):
        vals = state.values if hasattr(state, 'values') else state
        vals = jnp.array(vals)
        x = vals[..., 0]
        y = vals[..., 1]
        return jnp.stack([-y, x], axis=-1)

    # 3. Transform
    # We use VectorStateTransformation for the optimization
    transformer = VectorStateTransformation(op_rotate, target_space_class=VectorStateSpace)
    new_space = transformer.transform(space)

    # 4. Verify
    assert isinstance(new_space, VectorStateSpace)
    assert new_space.num_states == 2

    # Check rotation: (1,0) -> (0,1)
    target = jnp.array([0.0, 1.0])
    assert new_space.contains(target)


def test_abstract_to_vector_embedding():
    """Test converting concepts to vectors (Embedding)."""
    # 1. Setup: Traffic Light
    red = AbstractState("Red", {})
    green = AbstractState("Green", {})
    space = AbstractDiscreteStateSpace({red, green})

    # 2. Operation: Map to 1D vector (Red=1, Green=2)
    def op_embed(state):
        val = 1.0 if state.name == "Red" else 2.0
        return VectorState((val,))

    # 3. Transform
    # CRITICAL FIX: VectorStateSpace requires 'dim'.
    # The generic DiscreteStateTransformation only passes the list of states.
    # We use a lambda to inject dim=1.
    factory = lambda states: VectorStateSpace(states, dim=1)

    transformer = DiscreteStateTransformation(op_embed, target_space_class=factory)
    new_space = transformer.transform(space)

    # 4. Verify
    assert isinstance(new_space, VectorStateSpace)
    assert new_space.dim == 1
    assert new_space.contains(jnp.array([1.0]))
    assert new_space.contains(jnp.array([2.0]))


def test_vector_to_abstract_labeling():
    """Test clustering vectors into named regions."""
    # 1. Setup: Points at 0 and 10
    s1 = VectorState((0.0,))
    s2 = VectorState((10.0,))
    space = VectorStateSpace([s1, s2], dim=1)

    # 2. Operation: Label based on value
    def op_label(state):
        val = state.values[0]
        label = "Far" if val > 5 else "Near"
        return AbstractState(label, {})

    # 3. Transform
    transformer = DiscreteStateTransformation(op_label, target_space_class=AbstractDiscreteStateSpace)
    new_space = transformer.transform(space)

    # 4. Verify
    assert isinstance(new_space, AbstractDiscreteStateSpace)
    names = {s.name for s in new_space.states}
    assert "Far" in names
    assert "Near" in names