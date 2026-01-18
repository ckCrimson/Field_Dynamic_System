import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from src.field_dynamic_system.core.field.mappings import FieldMapper
from src.field_dynamic_system.core.field.algebra import RealFieldAlgebra, ComplexFieldAlgebra
from src.field_dynamic_system.core.field.stratergies import RealAddition, ComplexNorm
from src.field_dynamic_system.core.field.field_space_operations import FieldSpaceTransform, FieldSpaceComposition
from src.field_dynamic_system.core.state.interfaces import IDiscreteStateSpace


# Reuse Mock Space
class MockSpace(IDiscreteStateSpace):
    def __init__(self, n=10): self._n = n

    @property
    def n_states(self): return self._n

    def state_to_index(self, s): return int(s)

    def _tree_flatten(self): return ((), {"n": self._n})

    @classmethod
    def _tree_unflatten(cls, aux, _): return cls(n=aux["n"])

    # (Other abstract methods stubbed for brevity if needed, but this covers field usage)
    @property
    def num_states(self): return self._n

    @property
    def states(self): return jnp.arange(self._n)

    def get_index_of(self, s): return int(s)

    def create_subset(self, idx): return self


def test_space_transform_norm():
    print("\n=== TEST 1: Transform (Complex -> Norm -> Real) ===")
    space = MockSpace(10)
    c_alg = ComplexFieldAlgebra()
    r_alg = RealFieldAlgebra()

    # 1. Create Complex Field: Background=1j, Impulse at [0]=2j
    # Background
    f_complex = FieldMapper.define_constant_field(space, c_alg, 0.0 + 1.0j)
    # Override
    f_complex.set_value_at(0, 0.0 + 2.0j)

    # 2. Apply Norm Transform
    f_energy = FieldSpaceTransform.apply(
        source_mapper=f_complex,
        transform_op=ComplexNorm(),
        target_algebra=r_alg
    )

    # 3. Verify
    indices = jnp.arange(10)
    values = f_energy.get_fields_at(indices)

    # Check Impulse: |2j| = 2.0
    assert jnp.allclose(values[0], 2.0)
    # Check Background: |1j| = 1.0
    assert jnp.allclose(values[1:], 1.0)

    print("✅ Transform Correct: Handled both Buffer and Background")


def test_space_composition_add():
    print("\n=== TEST 2: Composition (Add Overlapping Fields) ===")
    space = MockSpace(10)
    alg = RealFieldAlgebra()

    # Field A: Impulse at [0] = 1.0
    f_a = FieldMapper.define_impulse_field(space, alg, 0)

    # Field B: Impulse at [0] = 0.5, Impulse at [1] = 0.5
    f_b = FieldMapper.define_impulse_field(space, alg, 0)
    f_b.set_value_at(0, 0.5)  # Overwrite logic check
    f_b.set_value_at(1, 0.5)

    # Compose: A + B
    f_sum = FieldSpaceComposition.apply(
        mapper_a=f_a,
        mapper_b=f_b,
        composition_op=RealAddition(),
        target_algebra=alg
    )

    vals = f_sum.get_fields_at(jnp.arange(3))

    # Index 0: 1.0 + 0.5 = 1.5 (Overlap)
    assert jnp.allclose(vals[0], 1.5)
    # Index 1: 0.0 + 0.5 = 0.5 (Unique to B)
    assert jnp.allclose(vals[1], 0.5)
    # Index 2: 0.0 + 0.0 = 0.0 (Background)
    assert jnp.allclose(vals[2], 0.0)

    print("✅ Composition Correct: Merged sparse buffers accurately")


if __name__ == "__main__":
    test_space_transform_norm()
    test_space_composition_add()