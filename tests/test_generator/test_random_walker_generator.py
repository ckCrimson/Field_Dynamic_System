import jax
import jax.numpy as jnp
from jax.experimental import sparse
import numpy as np

# 1. IMPORTS
from src.field_dynamic_system.core.field.algebra import VectorFieldAlgebra
from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper
from src.field_dynamic_system.core.state.discrete import VectorStateSpace, VectorState
from src.field_dynamic_system.neighbor.discrete import DiscreteTopology
from src.field_dynamic_system.generator.kernel import AbstractTransitionKernel
from src.field_dynamic_system.generator.field_genetators import DiscreteFieldGenerator
from src.field_dynamic_system.core.field.compositions import MultiplicationComposition
from src.field_dynamic_system.core.field.transform import FieldTransform


# ========================================================
# 2. LOCAL CUSTOM IMPLEMENTATIONS
# ========================================================

class CustomDeltaTopology(DiscreteTopology):
    """
    Local implementation for Test Suite.
    """

    def __init__(self, state_space, deltas):
        super().__init__(state_space)
        self.deltas = deltas

    def compute_neighbors(self, state_val):
        neighbors = []
        if hasattr(state_val, 'values'):
            raw = state_val.values
        else:
            raw = state_val

        base = np.array(raw)
        for d in self.deltas:
            res = base + np.array(d)
            neighbors.append(tuple(res.tolist()))
        return neighbors

class UniformKernel(AbstractTransitionKernel):
    """ Returns 1.0 for every connection (Broadcasting) """

    def compute_raw_batch(self, edge_indices, context_mapper):
        return jnp.ones((edge_indices.shape[0], 1), dtype=jnp.float32)



import jax
import jax.numpy as jnp
from jax.experimental import sparse
import numpy as np

# 1. IMPORTS
from src.field_dynamic_system.core.field.algebra import VectorFieldAlgebra
from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper
from src.field_dynamic_system.core.state.discrete import VectorStateSpace, VectorState
from src.field_dynamic_system.neighbor.discrete import DiscreteTopology
from src.field_dynamic_system.generator.kernel import AbstractTransitionKernel
from src.field_dynamic_system.generator.field_genetators import DiscreteFieldGenerator
from src.field_dynamic_system.core.field.compositions import MultiplicationComposition
from src.field_dynamic_system.core.field.transform import FieldTransform


# ========================================================
# 2. LOCAL CUSTOM IMPLEMENTATIONS
# ========================================================

class CustomDeltaTopology(DiscreteTopology):
    """
    Local implementation for Test Suite.
    """

    def __init__(self, state_space, deltas):
        super().__init__(state_space)
        self.deltas = deltas

    def compute_neighbors(self, state_val):
        neighbors = []
        if hasattr(state_val, 'values'):
            raw = state_val.values
        else:
            raw = state_val

        base = np.array(raw)
        for d in self.deltas:
            res = base + np.array(d)
            neighbors.append(tuple(res.tolist()))
        return neighbors

class UniformKernel(AbstractTransitionKernel):
    """ Returns 1.0 for every connection (Broadcasting) """

    def compute_raw_batch(self, edge_indices, context_mapper):
        return jnp.ones((edge_indices.shape[0], 1), dtype=jnp.float32)


class NormalizationTransform(FieldTransform):
    """ Renormalizes node state to sum to 1 (Does not affect outgoing count) """

    def __call__(self, vector: jnp.ndarray) -> jnp.ndarray:
        total = jnp.sum(vector)
        return jnp.where(total > 0, vector / total, vector)

    @property
    def is_linear_map(self): return False


# ========================================================
# 3. THE TEST
# ========================================================

def test_unbiased_random_walker():
    print("\n==================================================")
    print("🚶 TEST: Signal Propagation (Uniform Kernel 1.0)")
    print("==================================================")

    # 1. SETUP SPACE
    raw_states = [VectorState((i,)) for i in range(5)]
    space = VectorStateSpace(raw_states, dim=1)

    # 2. SETUP TOPOLOGY
    topology = CustomDeltaTopology(space, [(0,), (1,)])

    # 3. SETUP ALGEBRA
    algebra = VectorFieldAlgebra(dim=1)

    # 4. SETUP MAPPERS
    mapper = DiscreteFieldMapper(space, algebra)
    mapper.sync_size(5)
    init_vec = jnp.zeros((5, 1)).at[0].set(1.0)
    mapper.apply_vector(init_vec)

    # Global Field
    global_mapper = DiscreteFieldMapper(space, algebra)
    global_mapper.sync_size(5)
    global_mapper.apply_vector(jnp.ones((5, 1)))

    # 5. PHYSICS ENGINE
    physics_engine = DiscreteFieldGenerator(
        topology=topology,
        kernel=UniformKernel(),  # Sends 1.0 to ALL neighbors
        intrinsic_transform=NormalizationTransform(),
        extrinsic_transform=NormalizationTransform(),
        intrinsic_composer=MultiplicationComposition(),
        chain_composer=MultiplicationComposition(),
        global_field_mapper=global_mapper
    )

    print("-> System Initialized.")

    # 6. RUN
    print("-> Running 2-Step Evolution...")
    result_mapper = physics_engine.generate_multi_step(mapper, steps=2, global_mapper=global_mapper)
    res_vec = result_mapper.raw_buffer.block_until_ready()

    # 7. VERIFY
    val_0 = float(res_vec[0][0])
    val_1 = float(res_vec[1][0])
    val_2 = float(res_vec[2][0])

    print(f"\n-> Final State (Top 3):")
    print(f"   State 0: {val_0:.4f} (Expected 1.0)")
    print(f"   State 1: {val_1:.4f} (Expected 2.0)")
    print(f"   State 2: {val_2:.4f} (Expected 1.0)")

    # Assertions Updated for Energy Multiplication Logic
    assert abs(val_0 - 1.0) < 1e-5
    assert abs(val_1 - 2.0) < 1e-5
    assert abs(val_2 - 1.0) < 1e-5

    print("\n✅ SUCCESS: Uniform Kernel Propagation verified.")


if __name__ == "__main__":
    test_unbiased_random_walker()
class NormalizationTransform(FieldTransform):
    """ Renormalizes node state to sum to 1 (Does not affect outgoing count) """

    def __call__(self, vector: jnp.ndarray) -> jnp.ndarray:
        total = jnp.sum(vector)
        return jnp.where(total > 0, vector / total, vector)

    @property
    def is_linear_map(self): return False


# ========================================================
# 3. THE TEST
# ========================================================

def test_unbiased_random_walker():
    print("\n==================================================")
    print("🚶 TEST: Signal Propagation (Uniform Kernel 1.0)")
    print("==================================================")

    # 1. SETUP SPACE
    raw_states = [VectorState((i,)) for i in range(5)]
    space = VectorStateSpace(raw_states, dim=1)

    # 2. SETUP TOPOLOGY
    topology = CustomDeltaTopology(space, [(0,), (1,)])

    # 3. SETUP ALGEBRA
    algebra = VectorFieldAlgebra(dim=1)

    # 4. SETUP MAPPERS
    mapper = DiscreteFieldMapper(space, algebra)
    mapper.sync_size(5)
    init_vec = jnp.zeros((5, 1)).at[0].set(1.0)
    mapper.apply_vector(init_vec)

    # Global Field
    global_mapper = DiscreteFieldMapper(space, algebra)
    global_mapper.sync_size(5)
    global_mapper.apply_vector(jnp.ones((5, 1)))

    # 5. PHYSICS ENGINE
    physics_engine = DiscreteFieldGenerator(
        topology=topology,
        kernel=UniformKernel(),  # Sends 1.0 to ALL neighbors
        intrinsic_transform=NormalizationTransform(),
        extrinsic_transform=NormalizationTransform(),
        intrinsic_composer=MultiplicationComposition(),
        chain_composer=MultiplicationComposition(),
        global_field_mapper=global_mapper
    )

    print("-> System Initialized.")

    # 6. RUN
    print("-> Running 2-Step Evolution...")
    result_mapper = physics_engine.generate_multi_step(mapper, steps=2, global_mapper=global_mapper)
    res_vec = result_mapper.raw_buffer.block_until_ready()

    # 7. VERIFY
    val_0 = float(res_vec[0][0])
    val_1 = float(res_vec[1][0])
    val_2 = float(res_vec[2][0])

    print(f"\n-> Final State (Top 3):")
    print(f"   State 0: {val_0:.4f} (Expected 1.0)")
    print(f"   State 1: {val_1:.4f} (Expected 2.0)")
    print(f"   State 2: {val_2:.4f} (Expected 1.0)")

    # Assertions Updated for Energy Multiplication Logic
    assert abs(val_0 - 1.0) < 1e-5
    assert abs(val_1 - 2.0) < 1e-5
    assert abs(val_2 - 1.0) < 1e-5

    print("\n✅ SUCCESS: Uniform Kernel Propagation verified.")


if __name__ == "__main__":
    test_unbiased_random_walker()