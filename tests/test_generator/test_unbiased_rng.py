import pytest
import jax.numpy as jnp
import numpy as np
import itertools

# --- IMPORTS ---
from src.field_dynamic_system.core.field.algebra import VectorFieldAlgebra
from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper
from src.field_dynamic_system.core.state.discrete import VectorStateSpace, VectorState
from src.field_dynamic_system.neighbor.discrete import DiscreteTopology
from src.field_dynamic_system.generator.kernel import AbstractTransitionKernel
from src.field_dynamic_system.generator.field_genetators import DiscreteFieldGenerator

# ONLY STANDARD COMPONENTS
from src.field_dynamic_system.core.field.compositions import AdditionComposition
from src.field_dynamic_system.core.field.transform import FieldTransform


# ==============================================================================
# 1. PHYSICS COMPONENTS (Kernel & Normalizer)
# ==============================================================================

class UnbiasedKernel(AbstractTransitionKernel):
    """ Correct Physics: Emits 1/D probability to each neighbor. """

    def __init__(self, prob): self.p = prob

    def compute_raw_batch(self, edge_indices, context_mapper):
        return jnp.full((edge_indices.shape[0], 1), self.p, dtype=jnp.float32)


class PDFNormalizer(FieldTransform):
    """ L1 Normalization """

    def __call__(self, vector: jnp.ndarray) -> jnp.ndarray:
        total = jnp.sum(jnp.abs(vector))
        return jnp.where(total > 1e-9, vector / total, vector)

    @property
    def is_linear_map(self): return False


class RNGTopology(DiscreteTopology):
    def __init__(self, space, dims):
        super().__init__(space)
        self.deltas = np.eye(dims, dtype=int)

    def compute_neighbors(self, state_val):
        raw = state_val.values if hasattr(state_val, 'values') else state_val
        base = np.array(raw)
        return [tuple((base + d).tolist()) for d in self.deltas]


# ==============================================================================
# 2. THE CLEAN MARKOV TEST
# ==============================================================================

def test_rng_markov_automatic():
    print("\n==================================================")
    print("🎲 TEST: Unbiased Markov Chain (Automatic Time Step)")
    print("==================================================")

    D = 3

    # 1. SETUP
    grid = [VectorState(c) for c in itertools.product(range(3), repeat=D) if sum(c) <= 2]
    space = VectorStateSpace(grid, dim=D)

    algebra = VectorFieldAlgebra(dim=1)
    mapper = DiscreteFieldMapper(space, algebra)
    mapper.sync_size(len(grid))

    # Start at Origin (1.0)
    start_idx = space.get_index_of(VectorState((0, 0, 0)))
    mapper.apply_vector(jnp.zeros((len(grid), 1)).at[start_idx].set(1.0))

    # 2. GENERATOR
    rng_generator = DiscreteFieldGenerator(
        topology=RNGTopology(space, D),
        kernel=UnbiasedKernel(prob=1.0 / D),

        # TRANSFORMS
        intrinsic_transform=PDFNormalizer(),
        extrinsic_transform=PDFNormalizer(),

        # LOGIC
        # Step: Neighbors ADD their probabilities (Standard)
        step_composer=AdditionComposition(),

        # Chain: NONE (Defaults to Markov Replacement automatically)
        chain_composer=None,

        # Global: NONE (No bias)
        global_composer=None
    )

    # -----------------------------------------------------------
    # STEP 1
    # -----------------------------------------------------------
    print("\n-> Running Step 1...")
    res1 = rng_generator.generate_multi_step(mapper, steps=1)
    vec1 = res1.raw_buffer.block_until_ready()

    idx_000 = space.get_index_of(VectorState((0, 0, 0)))
    idx_100 = space.get_index_of(VectorState((1, 0, 0)))

    val_000 = float(vec1[idx_000][0])
    val_100 = float(vec1[idx_100][0])
    total_1 = float(jnp.sum(vec1))

    print(f"   Sum: {total_1:.4f}")
    print(f"   Val(0,0,0): {val_000:.4f} (Expected 0.0 - Token Moved)")
    print(f"   Val(1,0,0): {val_100:.4f} (Expected 0.3333)")

    assert abs(total_1 - 1.0) < 1e-5
    assert val_000 == 0.0  # Proves Replacement happened automatically
    assert abs(val_100 - 0.3333) < 1e-4

    print("\n✅ SUCCESS: Markov Behavior achieved without user configuration.")