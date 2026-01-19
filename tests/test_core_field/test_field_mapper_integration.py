import jax.numpy as jnp
from jax import config

# Enable 64-bit precision to match your system
config.update("jax_enable_x64", True)

# --- Imports from your System ---
from src.field_dynamic_system.core import VectorState
from src.field_dynamic_system.core.state.discrete import AbstractDiscreteStateSpace, VectorStateSpace
from src.field_dynamic_system.core.state.interfaces import StateSpace
from src.field_dynamic_system.core.field.mappings import FieldMapper, DiscreteFieldMapper, ContinuousFieldMapper
from src.field_dynamic_system.core.field.algebra import RealFieldAlgebra


# --- Minimal Hypercube Implementation for Testing ---
class HypercubeSpace(StateSpace):
    """
    Continuous space defined by simple Min/Max bounds.
    """

    def __init__(self, low, high, dim=1):
        self.low = jnp.array(low)
        self.high = jnp.array(high)
        self.dim = dim

    # Required by StateSpace interface
    def get_index_of(self, state):
        raise NotImplementedError("Continuous spaces do not support discrete indexing.")

    @property
    def num_states(self):
        return float('inf')  # Infinite states


# =========================================================================
# TEST CASE 1: Discrete Field Mapper (Abstract & Vector)
# =========================================================================
def test_discrete_mapper_integration():
    print("\n=== TEST 1: Discrete Mapper (Abstract & Vector) ===")
    algebra = RealFieldAlgebra()

    # --- Part A: Abstract Space (Strings) ---
    print("  -> Subtest A: Abstract Space (Strings)")
    abstract_states = ["Red", "Green", "Blue"]
    abs_space = AbstractDiscreteStateSpace(abstract_states)

    # Initialize with Constant 1.0
    f_abs = DiscreteFieldMapper.constant(abs_space, algebra, 1.0)

    # Modify "Green" -> 5.0
    f_abs.set_value_at("Green", 5.0)

    # Verify
    # "Green" should be 5.0
    res_green = f_abs.get_fields_at("Green")[0]
    assert jnp.allclose(res_green.value, 5.0)

    # "Red" should remain 1.0
    res_red = f_abs.get_fields_at("Red")[0]
    assert jnp.allclose(res_red.value, 1.0)

    # --- Part B: Vector Space (Geometry) ---
    print("  -> Subtest B: Vector Space (Geometry)")
    v_states = [VectorState((0.,)), VectorState((1.,))]
    vec_space = VectorStateSpace(v_states, dim=1)

    # Initialize Empty (Zero)
    f_vec = DiscreteFieldMapper(vec_space, algebra)

    # Set Value at (1.,)
    target = VectorState((1.,))
    f_vec.set_value_at(target, 100.0)

    # 1. Query by State Object
    res_obj = f_vec.get_fields_at(target)[0]
    assert jnp.allclose(res_obj.value, 100.0)

    # 2. Query by Raw Index (Internal Optimization Check)
    # The space sorts states. (0,) is likely index 0, (1,) is index 1.
    idx = vec_space.get_index_of(target)
    res_idx = f_vec.get_fields_at(idx)[0]
    assert jnp.allclose(res_idx.value, 100.0)

    print("✅ Discrete Mapper Success")


# =========================================================================
# TEST CASE 2: Continuous Mapper with HyperCube
# =========================================================================
def test_continuous_mapper_hypercube():
    print("\n=== TEST 2: Continuous Mapper (Hypercube) ===")
    algebra = RealFieldAlgebra()

    # 1. Define Space: 1D Line from 0 to 10
    cube_space = HypercubeSpace(low=0.0, high=10.0, dim=1)

    # 2. Define Background Function: f(x) = x^2
    def quadratic_bg(state_values):
        # Input state_values shape: (N, Dim)
        # Return shape: (N, Dim)
        return jnp.square(state_values)

    f_cont = ContinuousFieldMapper(cube_space, algebra, bg_func=quadratic_bg)

    # 3. Set an "Anomaly" (Sparse Override)
    # At x=2.0, normally 4.0. We overwrite it to -10.0
    anomaly_state = VectorState((2.0,))
    f_cont.set_value_at(anomaly_state, -10.0)

    # 4. Verification

    # Check Background Logic (e.g., at x=3.0 -> 9.0)
    normal_state = VectorState((3.0,))
    res_normal = f_cont.get_fields_at(normal_state)[0]
    assert jnp.allclose(res_normal.value, 9.0)

    # Check Sparse Override (at x=2.0 -> -10.0)
    res_anomaly = f_cont.get_fields_at(anomaly_state)[0]
    assert jnp.allclose(res_anomaly.value, -10.0)

    print("✅ Continuous Mapper Success")


# =========================================================================
# TEST CASE 3: Factory Instantiation
# =========================================================================
def test_field_mapper_factory():
    print("\n=== TEST 3: Factory Instantiation ===")
    algebra = RealFieldAlgebra()

    # Case A: Discrete Space -> DiscreteFieldMapper
    d_space = AbstractDiscreteStateSpace(["A", "B"])
    f_discrete = FieldMapper(d_space, algebra)

    print(f"  -> Input: AbstractDiscreteStateSpace | Output: {type(f_discrete).__name__}")
    assert isinstance(f_discrete, DiscreteFieldMapper)

    # Case B: Continuous Space -> ContinuousFieldMapper
    c_space = HypercubeSpace(low=0, high=1, dim=1)
    f_continuous = FieldMapper(c_space, algebra)

    print(f"  -> Input: HypercubeSpace           | Output: {type(f_continuous).__name__}")
    assert isinstance(f_continuous, ContinuousFieldMapper)

    print("✅ Factory Logic Success")


def test_scene_1_set_value_dynamics():
    print("\n🎬 Scene 1: set_value_at Dynamics")
    algebra = RealFieldAlgebra()

    # Setup: Space with ONE existing state
    s1 = VectorState((0, 0, 0))
    space = VectorStateSpace([s1], dim=3)
    mapper = DiscreteFieldMapper(space, algebra)

    # --- Case A: State IS Present ---
    print("   [Action] Setting value for EXISTING state...")
    mapper.set_value_at(s1, 5.0)

    val_a = mapper.get_fields_at(s1)[0].value
    assert jnp.allclose(val_a, 5.0)
    print("   ✅ Case A Passed: Existing state updated.")

    # --- Case B: State is NOT Present ---
    print("   [Action] Setting value for NEW state (expecting growth)...")
    s_new = VectorState((1, 1, 1))

    # Pre-check: Space size should be 1
    assert space.num_states == 1

    # Action: Set value for unknown state
    mapper.set_value_at(s_new, 10.0)

    # Post-check: Space size should be 2 (Dynamic Growth)
    assert space.num_states == 2

    # Verify value was set correctly
    val_b = mapper.get_fields_at(s_new)[0].value
    assert jnp.allclose(val_b, 10.0)
    print(f"   ✅ Case B Passed: Space grew to {space.num_states} states, value set.")


# =========================================================================
# SCENE 2: get_fields_at (Existence vs Background)
# =========================================================================
def test_scene_2_get_fields_existence():
    print("\n🎬 Scene 2: get_fields_at Existence")
    algebra = RealFieldAlgebra()

    # Setup: Space with known states
    s1 = VectorState((10, 0, 0))
    space = VectorStateSpace([s1], dim=3)

    # Initialize mapper with explicit background logic (Explicitly 0.0)
    mapper = DiscreteFieldMapper(space, algebra)

    # --- Case A: State IS Present ---
    print("   [Action] Getting value for PRESENT state...")
    mapper.set_value_at(s1, 1.0)  # Set it to 1.0

    result_a = mapper.get_fields_at(s1)[0]
    assert jnp.allclose(result_a.value, 1.0)
    print("   ✅ Case A Passed: Returned correct value.")

    # --- Case B: State is NOT Present ---
    # Requirement: "It should return None" (Interpreted as Background/Empty for JAX)
    print("   [Action] Getting value for MISSING state...")
    s_missing = VectorState((99, 99, 99))

    # Note: In current logic, get_fields_at triggers register_states (Auto-Add).
    # If we want pure "Get without Add", we check the space first.

    # Let's verify what happens.
    # Current behavior: Adds state, returns 0.0 (Background).
    result_b = mapper.get_fields_at(s_missing)[0]

    # Assertion 1: Returns Background Value (0.0), NOT random memory
    assert jnp.allclose(result_b.value, 0.0)

    # Assertion 2 (Behavior Verification): Did it add the state?
    # Currently: YES.
    is_in_space = space.get_index_of(s_missing) != -1
    print(f"   ℹ️ System Behavior: Missing state was auto-added? {is_in_space}")
    print(f"   ℹ️ Returned Value: {result_b.value} (Background)")

    print("   ✅ Case B Passed: Returned safe background value.")


if __name__ == "__main__":
    test_discrete_mapper_integration()
    test_continuous_mapper_hypercube()
    test_field_mapper_factory()
    test_scene_1_set_value_dynamics()
    test_scene_2_get_fields_existence()