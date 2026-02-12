import time
import pytest
from src.field_dynamic_system.core.state.discrete import AbstractDiscreteStateSpace
from src.field_dynamic_system.core.state import AbstractState
from src.field_dynamic_system.core.state.transformation import AbstractStateTransformation


def test_raw_abstract_transform_path():
    print("\n=== TEST: Raw Abstract Transformation Pipeline ===")

    # 1. SETUP DATA
    N = 100_000
    print(f"Generating {N:,} raw tuple states...")

    # We have the raw list right here!
    raw_data = [(f"Group_{i % 10}", i) for i in range(N)]

    # 2. DEFINE OPERATIONS
    op_obj = lambda x: x.value[0]
    op_raw = lambda x: x[0]

    transformer = AbstractStateTransformation(
        operation=op_obj,
        target_space_class=AbstractDiscreteStateSpace,
        raw_operation=op_raw
    )

    # -------------------------------------------------------
    # PATH A: The Object Way (Baseline)
    # -------------------------------------------------------
    print("\n[Path A] Standard Object Map")
    # We must create the space to simulate object overhead
    input_space = AbstractDiscreteStateSpace.from_raw_data(raw_data, AbstractState)

    t0 = time.time()
    # Simulate the work: Iterate objects -> Extract Value -> Create New Space
    raw_results_obj = input_space.map(op_obj)
    output_space_obj = AbstractDiscreteStateSpace(raw_results_obj)
    t_obj = time.time() - t0
    print(f"  - Time: {t_obj:.4f}s")

    # -------------------------------------------------------
    # PATH B: The Raw Way (Direct List)
    # -------------------------------------------------------
    print("\n[Path B] Direct Raw Transform")
    t0 = time.time()

    # SIMPLE: Just pass the raw list we already have!
    # No .raw_states, no accessors, no hidden copies.
    # Logic: List -> List
    raw_results_raw = transformer.transform_raw(raw_data)

    # Factory Create (Wrap the result)
    output_space_raw = AbstractDiscreteStateSpace.from_raw_data(
        raw_data=raw_results_raw,
        wrapper=AbstractState
    )

    t_raw = time.time() - t0
    print(f"  - Time: {t_raw:.4f}s")

    # -------------------------------------------------------
    # VERIFICATION
    # -------------------------------------------------------
    speedup = t_obj / (t_raw + 1e-9)
    print(f"  - Speedup: {speedup:.1f}x")

    assert t_raw < t_obj

    # Logic Checks
    assert output_space_obj.num_states == 10  # Unique
    assert output_space_raw.num_states == N  # Preserved
    assert output_space_raw.get_state_by_id(0).value == "Group_0"


if __name__ == "__main__":
    test_raw_abstract_transform_path()