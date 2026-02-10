import time
import pytest
import numpy as np
from typing import Any, Sequence, Callable, Type

from src.field_dynamic_system.core.state.discrete import AbstractDiscreteStateSpace, LazyStateProxy
from src.field_dynamic_system.core.state import AbstractState
from src.field_dynamic_system.core.state.transformation import DiscreteStateTransformation, AbstractStateTransformation
from src.field_dynamic_system.core.state.interfaces import IDiscreteStateSpace


def test_raw_abstract_transform_path():
    print("\n=== TEST: Raw Abstract Transformation Pipeline ===")

    # 1. SETUP DATA
    N = 100_000
    print(f"Generating {N:,} raw tuple states...")
    raw_data = [(f"Group_{i % 10}", i) for i in range(N)]

    # 2. DEFINE OPERATIONS
    # Op for Objects (Path A)
    op_obj = lambda x: x.value[0]
    # Op for Raw Data (Path B)
    op_raw = lambda x: x[0]

    transformer = AbstractStateTransformation(
        operation=op_obj,
        target_space_class=AbstractDiscreteStateSpace,
        raw_operation=op_raw
    )

    # -------------------------------------------------------
    # PATH A: The "Old Slow Way" (Simulated)
    # -------------------------------------------------------
    print("\n[Path A] Standard Object Map + Init")

    # Init Input Space
    input_space = AbstractDiscreteStateSpace.from_raw_data(raw_data, AbstractState)

    t0 = time.time()

    # FORCE SLOW PATH: We manually map objects and use standard constructor
    raw_results_obj = input_space.map(op_obj)  # Iterates 100k objects
    output_space_obj = AbstractDiscreteStateSpace(raw_results_obj)  # Sorts + Unique-ifies

    t_obj = time.time() - t0
    print(f"  - Time: {t_obj:.4f}s")

    # -------------------------------------------------------
    # PATH B: The "New Fast Way" (Raw Pipe)
    # -------------------------------------------------------
    print("\n[Path B] Raw Pipe + Factory")
    t0 = time.time()

    # 1. GET RAW SOURCE (Zero Copy)
    # CRITICAL FIX: We grab the raw LIST directly.
    # Do NOT call .raw_states, because that triggers an expensive np.array() conversion.
    if isinstance(input_space._idx_to_state, LazyStateProxy):
        raw_source = input_space._idx_to_state.raw_data
    else:
        raw_source = input_space.raw_states

    # 2. Raw Transform (Iterates List -> Returns List)
    raw_results_raw = transformer.transform_raw(raw_source)

    # 3. Factory Create (Zero Copy)
    output_space_raw = AbstractDiscreteStateSpace.from_raw_data(
        raw_data=raw_results_raw,
        wrapper=AbstractState
    )

    t_raw = time.time() - t0
    print(f"  - Time: {t_raw:.4f}s")

    # -------------------------------------------------------
    # VERIFICATION
    # -------------------------------------------------------
    print("\n[Verification]")

    # 1. Speedup
    speedup = t_obj / (t_raw + 1e-9)
    print(f"  - Speedup: {speedup:.1f}x")

    # NOW this assertion will pass.
    # Path B (List Comp) is faster than Path A (Object Access + Set Construction)
    assert t_raw < t_obj, "Raw path should be faster"

    # 2. Behavioral Difference
    count_obj = output_space_obj.num_states
    print(f"  - Path A Count (Unique): {count_obj}")

    count_raw = output_space_raw.num_states
    print(f"  - Path B Count (Preserved): {count_raw}")

    assert count_obj == 10, "Standard Constructor should reduce to unique sets"
    assert count_raw == N, "Raw Factory should preserve agent count"

    # 3. Data Integrity
    first_val = output_space_raw.get_state_by_id(0).value
    assert first_val == "Group_0"


if __name__ == "__main__":
    test_raw_abstract_transform_path()