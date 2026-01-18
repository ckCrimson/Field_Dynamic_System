import cProfile
import pstats
from src.field_dynamic_system.core.state import AbstractDiscreteStateSpace, AbstractState
from src.field_dynamic_system.core.state.transformation import DiscreteStateTransformation

# 1. Setup Data (Replicate your benchmark setup)
states = {AbstractState(f"State_{i}", {}) for i in range(5000)}
space = AbstractDiscreteStateSpace(states)

def dummy_op(state):
    return AbstractState(state.name + "_new", {})

transform = DiscreteStateTransformation(dummy_op, AbstractDiscreteStateSpace)

# 2. Run Profiler
print("Profiling Map Operation...")
with cProfile.Profile() as pr:
    # This is the line we are testing
    new_space = transform.transform(space)
# 3. Print Results
stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)  # Sort by 'Internal Time' (time spent strictly inside the function)
stats.print_stats(15)  # Show top 15 rows