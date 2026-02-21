import numpy as np

# Core & Systems
from src.field_dynamic_system.systems.static.state import DiscreteStaticStateSystem
from src.field_dynamic_system.operator.classical import ClassicalOperator
from src.field_dynamic_system.clock.window_clock import WindowedInternalClock
from src.field_dynamic_system.systems.dynamic.base import DynamicSystem

# Orchestration
from src.field_dynamic_system.orchestration.policies.keyboard import KeyboardPolicy
from src.field_dynamic_system.orchestration.runner.loop import SimulationRunner


def walker_strategy(current_state: np.ndarray, action_id: int) -> np.ndarray:
    """The pure math transition function."""
    return current_state + action_id


def render_walker(system: DynamicSystem, context: dict):
    """Custom drawing logic for this specific simulation."""
    pos = int(system.state[0])
    line = ["."] * 21
    draw_pos = max(0, min(20, 10 + pos))
    line[draw_pos] = "O"

    action_name = context.get('action_name', 'Unknown')
    iteration = system.clock.current_iteration
    print(f"[{''.join(line)}] | Action: {action_name:<5} | Pos: {pos:>2} | Iteration: {iteration}")


def test_run():
    print("==================================================")
    print("⚛️  BOOTING ORCHESTRATED DYNAMIC SYSTEM")
    print("Controls: [A/Left]=Left, [D/Right]=Right, [S/Down]=Rest, [Q]=Quit")
    print("==================================================\n")

    # 1. Assemble Physics
    state_sys = DiscreteStaticStateSystem(initial_state=np.array([0]))
    operator = ClassicalOperator(selection_strategy=walker_strategy)
    clock = WindowedInternalClock(max_history_frames=50)
    system = DynamicSystem(state_sys, operator, clock)

    # 2. Assemble Driver & Runner
    policy = KeyboardPolicy()
    runner = SimulationRunner(system, policy)

    # 3. Execute
    runner.run_blocking(render_callback=render_walker)

    # 4. Dump Data on Exit
    print("\n📜 TEMPORAL LEDGER DUMP (Last 5 frames):")
    for frame in system.clock.get_history()[-5:]:
        print(f"Tick: {frame['tick']} | Iteration: {frame['iteration']} | State: {frame['state']}")


if __name__ == "__main__":
    test_run()