from typing import Callable, Optional
from src.field_dynamic_system.systems.dynamic.base import DynamicSystem
from src.field_dynamic_system.orchestration.policies.base import IPolicy

class SimulationRunner:
    """
    The self-contained execution loop.
    Marries a DynamicSystem with an Input Source (Policy).
    """

    def __init__(self, system: DynamicSystem, policy: IPolicy):
        self.system = system
        self.policy = policy

    def run_blocking(self,
                     max_steps: Optional[int] = None,
                     render_callback: Optional[Callable] = None) -> None:
        """
        Takes over the main thread and runs the simulation.
        """
        step = 0
        while True:
            if max_steps is not None and step >= max_steps:
                break

            # 1. Driver decides what to do (Unpack the cleanly separated data)
            runner_meta, physics_data = self.policy.get_action(self.system.state)

            if runner_meta.get("quit", False):
                break

            # 2. Time passes
            self.system.clock.tick(1)

            # 3. Physics engine applies the move (ONLY gets physics data)
            self.system.apply_operator(context_kwargs=physics_data)

            # 4. Draw the result (Merge them back together just for the UI printout)
            if render_callback:
                combined_context = {**runner_meta, **physics_data}
                render_callback(self.system, combined_context)

            step += 1