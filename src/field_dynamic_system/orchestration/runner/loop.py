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

            # 1. Driver decides what to do
            context_kwargs = self.policy.get_action(self.system.state)

            if context_kwargs.get("quit", False):
                break

            # 2. Time passes
            self.system.clock.tick(1)

            # 3. Physics engine applies the move
            self.system.apply_operator(context_kwargs=context_kwargs)

            # 4. Draw the result
            if render_callback:
                render_callback(self.system, context_kwargs)

            step += 1