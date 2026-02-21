from abc import ABC, abstractmethod
from typing import Any, Dict

class IPolicy(ABC):
    """
    The Input Generator (The Driver).
    Observes the system state and decides the next action or context.
    """
    @abstractmethod
    def get_action(self, state: Any) -> Dict[str, Any]:
        """
        Returns the InteractionContext kwargs (e.g., {'action_id': 1}).
        Must include a 'quit': True flag if the policy wants to terminate the simulation.
        """
        pass