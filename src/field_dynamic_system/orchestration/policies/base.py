from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

class IPolicy(ABC):
    """
    The Input Generator (The Driver).
    Observes the system state and decides the next action or context.
    """
    @abstractmethod
    def get_action(self, state: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Returns a tuple: (runner_metadata, physics_kwargs).
        runner_metadata: Controls for the loop (e.g., {'quit': True, 'action_name': 'Quit'}).
        physics_kwargs: Data specifically for the InteractionContext (e.g., {'action_id': 1}).
        """
        pass