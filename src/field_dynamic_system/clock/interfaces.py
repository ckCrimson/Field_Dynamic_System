from abc import ABC, abstractmethod
from typing import Any, Dict, List


class IInternalClock(ABC):
    """
    The Temporal Ledger of a Dynamic System.
    Manages unobserved time (ticks), state changes (iterations), and historical context.
    """

    @property
    @abstractmethod
    def current_tick(self) -> int:
        """Total unobserved steps elapsed."""
        pass

    @property
    @abstractmethod
    def current_iteration(self) -> int:
        """Total number of times the state has been transitioned/observed."""
        pass

    @abstractmethod
    def tick(self, steps: int = 1) -> None:
        """Advances time by N steps without recording a state change."""
        pass

    @abstractmethod
    def record_snapshot(self, context_snapshot: Dict[str, Any]) -> None:
        """Saves the current system context to the history window."""
        pass

    @abstractmethod
    def get_history(self) -> List[Dict[str, Any]]:
        """Retrieves the window of past context snapshots."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets the clock and wipes the history."""
        pass