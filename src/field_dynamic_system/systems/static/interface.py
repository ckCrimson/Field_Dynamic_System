from abc import ABC, abstractmethod
from typing import Any

class ISystem(ABC):
    """
    The Universal Contract for all FDS Components.
    Enforces that every system can be reset and queried for its raw data,
    without dictating how that data is stored internally.
    """

    @abstractmethod
    def reset(self) -> None:
        """ Forces the system to return to its initial/default state. """
        pass

    @abstractmethod
    def get_raw_data(self) -> Any:
        """
        Returns the minimal physical data.
        (e.g., Topology adjacency, Field arrays, or Continuous parameters).
        """
        pass

    @abstractmethod
    def get_raw_state_space(self) -> Any:
        """ Returns the spatial coordinates or bounds of the system. """
        pass