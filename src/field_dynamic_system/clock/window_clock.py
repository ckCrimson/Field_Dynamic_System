from typing import Any, Dict, List, Optional
from collections import deque
from src.field_dynamic_system.clock.interfaces import IInternalClock


class WindowedInternalClock(IInternalClock):
    """
    A high-performance clock that implements a Sliding Window (Ring Buffer)
    to bound memory usage during infinite continuous simulations.
    """

    def __init__(self, max_history_frames: Optional[int] = 100):
        self._tick = 0
        self._iteration = 0

        # The Window Approach: Automatically drops oldest frame when limit is reached
        self.max_history_frames = max_history_frames
        self._history = deque(maxlen=max_history_frames)

    @property
    def current_tick(self) -> int:
        return self._tick

    @property
    def current_iteration(self) -> int:
        return self._iteration

    def tick(self, steps: int = 1) -> None:
        """ Advance the background time. Zero Python overhead for bulk steps. """
        if steps < 0:
            raise ValueError("Time cannot flow backwards (steps must be >= 0).")
        self._tick += steps

    def record_snapshot(self, context_snapshot: Dict[str, Any]) -> None:
        """
        Record an observed state transition.
        Auto-injects timestamps so the system doesn't have to.
        """
        # Inject the temporal metadata
        snapshot_to_save = context_snapshot.copy()
        snapshot_to_save["tick"] = self._tick
        snapshot_to_save["iteration"] = self._iteration

        # Append to the right. If maxlen is exceeded, the oldest on the left is deleted.
        self._history.append(snapshot_to_save)
        self._iteration += 1

    def get_history(self) -> List[Dict[str, Any]]:
        """ Returns the windowed history as a standard Python list. """
        return list(self._history)

    def reset(self) -> None:
        """ Completely zeros out the clock. """
        self._tick = 0
        self._iteration = 0
        self._history.clear()