from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Set, TypeVar, Generic, Optional, Iterable

from fds.core.fds_state.state import State
from fds.core.fds_state.state_space import StatSpace, DiscreteFiniteStatSpace

S = TypeVar("S", bound=State)


class IsAllowedState(ABC, Generic[S]):
    """
    Policy object: decides if a candidate state is 'allowed'
    relative to an initial/reference state.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params: Dict[str, Any] = params or {}

    @abstractmethod
    def is_state_allowed(self, state_initial: S, state: S) -> bool:
        """Return True if `state` is allowed with respect to `state_initial`."""
        raise NotImplementedError

class Reachable(ABC, Generic[S]):
    """
    Base for 'reachable set' providers.
    Subclasses define how to compute reachable states in one step.
    """
    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        is_allowed: Optional[IsAllowedState[S]] = None,
        from_allowed: bool = False,
    ):
        self.params: Dict[str, Any] = params or {}
        self.is_allowed: Optional[IsAllowedState[S]] = is_allowed
        self.from_allowed_only: bool = from_allowed

    @abstractmethod
    def get_reachable(
        self,
        state: S,
        within_state_space: bool = False,
        from_allowed: Optional[bool] = None,
        state_space: Optional[StatSpace] = None
    ) -> StatSpace[S]:
        """
        Return a state space of states reachable from `state` in one step.
        If `within_state_space` is True, constrain to the system's current space.
        If `from_allowed` is provided, override `self.from_allowed_only` for this call.
        """
        raise NotImplementedError

    # ---------- utility: reachable-from-allowed ----------
    def get_reachable_from_allowed(
            self,
            state: S,
            space: "StatSpace[S]",
    ) -> "StatSpace[S]":
        """
        Returns a NEW finite space of allowed destinations from 'state'.
        Fast path uses ids/masks if policy exposes is_allowed_batch.
        Slow path falls back to per-state predicate over get_all_states().
        """
        # Try to resolve source id if supported
        src_id: Optional[int] = None
        try:
            src_id = space.get_id(state)  # O(1) for DiscreteFiniteStatSpace
        except Exception:
            pass  # non-finite or non-discrete; will use slow path

        # --- FAST PATH: batch over ids/masks ---
        if src_id is not None and hasattr(self.is_allowed, "is_allowed_batch"):
            try:
                ids = space.ids_view()  # np.int32 [0..N-1]
                mask = self.is_allowed.is_allowed_batch(src_id, space, ids)  # np.bool_ [N]
                sel_ids = ids[mask.astype(bool)]
                # Materialize states ONCE at the end (outside the kernel)
                dst_states = [space.get_state_by_id(i) for i in sel_ids]
                current = state if (src_id in sel_ids) else (dst_states[0] if dst_states else None)
                return DiscreteFiniteStatSpace(dst_states, current=current,
                                               dim=getattr(space, "dimension", lambda: 1)())
            except NotImplementedError:
                pass  # policy didn't implement the fast path after all

        # --- SLOW PATH: iterate states (kept for compatibility / small N) ---
        try:
            all_states: Iterable[S] = space.get_all_states()
        except AttributeError as e:
            raise TypeError("Reachable requires a finite StateSpace with get_all_states() or ids_view().") from e

        allowed: list[S] = []
        for s in all_states:
            if self.is_allowed.is_state_allowed(state, s):
                allowed.append(s)

        current = state if any(s == state for s in allowed) else (allowed[0] if allowed else None)
        return DiscreteFiniteStatSpace(allowed, current=current, dim=getattr(space, "dimension", lambda: 1)())


# ---------- local helper (no deepcopy, no caller mutation) ----------

def _rebuild_like(proto: StatSpace[S], states: Set[S], current: Optional[S]) -> StatSpace[S]:
    """
    Rebuild a NEW space of the same concrete type as `proto` from `states`.
    Prefers a build method if available; otherwise, tries a minimal constructor.
    Never mutates `proto`.
    """
    # Try: constructor with initial state, then build_from_states
    new_space = type(proto)(proto.get_state())  # minimal ctor contract across your spaces
    if hasattr(new_space, "build_from_states"):
        # type: ignore[attr-defined]
        new_space.build_from_states(states, current=current)
        return new_space

    # If your finite spaces expose a named constructor, you can special-case here:
    # from .spaces.finite import DiscreteFiniteStateSpace
    # if isinstance(proto, DiscreteFiniteStateSpace):
    #     return DiscreteFiniteStateSpace(
    #         initial_state=current or (next(iter(states)) if states else proto.get_state()),
    #         states=states
    #     )

    # Last resort: return a minimal same-type instance (document limitation)
    return new_space
