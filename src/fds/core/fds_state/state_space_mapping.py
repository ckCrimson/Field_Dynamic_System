
from fds.core.fds_state.state_space import *

Sin = TypeVar('Sin', bound=State)
Sout = TypeVar('Sout', bound=State)


class StateSpaceMapping(ABC, Generic[Sin, Sout]):
    """
    Abstract mapping between state spaces.

    - For one-to-one mappings, implement map_state().
    - For one-to-many, implement map_state_to_states().
    Exactly one of them must be implemented in a subclass.
    """
    def __init__(self, mapping_reverse: Optional['StateSpaceMapping[Sout, Sin]'] = None):
        self.mapping_reverse = mapping_reverse

    def get_mapping(self, state_in: Sin, stateSpaceOut : StatSpace[Sout]) -> StatSpace[Sout]:
        """Map a single state to a state-space of output states."""
        pass


    def is_invertible(self) -> bool:
        return self.get_mapping is not None

    def reverse(self) -> "StateSpaceMapping[Sout, Sin]":
        if self.mapping_reverse is None:
            raise TypeError(f"{type(self).__name__} has no reverse mapping")
        return self.mapping_reverse

    #----------mapping over the state space --------------#
    def get_mapping_over_state_space(self, space_in: StatSpace[Sin], space_out_prototype: StatSpace[Sout],
                                     preserve_current: bool = True,
                                     ) -> StatSpace[Sout]:
        """
        Map a (finite) input space to an output space instance, using 'space_out_prototype'
        as a type/prototype. For non-finite spaces, prefer a procedural/iterator form.
        """
        # This assumes space_in is finite (or at least provides get_all_states()).
        # Document this contract clearly.
        try:
            in_states: Iterable[Sin] = space_in.get_all_states()  # finite-only
        except AttributeError:
            raise TypeError("map_space requires a finite StateSpace with get_all_states().")

        out_states: set[Sout] = set()
        mapped_current: Optional[Sout] = None
        in_current: Optional[Sin] = space_in.get_state()
        out_current_preserve: Optional[Sout] = space_out_prototype.get_state()

        for s in in_states:
            for t in self.get_mapping(s,space_out_prototype).get_all_states():
                out_states.add(t)
                if mapped_current is None and preserve_current and in_current is not None and s == in_current:
                    mapped_current = t

        if not out_states:
            # Return an empty/identity-like space of the same "kind".
            # Fallback: clone prototype without deepcopy by rebuilding.
            result = self._rebuild_like(proto_out_space=space_out_prototype, states=set())
            return result

        if preserve_current:
            if mapped_current is None:
                mapped_current = out_current_preserve if out_current_preserve in out_states else next(iter(out_states))
        else:
            mapped_current = next(iter(out_states))

        result = self._rebuild_like(proto_out_space=space_out_prototype,states= out_states, current=mapped_current)
        return result

    # ---- helpers ----

    def _rebuild_like(
            self,
            proto_out_space: StatSpace[Sout],
            states: set[Sout],
            current:  Optional[Sout] = None
    ) -> StatSpace[Sout]:
        """
        Rebuild a new space like 'proto' without using deepcopy.
        Prefer calling a finite-space constructor when possible.
        """
        # If you have a known finite implementation, use it here.
        # Example:
        # if isinstance(proto, DiscreteFiniteStateSpace):
        #     return DiscreteFiniteStateSpace(
        #         initial_state=current or next(iter(states), proto.get_state()),
        #         states=states
        #     )
        #
        # Generic fallback: rely on the proto API if it exposes build_from_states()
        new_space = type(proto_out_space)(proto_out_space.get_state())
        if hasattr(new_space, "build_from_states"):
            # type: ignore[attr-defined]
            new_space.build_from_states(set(states), current=current)  # finite contract
            return new_space
        # Last resort: return proto itself if truly no way to rebuild (documented limitation)
        return proto_out_space
