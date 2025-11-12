from src.fds.core.fds_state import State

class IntegerState(State):

    state: int

    def __post_init__(self):
        s=self.state
        object.__setattr__(self, 'state', s)

    def __str__(self) -> str:
        return str(self.state)

    def __eq__(self, other):
        return self.state == other.state

    def __hash__(self):
        return hash(self.state)

    def __lt__(self, other):
        return self.state < other.state

    def __gt__(self, other):
        return self.state > other.state

    def __le__(self, other):
        return self.state <= other.state

    def __ge__(self, other):
        return self.state >= other.state

    def __add__(self, other):
        return IntegerState( self.state + other.state )

    def __sub__(self, other):
        return IntegerState( self.state - other.state )

    def __mul__(self, other):
        return IntegerState( self.state * other )

    def __truediv__(self, other):
        return IntegerState( self.state / other )

    def __floordiv__(self, other):
        return IntegerState( self.state // other )