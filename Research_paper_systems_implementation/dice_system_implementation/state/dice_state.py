from fds import State


class DiceState(State):

    state: float
    STATE_LOWER_BOUND: int = 1

    def __post_init__(self):
       if self.state < self.STATE_LOWER_BOUND:
           raise ValueError(f"state must be between 1 greater than {self.STATE_LOWER_BOUND}")

    def __eq__(self, other):
        return self.state == other.state
    def __abs__(self):
        return self.state
    def __hash__(self):
        return hash(self.state)
    def __repr__(self):
        return f"DiceState({self.state})"
    def __str__(self):
        return f"DiceState({self.state})"

    def __gt__(self, other):
        if other is isinstance(self, DiceState):
            return self.state > other.state
        elif other is isinstance(other, float):
            return self.state > other
        elif other is isinstance(other, int):
            return self.state > other
        else:
            raise "In valid Comparison"

    def __lt__(self, other):
        if other is isinstance(other, DiceState):
            return self.state < other.state
        elif other is isinstance(other, float):
            return self.state < other
        elif other is isinstance(other, int):
            return self.state < other
        else:
            raise "In valid Comparison"

    def __mul__(self, other):
        if other is isinstance(self, DiceState):
            return self.state * other.state
        elif other is isinstance(self, float):
            return self.state * other
        elif other is isinstance(self, int):
            return self.state * other
