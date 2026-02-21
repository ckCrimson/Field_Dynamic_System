import jax.numpy as jnp
from typing import NamedTuple, Optional, Union, List, Any, Dict
from abc import ABC, abstractmethod


# --- 1. Data Structures ---

class InteractionContext(NamedTuple):
    """
    The 'Input Payload' for the Operator.

    Acts as the bridge between the Simulation Loop (System) and the
    Observation Logic (Operator). Being a NamedTuple makes it immutable
    and JIT-friendly (zero overhead for JAX tracing).
    """
    # Source of Randomness: Essential for Probabilistic Operators.
    # Passed as a JAX PRNGKey array.
    rng_key: Optional[jnp.ndarray] = None

    # User Action: Essential for Classical/Game Operators.
    # e.g., 0=Idle, 1=Left, 2=Right. JAX treats this as a static input or scalar.
    action_id: int = 0

    # Global Parameters: Context for the physics engine.
    # e.g., {'dt': 0.1, 'temperature': 1.5, 'gravity': 9.8}
    global_params: Optional[Dict[str, Any]] = None


# --- 2. Type Definitions ---

# A single state can be an Integer Index (discrete) or an Array (continuous/vector)
State = Union[int, float, jnp.ndarray]

# An Observation is what the UI/Screen consumes:
# It can be a single State OR a Trajectory (List of States).
Observation = Union[State, List[State]]


# --- 3. The Interface ---

class IOperator(ABC):
    """
    The 'Observer' Contract.

    Responsibility:
    1. Analyze the current System State (Field or Raw Value).
    2. Incorporate External Context (RNG, Input).
    3. Return a concrete Observation (Data) for the application layer.
    """

    @abstractmethod
    def observe(self, system_state: Any, context: InteractionContext) -> Observation:
        """
        Derives the next observation based on the system state and context.

        Args:
            system_state: A MUTABLE reference to the system container.
                          - For Field Systems: A 'FieldMapper' wrapper.
                          - For Classical Systems: A 'StateContainer' wrapper.
                          The operator MAY modify this in-place (e.g., collapsing a field).

            context:      The InteractionContext containing external signals
                          (RNG keys, User Inputs, Time).

        Returns:
            Observation:  Raw data (int, list, or array) ready for immediate
                          rendering or logging.
        """
        pass

    @property
    @abstractmethod
    def selection_strategy(self) -> Any:
        """ Defines the algorithmic strategy used to select the next state. """
        pass