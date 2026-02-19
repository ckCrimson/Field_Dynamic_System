import jax
import jax.numpy as jnp
from typing import Any, Callable

from src.field_dynamic_system.generator.generator_interfaces import IContinuousFieldGenerator


# Assuming IContinuousFieldGenerator is imported from your core interfaces
class GaussianWavePacketGenerator(IContinuousFieldGenerator):
    """
    Simulates a 2D Gaussian Wave Packet moving through continuous space.
    """

    def __init__(self, initial_center=(0.0, 0.0), velocity=(1.0, 0.5), spread=2.0, amplitude=1.0):
        # We store the initial setup to pass to the Orchestrator
        self.initial_center = jnp.array(initial_center, dtype=jnp.float32)
        self.velocity = jnp.array(velocity, dtype=jnp.float32)
        self.spread = jnp.array(spread, dtype=jnp.float32)
        self.amplitude = jnp.array(amplitude, dtype=jnp.float32)
        super().__init__()

    def get_initial_parameters(self) -> Any:
        """
        1. THE STATE: Returns the PyTree (Dictionary) of JAX parameters.
        This is what the Orchestrator will store as `current_raw_data`.
        """
        return {
            "center": self.initial_center,
            "velocity": self.velocity,
            "spread": self.spread,
            "amplitude": self.amplitude
        }

    def get_base_function(self) -> Callable[[jnp.ndarray, dict], jnp.ndarray]:
        """
        2. THE MATH: Returns the pure functional equation f(coords, params).
        The ContinuousFieldMapper will use this to evaluate the field at any X,Y.
        """

        def gaussian_equation(coords: jnp.ndarray, params: dict) -> jnp.ndarray:
            # coords shape: (N, 2) representing N arbitrary points in space
            # params["center"] shape: (2,)

            # 1. Calculate the distance from each coordinate to the wave center
            diff = coords - params["center"]

            # 2. Square the distance: (x - cx)^2 + (y - cy)^2
            dist_sq = jnp.sum(diff ** 2, axis=-1, keepdims=True)

            # 3. Apply the Gaussian bell curve formula
            exponent = -dist_sq / (2.0 * params["spread"] ** 2)
            values = params["amplitude"] * jnp.exp(exponent)

            return values

        return gaussian_equation

    def generate_continuous_step(self, params: dict, steps: int, dt: float = 0.1, **kwargs) -> dict:
        """
        3. THE EVOLUTION: The JAX-compiled physics loop.
        Updates the parameters (moves the center) over time.
        """

        def physics_step(i, current_params):
            # Kinematics: New Position = Old Position + (Velocity * Time)
            new_center = current_params["center"] + (current_params["velocity"] * dt)

            # Return the updated dictionary of parameters.
            # JAX requires the output dictionary to have the exact same keys and shapes!
            return {
                "center": new_center,
                "velocity": current_params["velocity"],  # Velocity is constant
                "spread": current_params["spread"],  # Spread is constant
                "amplitude": current_params["amplitude"]  # Amplitude is constant
            }

        # Run the fast JAX loop to step the parameters forward in time
        final_params = jax.lax.fori_loop(0, steps, physics_step, params)
        return final_params