from abc import ABC

import jax
import jax.numpy as jnp
from typing import Optional

# Interfaces
from src.field_dynamic_system.generator.generator_interfaces import IDiscreteFieldGenerator
from src.field_dynamic_system.core.field.compositions import FieldComposition, AdditionComposition


# 2. THE ABSTRACT BASE (Entry Point)
class DiscreteFieldGenerator(IDiscreteFieldGenerator, ABC):
    """
    Acts as an entry point.
    Future shared logic for ALL generators (validation, logging) goes here.
    """
    pass

class GenericMarkovianDiscreteFieldGenerator(DiscreteFieldGenerator):
    def __init__(self,
                 topology,
                 kernel,
                 # --- 3 PHYSICS COMPOSERS ---
                 step_composer: FieldComposition = AdditionComposition(),
                 chain_composer: Optional[FieldComposition] = None,
                 global_composer: Optional[FieldComposition] = None,

                 # --- TRANSFORMS ---
                 intrinsic_transform=None,
                 extrinsic_transform=None):

        self.topology = topology
        self.kernel = kernel

        self.step_composer = step_composer
        self.chain_composer = chain_composer
        self.global_composer = global_composer

        self.intrinsic_transform = intrinsic_transform
        self.extrinsic_transform = extrinsic_transform

        self.adjacency_matrix = self.topology.adjacency_matrix

    def generate_multi_step(self, field_mapper, steps: int, global_mapper=None):
        """
        Executes the 3-Stage Physics Loop on JAX.
        """
        current_buffer = field_mapper.raw_buffer

        # Handle Global Context
        if global_mapper:
            global_buffer = global_mapper.raw_buffer
        else:
            global_buffer = None

        def physics_step(i, carry_state):
            prev_state = carry_state

            # --- STAGE A: PRE-PROCESSING ---
            processed_state = prev_state
            if self.intrinsic_transform:
                processed_state = self.intrinsic_transform(prev_state)

            # --- STAGE 1: STEP COMPOSITION (Propagation) ---
            # x_prop = A @ x_in
            propagated_state = self.adjacency_matrix @ processed_state

            # --- STAGE B: POST-PROCESSING ---
            if self.extrinsic_transform:
                propagated_state = self.extrinsic_transform(propagated_state)

            # --- STAGE 2: CHAIN COMPOSITION (Time Evolution) ---
            if self.chain_composer:
                # Custom Logic (e.g. Accumulation)
                evolved_state = self.chain_composer.compose(propagated_state, prev_state)
            else:
                # Default: Markov Logic (Replacement)
                evolved_state = propagated_state

            # --- STAGE 3: GLOBAL COMPOSITION (Bias) ---
            if self.global_composer and global_buffer is not None:
                final_state = self.global_composer.compose(evolved_state, global_buffer)
            else:
                final_state = evolved_state

            return final_state

        # Execute Loop
        final_buffer = jax.lax.fori_loop(0, steps, physics_step, current_buffer)

        # --- FIX: Create Result Mapper Correctly ---
        # We use type(field_mapper) to use the exact class of the input
        # We pass 'explicit_buffer' to the constructor to set the read-only raw_buffer
        result_mapper = type(field_mapper)(
            field_mapper.state_space,
            field_mapper.algebra,
            explicit_buffer=final_buffer
        )

        return result_mapper