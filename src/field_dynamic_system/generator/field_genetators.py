from abc import ABC
from collections import namedtuple

import jax
import jax.numpy as jnp
from typing import Optional, Any

import numpy as np
from jax.experimental import sparse
from functools import partial

from src.field_dynamic_system.core.field.mappings import DiscreteFieldMapper
# Interfaces
from src.field_dynamic_system.generator.generator_interfaces import IDiscreteFieldGenerator
from src.field_dynamic_system.core.field.compositions import FieldComposition, AdditionComposition
from src.field_dynamic_system.generator.kernel import ElementwiseKernel, AbstractTransitionKernel
from src.field_dynamic_system.neighbor.discrete import DiscreteTopology


# 2. THE ABSTRACT BASE (Entry Point)
class DiscreteFieldGenerator(IDiscreteFieldGenerator, ABC):
    """
    Acts as an entry point.
    Future shared logic for ALL generators (validation, logging) goes here.
    """
    pass


class GenericMarkovianDiscreteFieldGenerator:
    """
    STRICT PIPELINE: Global -> Intrinsic -> Edge -> Chain -> Step -> Extrinsic

    CLEANED: Delegates all algebra/identity logic to the FieldComposition classes.
    """

    def __init__(self,
                 topology: Optional[DiscreteTopology] = None, # 1. Make Optional
                 kernel: Any = None,
                 step_composer: FieldComposition = AdditionComposition(),
                 chain_composer: Optional[FieldComposition] = None,
                 global_composer: Optional[FieldComposition] = None,
                 intrinsic_transform=None,
                 extrinsic_transform=None):

        self.topology = topology
        self.kernel = kernel

        # 2. THE FIX: Lazy/Conditional Matrix Building
        # If topology is None, we are in 'Stateless/Raw' mode.
        # We don't build the matrix here; it will be provided in generate_raw_multi_step.
        self.adjacency_matrix = None
        if self.topology is not None:
            self.adjacency_matrix = self.topology.get_adjacency_matrix(self.kernel)

        self.step_composer = step_composer
        self.chain_composer = chain_composer
        self.global_composer = global_composer
        self.intrinsic_transform = intrinsic_transform
        self.extrinsic_transform = extrinsic_transform

    def generate_multi_step(self, field_mapper, steps: int,
                            global_mapper=None):

        current_buffer = field_mapper.raw_buffer

        if global_mapper:
            global_buffer = global_mapper.raw_buffer
        else:
            global_buffer = None

        # Pre-fetch Matrix components
        mat_indices = self.adjacency_matrix.indices
        mat_weights = self.adjacency_matrix.data
        tgt_indices = mat_indices[:, 0]
        src_indices = mat_indices[:, 1]
        num_nodes = current_buffer.shape[0]
        weight_column = mat_weights.reshape(-1, 1)

        def physics_step(i, carry_state):
            prev_state = carry_state

            # --- PHASE A: EDGE OPERATIONS ---
            # 1. GATHER
            row_vals = prev_state[src_indices]

            # 2. GLOBAL COMPOSITION
            if self.global_composer and global_buffer is not None:
                global_vals = global_buffer[src_indices]
                row_vals = self.global_composer.compose(row_vals, global_vals)

            # 3. INTRINSIC TRANSFORM
            if self.intrinsic_transform:
                row_vals = self.intrinsic_transform(row_vals)

            # 4. CHAIN COMPOSITION (Kernel Physics)
            # Correctly composes the field value with the transition weight
            if self.chain_composer:
                row_vals = self.chain_composer.compose(row_vals, weight_column)
            else:
                raise ValueError("A chain_composer (e.g., MultiplicationComposition) is strictly required.")

            # --- PHASE B: AGGREGATION ---
            # 5. STEP COMPOSITION
            if hasattr(self.step_composer, 'compose_reduction'):
                aggregated_state = self.step_composer.compose_reduction(row_vals, tgt_indices, num_nodes)
            else:
                raise ValueError("step_composer must implement compose_reduction.")

            # --- PHASE C: FINAL TRANSFORM ---
            # 6. EXTRINSIC TRANSFORM
            if self.extrinsic_transform:
                final_state = self.extrinsic_transform(aggregated_state)
            else:
                final_state = aggregated_state

            return final_state

        final_buffer = jax.lax.fori_loop(0, steps, physics_step, current_buffer)

        # Wrap back into OOP Mapper
        result_mapper = type(field_mapper)(
            field_mapper.state_space,
            field_mapper.algebra,
            explicit_buffer=final_buffer
        )

        return result_mapper

    def generate_raw_multi_step(self,
                                raw_field: jnp.ndarray,
                                raw_topology: tuple,
                                steps: int,
                                raw_global_field: Optional[jnp.ndarray] = None) -> jnp.ndarray:

        tgt_indices, src_indices, mat_weights, num_nodes = raw_topology
        weight_column = mat_weights.reshape(-1, 1)

        def physics_step(i, carry_state):
            prev_state = carry_state

            # --- PHASE A: EDGE OPERATIONS ---
            # 1. GATHER
            row_vals = prev_state[src_indices]

            # 2. GLOBAL COMPOSITION
            if self.global_composer and raw_global_field is not None:
                global_vals = raw_global_field[src_indices]
                row_vals = self.global_composer.compose(row_vals, global_vals)

            # 3. INTRINSIC TRANSFORM
            if self.intrinsic_transform:
                row_vals = self.intrinsic_transform(row_vals)

            # 4. CHAIN COMPOSITION (The Kernel Physics)
            # Replaces the hardcoded `*`. Composes the field value with the edge weight.
            if self.chain_composer:
                row_vals = self.chain_composer.compose(row_vals, weight_column)
            else:
                raise ValueError(
                    "A chain_composer (e.g., MultiplicationComposition) is strictly required to apply kernel weights.")

            # --- PHASE B: AGGREGATION ---
            # 5. STEP COMPOSITION
            if hasattr(self.step_composer, 'compose_reduction'):
                aggregated_state = self.step_composer.compose_reduction(row_vals, tgt_indices, num_nodes)
            else:
                raise ValueError("step_composer must implement compose_reduction.")

            # --- PHASE C: FINAL TRANSFORM ---
            # 6. EXTRINSIC TRANSFORM
            if self.extrinsic_transform:
                final_state = self.extrinsic_transform(aggregated_state)
            else:
                final_state = aggregated_state

            return final_state

        final_buffer = jax.lax.fori_loop(0, steps, physics_step, raw_field)
        return final_buffer
