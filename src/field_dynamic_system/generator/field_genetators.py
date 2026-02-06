from abc import ABC
from collections import namedtuple

import jax
import jax.numpy as jnp
from typing import Optional

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
                 topology: DiscreteTopology,
                 kernel,
                 # --- COMPOSERS ---
                 step_composer: FieldComposition = AdditionComposition(),
                 chain_composer: Optional[FieldComposition] = None,
                 global_composer: Optional[FieldComposition] = None,
                 # --- TRANSFORMS ---
                 intrinsic_transform=None,
                 extrinsic_transform=None):

        self.topology = topology
        self.kernel = kernel

        # 1. BUILD MATRIX
        self.adjacency_matrix = self.topology.get_adjacency_matrix(self.kernel)

        self.step_composer = step_composer
        self.chain_composer = chain_composer
        self.global_composer = global_composer
        self.intrinsic_transform = intrinsic_transform
        self.extrinsic_transform = extrinsic_transform

    def generate_multi_step(self, field_mapper: DiscreteFieldMapper, steps: int,
                            global_mapper: Optional[DiscreteFieldMapper] = None):

        current_buffer = field_mapper.raw_buffer

        # Prepare Global Buffer
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

        def physics_step(i, carry_state):
            prev_state = carry_state

            # --- PHASE A: EDGE OPERATIONS ---

            # 1. GATHER: Pull value from Source
            row_vals = prev_state[src_indices]

            # 2. GLOBAL COMPOSITION
            # Logic: "Interact Edge Signal with Global Field"
            # If Global Field is missing/zero, the 'global_composer' handles it internally.
            if self.global_composer and global_buffer is not None:
                global_vals = global_buffer[src_indices]
                row_vals = self.global_composer.compose(row_vals, global_vals)

            # 3. INTRINSIC TRANSFORM
            if self.intrinsic_transform:
                row_vals = self.intrinsic_transform(row_vals)

            # 4. WEIGHT APPLICATION (Kernel Physics)
            row_vals = row_vals * mat_weights.reshape(-1, 1)

            # 5. CHAIN COMPOSITION
            # Logic: "Interact Edge Signal with Target's Past"
            # If Target History is zero, 'chain_composer' handles it (e.g. Identity swapping).
            if self.chain_composer:
                target_prev_vals = prev_state[tgt_indices]
                row_vals = self.chain_composer.compose(row_vals, target_prev_vals)

            # --- PHASE B: AGGREGATION ---

            # 6. STEP COMPOSITION (Reduction)
            # Logic: "Collapse all incoming edges to one node value"
            # We must use compose_reduction to ensure we use Sum/Prod/Max correctly.
            if hasattr(self.step_composer, 'compose_reduction'):
                aggregated_state = self.step_composer.compose_reduction(row_vals, tgt_indices, num_nodes)
            else:
                # Fallback to Summation if not specified
                aggregated_state = jax.ops.segment_sum(row_vals, tgt_indices, num_segments=num_nodes)

            # --- PHASE C: FINAL TRANSFORM ---

            # 7. EXTRINSIC TRANSFORM
            if self.extrinsic_transform:
                final_state = self.extrinsic_transform(aggregated_state)
            else:
                final_state = aggregated_state

            return final_state

        final_buffer = jax.lax.fori_loop(0, steps, physics_step, current_buffer)

        result_mapper = type(field_mapper)(
            field_mapper.state_space,
            field_mapper.algebra,
            explicit_buffer=final_buffer
        )

        return result_mapper
