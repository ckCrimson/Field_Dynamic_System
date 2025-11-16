import random
from typing import Dict

import numpy as np

from fds.affecting.affecting_systems_framework.correlated_state import CorrelatedState
from fds.affecting.affecting_systems_framework.grouped_operators import AffectedSystemsOperator
from one_dim_random_walker.core.real_fields_for_walker.one_dim_walker_real_field import RealField
from one_dim_random_walker.correllation_affected_no_collision.correlation_mapping import CorrerlatedAffectingRWMapping
from one_dim_random_walker.dynamic_systems.dynamic_system import OneDimWalkerFieldDynamicSystem


class CorrelationGroupOperators(AffectedSystemsOperator):
    def __init__(self, affecting_group: Dict[str,OneDimWalkerFieldDynamicSystem], mapping : CorrerlatedAffectingRWMapping=None):
        if mapping is None:
            mapping = CorrerlatedAffectingRWMapping()
        super().__init__(affecting_group, mapping)

    def get_next_state(self, field: RealField) -> CorrelatedState:
        states = field.state_space
        ids = states.ids_view()
        weight_list = []
        for id in ids:
            state = states.get_state_by_id(int(id))
            weight_list.append(field.get_field(state).data.value)
        return  states.get_state_by_id(self._sample_rv_by_weight(ids,weight_list) )


    def _find_closest_to_expectation(self,rvs, weights):
        """
        Find the RV closest to the expectation value.

        Parameters:
        rvs (list): List of random variable values
        weights (list): List of un-normalized weights for each RV

        Returns:
        tuple: (expectation_value, closest_rv, index_of_closest)
        """
        # Convert to numpy arrays for easier computation
        rvs = np.array(rvs)
        weights = np.array(weights)

        # Normalize the weights
        normalized_weights = weights / np.sum(weights)

        # Calculate expectation value
        expectation = np.sum(rvs * normalized_weights)

        # Calculate distances from expectation
        distances = np.abs(rvs - expectation)

        # Find the minimum distance
        min_distance = np.min(distances)

        # Find all indices with minimum distance
        min_indices = np.where(distances == min_distance)[0]

        if len(min_indices) == 1:
            # Single closest RV
            closest_index = min_indices[0]
        else:
            # Multiple RVs at same distance, choose the smaller one
            closest_rvs = rvs[min_indices]
            closest_index = min_indices[np.argmin(closest_rvs)]

        return  rvs[closest_index]

    def  _sample_rv_by_weight(elf,rvs, weights):
        """
        Return a random RV based on the weights.

        Parameters:
        rvs (list): List of random variable values
        weights (list): List of un-normalized weights for each RV

        Returns:
        tuple: (sampled_rv, index_of_sampled)
        """
        # Normalize the weights to get probabilities
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]

        # Sample based on probabilities
        sampled_index = random.choices(range(len(rvs)), weights=probabilities)[0]

        return rvs[sampled_index]