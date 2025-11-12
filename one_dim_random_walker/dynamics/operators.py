import random
from abc import ABC

import numpy as np

from fds.dynamics.fds_operator import Operator
from one_dim_random_walker.core.real_fields_for_walker.one_dim_walker_real_field import RealField
from one_dim_random_walker.core.states.state import IntegerState



class OneDimRandomWalkerExpectationOperator(Operator):
    def __init__(self):
        super().__init__()

    def get_next_state(self, field: RealField) -> IntegerState:
        states_list = []
        weights_list = []
        space = field.state_space
        id = space.ids_view()
        for sid in id:
            s = space.get_state_by_id(int(sid))
            states_list.append(s.state)
            weights_list.append(field.get_field(s).data.value)
        return IntegerState(self._find_closest_to_expectation(states_list, weights_list))

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

class OneDimRandomWalkerSampleOperator(Operator):
    def __init__(self):
        super().__init__()

    def get_next_state(self, field: RealField) -> IntegerState:
        states_list = []
        weights_list = []
        space = field.state_space
        id = space.ids_view()
        for sid in id:
            s = space.get_state_by_id(int(sid))
            states_list.append(s.state)
            weights_list.append(field.get_field(s).data.value)
        return IntegerState(self._sample_rv_by_weight(states_list, weights_list))

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