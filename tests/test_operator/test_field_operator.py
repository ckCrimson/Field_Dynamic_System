import pytest
import jax
import jax.numpy as jnp
from dataclasses import dataclass

from src.field_dynamic_system.core import AbstractState
from src.field_dynamic_system.core.state.state import State
from src.field_dynamic_system.core.state.discrete import AbstractDiscreteStateSpace as DiscreteFiniteStatSpace
from src.field_dynamic_system.core.field.mappings import FieldMapper
from src.field_dynamic_system.core.field.algebra import RealFieldAlgebra
from src.field_dynamic_system.operator.base import InteractionContext
from src.field_dynamic_system.operator.field import FieldBasedOperator, Strategies
from src.field_dynamic_system.operator.statistics import ExpectationOperator




class TestDiceSystem:

    def setup_dice(self, D: int, is_fair: bool):
        self.D = D

        # 1. Create States
        self.states_list = [AbstractState(i) for i in range(1, D + 1)]
        self.space = DiscreteFiniteStatSpace(self.states_list)

        self.algebra = RealFieldAlgebra()

        # 2. Create Values
        if is_fair:
            probs = jnp.ones(D, dtype=jnp.float32) / D
        else:
            key = jax.random.PRNGKey(42)
            probs = jax.random.uniform(key, shape=(D,), dtype=jnp.float32)
            probs = probs / jnp.sum(probs)

        # 3. Initialize Mapper (Empty)
        self.mapper = FieldMapper(self.space, self.algebra)

        # 4. Set Values Loop (Adhering to your Single-State API)
        # We iterate over states and probs and set them one by one.
        for state, prob in zip(self.states_list, probs):
            self.mapper.set_value_at(state, prob)

        self.ctx = InteractionContext(rng_key=jax.random.PRNGKey(0))

    def test_fair_dice_maximum(self):
        """Tc: Fair dice -> Max operator returns ALL states."""
        self.setup_dice(D=6, is_fair=True)
        op = FieldBasedOperator(selection_strategy=Strategies.argmax_all)

        observed = op.observe(self.mapper, self.ctx)

        # Should return all 6 faces
        assert len(observed) == 6
        assert observed[0].name == 1

    def test_fair_dice_expectation(self):
        """Tc: Fair dice -> Expectation returns 3.5."""
        self.setup_dice(D=6, is_fair=True)
        op = ExpectationOperator()

        value = op.observe(self.mapper, self.ctx)
        assert jnp.isclose(value, 3.5)

    def test_unfair_dice_sampling(self):
        """Tc: Unfair dice -> Returns valid single states."""
        self.setup_dice(D=6, is_fair=False)
        op = FieldBasedOperator(selection_strategy=Strategies.sample)

        for i in range(5):
            ctx = InteractionContext(rng_key=jax.random.PRNGKey(i))
            obs = op.observe(self.mapper, ctx)
            assert isinstance(obs, AbstractState)