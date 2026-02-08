import pytest
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse
from typing import Any

# --- Import your actual classes ---
from src.field_dynamic_system.systems.static import (
    StaticFieldTopologySystem,
    StaticSystem
)
from src.field_dynamic_system.core.field.mappings import (
    DiscreteFieldMapper,
    ContinuousFieldMapper,
    FieldMapper
)
from src.field_dynamic_system.core.field.algebra import RealFieldAlgebra, VectorFieldAlgebra
# CRITICAL: Import the Lazy implementation
from src.field_dynamic_system.core.state.discrete import LazyDiscreteStateSpace
from src.field_dynamic_system.neighbor.discrete import DiscreteTopology


# --- MOCKS & FIXTURES ---

class MockTopology(DiscreteTopology):
    """
    A dummy topology.
    We explicitly override get_adjacency_matrix to ensure the
    StaticFieldTopologySystem detects it as an 'explicit_matrix' source.
    """

    def __init__(self, state_space):
        # Bypass parent init to avoid complex setup
        self._state_space = state_space

    @property
    def state_space(self):
        return self._state_space

    def compute_neighbors(self, state):
        return [state]  # Self-loop

    def get_adjacency_matrix(self):
        # Return a simple Identity Matrix as a JAX BCOO or Array
        # This triggers the 'explicit_matrix' path in the System
        return jnp.eye(self.state_space.num_states)


@pytest.fixture
def lazy_space():
    """
    A LazyDiscreteStateSpace with 2 states.
    Raw Data: [[0], [1]] (Indices as coordinates)
    """
    raw_data = np.array([[0], [1]])
    # Simple wrapper just returns the integer
    return LazyDiscreteStateSpace(raw_data=raw_data, wrapper_class=int)


@pytest.fixture
def scalar_field(lazy_space):
    """A scalar field [10.0, 20.0]."""
    mapper = DiscreteFieldMapper(lazy_space, RealFieldAlgebra())
    # Manually set raw buffer for speed/testing
    # Note: LazySpace ensures 2D internal data, so we match that
    mapper.explicit_buffer = jnp.array([[10.0], [20.0]])
    return mapper


@pytest.fixture
def vector_field(lazy_space):
    """A 2D vector field [[1, 1], [-1, -1]]."""
    mapper = DiscreteFieldMapper(lazy_space, VectorFieldAlgebra(dim=2))
    mapper.explicit_buffer = jnp.array([[1.0, 1.0], [-1.0, -1.0]])
    return mapper


@pytest.fixture
def topology(lazy_space):
    return MockTopology(lazy_space)


# --- TEST CASES ---

class TestStaticFieldTopologySystem:

    def test_initialization_extracts_raw_data(self, lazy_space, topology, scalar_field):
        """
        Test that initializing the system STRIPS the FieldMapper object
        and stores only the JAX array.
        """
        system = StaticFieldTopologySystem(
            state=jnp.array([0]),
            topology=topology,
            space=lazy_space,
            field_mappers={"height": scalar_field}
        )

        # 1. Check Field Names
        assert "height" in system.field_names

        # 2. Check Data Type (Crucial!)
        raw_data = system.get_field("height")
        # Should be a JAX array, NOT a FieldMapper object
        assert isinstance(raw_data, (jnp.ndarray, np.ndarray))
        assert not isinstance(raw_data, DiscreteFieldMapper)

        # 3. Check Values
        assert raw_data[0] == 10.0
        assert raw_data[1] == 20.0

    def test_topology_extraction(self, lazy_space, topology):
        """
        Test that DiscreteTopology extracts the raw matrix.
        """
        system = StaticFieldTopologySystem(
            state=0,
            topology=topology,
            space=lazy_space
        )

        # Should be 'explicit_matrix' because MockTopology returns a JAX array
        assert system._topology_type == "explicit_matrix"

        # Verify the data stored is indeed the matrix
        assert isinstance(system.topology_data, (jnp.ndarray, np.ndarray, sparse.BCOO))
        assert system.topology_data.shape == (2, 2)

    def test_immutability_with_field(self, lazy_space, topology, scalar_field, vector_field):
        """
        Test that adding a field creates a NEW system and leaves OLD one alone.
        """
        # System 1: Height Only
        sys1 = StaticFieldTopologySystem(
            state=0,
            topology=topology,
            space=lazy_space,
            field_mappers={"height": scalar_field}
        )

        # System 2: Height + Wind (Created via with_field)
        sys2 = sys1.with_field("wind", vector_field)

        # CHECKS

        # A. Identity: They are different objects
        assert sys1 is not sys2

        # B. Content: Sys1 is unchanged
        assert "height" in sys1.field_names
        assert "wind" not in sys1.field_names

        # C. Content: Sys2 has both
        assert "height" in sys2.field_names
        assert "wind" in sys2.field_names

        # D. Data Integrity: Sys2's wind data is correct
        raw_wind = sys2.get_field("wind")
        assert raw_wind.shape == (2, 2)
        assert raw_wind[0, 0] == 1.0

    def test_continuous_field_mapper(self, lazy_space, topology):
        """
        Test that Continuous Fields store the FUNCTION, not an array.
        """
        # Create a continuous field (Function: f(x) = x * 2)
        bg_func = lambda x: x * 2.0
        # Continuous Mapper can wrap a Discrete/Lazy Space conceptually for background fields
        cont_mapper = ContinuousFieldMapper(lazy_space, RealFieldAlgebra(), bg_func=bg_func)

        system = StaticFieldTopologySystem(
            state=0,
            topology=topology,
            field_mappers={"gravity": cont_mapper}
        )

        # Get the field
        stored_func = system.get_field("gravity")

        # Assert it is callable
        assert callable(stored_func)

        # Verify logic works
        # Note: We wrap input in JAX array to match ContinuousFieldMapper expectations
        res = stored_func(jnp.array([10.0]))
        assert res == 20.0

    def test_raw_data_injection(self, lazy_space, topology):
        """
        Test we can bypass FieldMappers entirely and inject raw dictionaries.
        (Useful for serialization/deserialization).
        """
        raw_dict = {
            "temperature": jnp.array([100.0, 200.0])
        }

        system = StaticFieldTopologySystem(
            state=0,
            topology=topology,
            raw_fields=raw_dict
        )

        assert "temperature" in system.field_names
        assert system.get_field("temperature")[1] == 200.0

    def test_error_handling(self, lazy_space, topology):
        """Test getting a non-existent field raises KeyError."""
        system = StaticFieldTopologySystem(state=0, topology=topology)

        with pytest.raises(KeyError):
            system.get_field("non_existent_field")