import copy
import datetime

import numpy as np
import pytest


from ..mixture import GaussianMixture
from ..state import (GaussianState, WeightedGaussianState,
                     TaggedWeightedGaussianState)


def test_GaussianMixture_empty_components():
    mixturestate = GaussianMixture(components=[])
    assert len(mixturestate) == 0


def test_GaussianMixture():
    dim = 5
    num_states = 10
    states = [
        WeightedGaussianState(
            state_vector=np.random.rand(dim, 1),
            covar=np.eye(dim),
            weight=np.random.rand()
        ) for _ in range(num_states)
    ]
    mixturestate = GaussianMixture(components=states)
    assert len(mixturestate) == num_states
    for component1, component2 in zip(mixturestate, states):
        assert component1 == component2

    # Test iterator functionality implemented with __iter__ and __next__
    index = 0
    for component in mixturestate:
        assert component == states[index]
        index += 1

    # Test number_of_components
    assert len(mixturestate) == len(states)

    # Test ndim
    assert mixturestate.ndim == dim


def test_GaussianMixture_append():
    dim = 5
    state = WeightedGaussianState(
            state_vector=np.random.rand(dim, 1),
            covar=np.eye(dim),
            weight=np.random.rand()
        )

    mixturestate = GaussianMixture(components=[])
    mixturestate.append(state)
    assert len(mixturestate) == 1
    for component in mixturestate:
        assert component == state

    # Test iterator functionality implemented with __iter__ and __next__
    index = 0
    for component in mixturestate:
        assert component == state
        index += 1

    # Test number_of_components
    assert len(mixturestate) == 1


def test_GaussianMixture_with_tags():
    dim = 5
    num_states = 10
    states = [
        TaggedWeightedGaussianState(
            state_vector=np.random.rand(dim, 1),
            covar=np.eye(dim),
            weight=np.random.rand(),
            tag=i+1
        ) for i in range(num_states)
    ]
    mixturestate = GaussianMixture(components=states)
    assert len(mixturestate) == num_states
    # Check equality of components of GaussianMixture and states
    for component1, component2 in zip(mixturestate, states):
        assert component1 == component2
    # Check each component has its proper tag assigned
    for i in range(num_states):
        assert mixturestate[i].tag == i+1


def test_GaussianMixture_with_single_component():
    dim = 5
    state = TaggedWeightedGaussianState(
            state_vector=np.random.rand(dim, 1),
            covar=np.eye(dim),
            weight=np.random.rand(),
            tag=2
        )
    mixturestate = GaussianMixture(components=[state])
    assert len(mixturestate) == 1
    # Check equality of components of GaussianMixture and states
    for component in mixturestate:
        assert component == state


def test_GaussianMixture_wrong_type():
    dim = 5
    state = [GaussianState(
            state_vector=np.random.rand(dim, 1),
            covar=np.eye(dim),
        )]
    with pytest.raises(ValueError):
        GaussianMixture(components=state)


def test_GaussianMixture_timestamp():
    dim = 5
    timestamp = datetime.datetime.now()
    state = WeightedGaussianState(
        state_vector=np.random.rand(dim, 1),
        covar=np.eye(dim),
        timestamp=timestamp
    )
    states = [copy.copy(state) for _ in range(5)]

    mixturestate = GaussianMixture(components=states)
    assert mixturestate.timestamp == timestamp

    states[-1].timestamp += datetime.timedelta(hours=1)
    with pytest.raises(ValueError):
        GaussianMixture(components=states)


def test_GaussianMixture_get_and_set_item():
    dim = 5
    num_states = 10
    weights = np.linspace(0, 1.0, num=num_states)
    states = [
        WeightedGaussianState(
            state_vector=np.random.rand(dim, 1),
            covar=np.eye(dim),
            weight=weights[i]
        ) for i in range(num_states)
    ]
    mixturestate = GaussianMixture(components=states)
    new_component = WeightedGaussianState(
        state_vector=np.random.rand(dim, 1),
        covar=np.eye(dim),
        weight=1
        )
    mixturestate[0] = new_component
    assert mixturestate[0] == new_component
    with pytest.raises(TypeError):
        assert mixturestate["Test"]

    with pytest.raises(ValueError):
        mixturestate[0] = GaussianState([1], [[1]])

    with pytest.raises(ValueError):
        mixturestate.append(GaussianState([1], [[1]]))


def test_GaussianMixture_contains_item():
    dim = 5
    num_states = 10
    weights = np.linspace(0, 1.0, num=num_states)
    states = [
        WeightedGaussianState(
            state_vector=np.random.rand(dim, 1),
            covar=np.eye(dim),
            weight=weights[i]
        ) for i in range(num_states)
    ]
    mixturestate = GaussianMixture(components=states)
    check_component_in = mixturestate[0]
    check_component_not_in = WeightedGaussianState(
        state_vector=np.random.rand(dim, 1),
        covar=np.eye(dim),
        weight=np.random.rand(1, 1)
    )
    assert check_component_in in mixturestate
    assert check_component_not_in not in mixturestate
    with pytest.raises(ValueError):
        assert 1 in mixturestate


def test_GaussianMixture_copy():
    dim = 5
    num_states = 10
    weights = np.linspace(0, 1.0, num=num_states)
    states = [
        WeightedGaussianState(
            state_vector=np.random.rand(dim, 1),
            covar=np.eye(dim),
            weight=weights[i]
        ) for i in range(num_states)
    ]
    mixturestate = GaussianMixture(components=states)
    assert mixturestate.components is states

    new_mixturestate = copy.copy(mixturestate)

    assert new_mixturestate.components == mixturestate.components
    assert new_mixturestate is not mixturestate.components

    new_mixturestate.pop()
    assert len(new_mixturestate) == len(mixturestate) - 1


def test_GaussianMixture_Gaussian_state():
    dim = 5
    num_states = 10
    timestamp = datetime.datetime.now()
    states = [
        WeightedGaussianState(
            state_vector=[float(i)]*dim,
            covar=np.eye(dim),
            weight=1.,
            timestamp=timestamp
        ) for i in range(1, num_states)
    ]

    mixturestate = GaussianMixture(components=states)

    assert np.allclose(mixturestate.mean, [5]*dim)
    assert np.allclose(mixturestate.covar, np.full(dim, 20/3) + np.eye(dim))
    assert mixturestate.timestamp == timestamp
