# -*- coding: utf-8 -*-
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
