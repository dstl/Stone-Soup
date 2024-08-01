import numpy as np
import pytest

from stonesoup.mixturereducer.gaussianmixture import GaussianMixtureReducer
from stonesoup.types.mixture import GaussianMixture
from stonesoup.types.state import (TaggedWeightedGaussianState,
                                   WeightedGaussianState)


@pytest.fixture(params=[None, 10])
def kdtree_max_distance(request):
    return request.param


def test_gaussianmixture_reducer_w_tags(kdtree_max_distance):
    dim = 4
    num_states = 10
    low_weight_states = [
        TaggedWeightedGaussianState(
            state_vector=np.random.rand(dim, 1)*10,
            covar=np.eye(dim),
            weight=1e-10,
            tag=i+1
        ) for i in range(num_states)
    ]
    states_to_be_merged = [
        TaggedWeightedGaussianState(
            state_vector=np.array([[1], [2], [1], [2]]),
            covar=np.eye(dim)*5,
            weight=0.05,
            tag=i+1
        ) for i in range(num_states, num_states*2)
    ]

    mixturestate = GaussianMixture(
        components=low_weight_states+states_to_be_merged
    )
    merge_threshold = 16
    prune_threshold = 1e-6
    mixturereducer = GaussianMixtureReducer(
        prune_threshold=prune_threshold,
        merge_threshold=merge_threshold,
        kdtree_max_distance=kdtree_max_distance)
    reduced_mixture_state = mixturereducer.reduce(mixturestate)
    assert len(reduced_mixture_state) == 1


def test_gaussianmixture_reducer(kdtree_max_distance):
    dim = 4
    num_states = 10
    low_weight_states = [
        WeightedGaussianState(
            state_vector=np.random.rand(dim, 1)*10,
            covar=np.eye(dim),
            weight=1e-10,
        ) for i in range(num_states)
    ]
    states_to_be_merged = [
        WeightedGaussianState(
            state_vector=np.array([[1], [2], [1], [2]]),
            covar=np.eye(dim)*5,
            weight=0.05,
        ) for i in range(num_states, num_states*2)
    ]

    mixturestate = GaussianMixture(
        components=low_weight_states+states_to_be_merged
    )
    merge_threshold = 16
    prune_threshold = 1e-6
    mixturereducer = GaussianMixtureReducer(
        prune_threshold=prune_threshold,
        merge_threshold=merge_threshold,
        kdtree_max_distance=kdtree_max_distance)
    reduced_mixture_state = mixturereducer.reduce(mixturestate)
    assert len(reduced_mixture_state) == 1


def test_gaussianmixture_truncating():
    """
    Test that the trucating function of the GaussianMixtureReducer works
    properly. It should remove low weight components and keep only a certain
    number of them (in this case, 5).
    """
    dim = 4
    num_states = 10
    states = [
        WeightedGaussianState(
            state_vector=np.random.rand(dim, 1)*10,
            covar=np.eye(dim),
            weight=i/10,
        ) for i in range(1, num_states+1)
    ]
    mixture = GaussianMixture(components=states)

    prune_threshold = 0.15  # one component will be pruned
    mixturereducer = GaussianMixtureReducer(prune_threshold=prune_threshold,
                                            merging=False,
                                            max_number_components=5)
    reduced_mixture = mixturereducer.reduce(mixture)
    assert len(reduced_mixture) == 5
