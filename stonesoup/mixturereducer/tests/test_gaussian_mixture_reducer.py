# -*- coding: utf-8 -*-
import numpy as np

from stonesoup.mixturereducer.gaussianmixture import GaussianMixtureReducer
from stonesoup.types.mixture import GaussianMixture
from stonesoup.types.state import (TaggedWeightedGaussianState,
                                   WeightedGaussianState)


def test_gaussianmixture_reducer_w_tags():
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
    mixturereducer = GaussianMixtureReducer(prune_threshold=prune_threshold,
                                            merge_threshold=merge_threshold)
    reduced_mixture_state = mixturereducer.reduce(mixturestate)
    assert len(reduced_mixture_state) == 1


def test_gaussianmixture_reducer():
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
    mixturereducer = GaussianMixtureReducer(prune_threshold=prune_threshold,
                                            merge_threshold=merge_threshold)
    reduced_mixture_state = mixturereducer.reduce(mixturestate)
    assert len(reduced_mixture_state) == 1
