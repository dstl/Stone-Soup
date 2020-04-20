# -*- coding: utf-8 -*-
import datetime

import numpy as np

from ..pointprocess import GMPHDTargetTracker
from ...types.state import TaggedWeightedGaussianState
from ...types.mixture import GaussianMixture
from ...mixturereducer.gaussianmixture import GaussianMixtureReducer
from ...updater.pointprocess import PHDUpdater
from ...hypothesiser.gaussianmixture import GaussianMixtureHypothesiser
from ...hypothesiser.distance import DistanceHypothesiser
from ... import measures
from ...models.measurement.linear import LinearGaussian
from ...updater.kalman import KalmanUpdater


def test_gmphd_multi_target_tracker_init():
    GMPHDTargetTracker()


def test_gmphd_multi_target_tracker_init_w_components():
    dim = 5
    num_states = 10
    components = [
        TaggedWeightedGaussianState(
            state_vector=np.random.rand(dim, 1),
            covar=np.eye(dim),
            weight=np.random.rand(),
            tag=i+1
        ) for i in range(num_states)
    ]
    gaussian_mixture = GaussianMixture(components=components)
    tracker = GMPHDTargetTracker(
        gaussian_mixture=gaussian_mixture)
    # check all components are in mixture
    assert set(gaussian_mixture) == set(tracker.gaussian_mixture.components)


def test_gmphd_multi_target_tracker_cycle(detector, predictor):
    previous_time = datetime.datetime(2018, 1, 1, 13, 59)

    timestamp = datetime.datetime.now()
    birth_mean = np.array([[40]])
    birth_covar = np.array([[1000]])
    birth_component = TaggedWeightedGaussianState(
        birth_mean,
        birth_covar,
        weight=0.5,
        tag="birth",
        timestamp=timestamp)

    # Initialise a Kalman Updater
    measurement_model = LinearGaussian(ndim_state=1, mapping=[0],
                                       noise_covar=np.array([[0.04]]))
    updater = KalmanUpdater(measurement_model=measurement_model)
    # Initialise a Gaussian Mixture hypothesiser
    measure = measures.Mahalanobis()
    base_hypothesiser = DistanceHypothesiser(
        predictor, updater, measure=measure, missed_distance=16)
    hypothesiser = GaussianMixtureHypothesiser(predictor, updater,
                                               hypothesiser=base_hypothesiser,
                                               order_by_detection=True)

    # Initialise a Gaussian Mixture reducer
    merge_threshold = 8
    prune_threshold = 1e-6
    reducer = GaussianMixtureReducer(prune_threshold=prune_threshold,
                                     merge_threshold=merge_threshold)

    # Initialise a Point Process updater
    phd_updater = PHDUpdater(updater=updater, prob_detection=0.9)

    tracker = GMPHDTargetTracker(
        detector=detector,
        updater=phd_updater,
        hypothesiser=hypothesiser,
        reducer=reducer,
        birth_component=birth_component
        )

    for time, tracks in tracker:
        assert time == previous_time + datetime.timedelta(minutes=1)
        assert tracker.estimated_number_of_targets > 0
        assert tracker.estimated_number_of_targets < 4
        previous_time = time
        # Shouldn't have more than three tracks
        assert (len(tracks) >= 1) & (len(tracks) <= 3)
