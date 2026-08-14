#!/usr/bin/env python

"""
=========================
DetectorContext tutorial
=========================
"""

# %%
# Some probabilistic trackers use detection probability and clutter density in their hypothesis
# weights. DetectorContext allows these quantities to depend on the current hypothesis or
# detection while preserving the existing scalar configuration path.

import datetime

import numpy as np

from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.hypothesiser.gaussianmixture import GaussianMixtureHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.mixturereducer.gaussianmixture import GaussianMixtureReducer
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import ConstantVelocity
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.tracker.pointprocess import PointProcessMultiTargetTracker
from stonesoup.types.detection import Detection, DetectionSet
from stonesoup.types.detector_context import SimpleDetectorContext
from stonesoup.types.state import TaggedWeightedGaussianState
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.updater.pointprocess import PHDUpdater


start_time = datetime.datetime(2026, 1, 1, 12)
measurement_model = LinearGaussian(
    ndim_state=2,
    mapping=[0],
    noise_covar=np.array([[0.2]]))


def reader_with_context():
    # Detection readers still yield ``(time, detections)``. When context is
    # available, ``detections`` can be a DetectionSet.
    for step in range(6):
        time = start_time + datetime.timedelta(seconds=step)
        detection = Detection(
            state_vector=np.array([[float(step)]]),
            timestamp=time,
            measurement_model=measurement_model)

        detector_context = SimpleDetectorContext(
            prob_detection=lambda hypothesis:
                0.9 if hypothesis.prediction.state_vector[0, 0] < 3 else 0.4,
            clutter_spatial_density=lambda detection_:
                1e-3 if detection_.state_vector[0, 0] < 3 else 1e-2)

        yield time, DetectionSet({detection}, detector_context=detector_context)


predictor = KalmanPredictor(transition_model=ConstantVelocity(noise_diff_coeff=0.05))
updater = KalmanUpdater(measurement_model=measurement_model)
hypothesiser = GaussianMixtureHypothesiser(
    hypothesiser=DistanceHypothesiser(
        predictor=predictor,
        updater=updater,
        measure=Mahalanobis(),
        missed_distance=16),
    order_by_detection=True)
phd_updater = PHDUpdater(
    updater=updater,
    prob_detection=0.9,
    clutter_spatial_density=1e-3)
reducer = GaussianMixtureReducer(prune_threshold=1e-5, merge_threshold=4)
birth_component = TaggedWeightedGaussianState(
    state_vector=np.array([[0.0], [1.0]]),
    covar=np.diag([1.0, 1.0]),
    weight=0.25,
    tag=TaggedWeightedGaussianState.BIRTH,
    timestamp=start_time)

# %%
# The tracker extracts the context from each detection set and passes it to the
# underlying components. Existing scalar
# ``prob_detection`` and ``clutter_spatial_density`` values still provide the
# fallback behaviour when no detector context is supplied.

tracker = PointProcessMultiTargetTracker(
    detector=reader_with_context(),
    updater=phd_updater,
    hypothesiser=hypothesiser,
    reducer=reducer,
    birth_component=birth_component)

for time, tracks in tracker:
    print(time, len(tracks), tracker.estimated_number_of_targets)
