import datetime

import numpy as np

from ..pointprocess import PointProcessMultiTargetTracker
from ...types.state import TaggedWeightedGaussianState
from ...mixturereducer.gaussianmixture import GaussianMixtureReducer
from ...updater.pointprocess import PHDUpdater
from ...hypothesiser.gaussianmixture import GaussianMixtureHypothesiser
from ...hypothesiser.distance import DistanceHypothesiser
from ... import measures
from ...models.measurement.linear import LinearGaussian
from ...types.detection import DetectionSet
from ...types.detector_context import SimpleDetectorContext
from ...updater.kalman import KalmanUpdater


def test_point_process_multi_target_tracker_cycle(detector, predictor):
    previous_time = datetime.datetime(2018, 1, 1, 13, 59)
    timestamp = datetime.datetime.now()
    birth_mean = np.array([[40]])
    birth_covar = np.array([[1000]])
    birth_component = TaggedWeightedGaussianState(
        birth_mean,
        birth_covar,
        weight=0.3,
        tag=TaggedWeightedGaussianState.BIRTH,
        timestamp=timestamp)

    # Initialise a Kalman Updater
    measurement_model = LinearGaussian(ndim_state=1, mapping=[0],
                                       noise_covar=np.array([[0.04]]))
    updater = KalmanUpdater(measurement_model=measurement_model)
    # Initialise a Gaussian Mixture hypothesiser
    measure = measures.Mahalanobis()
    base_hypothesiser = DistanceHypothesiser(
        predictor, updater, measure=measure, missed_distance=16)
    hypothesiser = GaussianMixtureHypothesiser(hypothesiser=base_hypothesiser,
                                               order_by_detection=True)

    # Initialise a Gaussian Mixture reducer
    merge_threshold = 4
    prune_threshold = 1e-5
    reducer = GaussianMixtureReducer(prune_threshold=prune_threshold,
                                     merge_threshold=merge_threshold)

    # Initialise a Point Process updater
    phd_updater = PHDUpdater(updater=updater, prob_detection=0.8)

    tracker = PointProcessMultiTargetTracker(
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
        # Shouldn't have more than three active tracks
        assert (len(tracks) >= 1) & (len(tracks) <= 3)
        # All tracks should have unique IDs
        assert len(tracker.gaussian_mixture.component_tags) == len(tracker.gaussian_mixture)


def test_point_process_multi_target_tracker_detector_context(detector, predictor):
    detector_context = SimpleDetectorContext(prob_detection=0.8)
    detector = (
        (time, DetectionSet(detections, detector_context=detector_context))
        for time, detections in detector)
    timestamp = datetime.datetime.now()
    birth_component = TaggedWeightedGaussianState(
        np.array([[40]]),
        np.array([[1000]]),
        weight=0.3,
        tag=TaggedWeightedGaussianState.BIRTH,
        timestamp=timestamp)

    measurement_model = LinearGaussian(ndim_state=1, mapping=[0],
                                       noise_covar=np.array([[0.04]]))
    updater = KalmanUpdater(measurement_model=measurement_model)
    base_hypothesiser = DistanceHypothesiser(
        predictor, updater, measure=measures.Mahalanobis(), missed_distance=16)
    hypothesiser = GaussianMixtureHypothesiser(hypothesiser=base_hypothesiser,
                                               order_by_detection=True)

    class ContextHypothesiser:
        def __init__(self, hypothesiser):
            self.hypothesiser = hypothesiser
            self.detector_contexts = []

        def hypothesise(self, components, detections, timestamp, detector_context=None):
            self.detector_contexts.append(detector_context)
            return self.hypothesiser.hypothesise(components, detections, timestamp)

    class ContextUpdater(PHDUpdater):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.detector_contexts = []

        def update(self, hypotheses, detector_context=None, **kwargs):
            self.detector_contexts.append(detector_context)
            return super().update(hypotheses, detector_context=detector_context, **kwargs)

    hypothesiser = ContextHypothesiser(hypothesiser)
    reducer = GaussianMixtureReducer(prune_threshold=1e-5, merge_threshold=4)
    phd_updater = ContextUpdater(updater=updater, prob_detection=0.1)

    tracker = PointProcessMultiTargetTracker(
        detector=detector,
        updater=phd_updater,
        hypothesiser=hypothesiser,
        reducer=reducer,
        birth_component=birth_component
        )

    time, tracks = next(iter(tracker))
    assert time == datetime.datetime(2018, 1, 1, 14)
    assert tracks
    assert hypothesiser.detector_contexts == [detector_context]
    assert phd_updater.detector_contexts == [detector_context]
