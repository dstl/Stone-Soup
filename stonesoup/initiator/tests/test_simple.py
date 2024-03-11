import datetime

import numpy as np
import pytest
from pytest import approx

from ...models.base import LinearModel, ReversibleModel
from ...models.measurement.linear import LinearGaussian
from ...models.measurement.nonlinear import CartesianToBearingRange, Cartesian2DToBearing, \
    CombinedReversibleGaussianMeasurementModel
from ...models.transition.linear import \
    CombinedLinearGaussianTransitionModel, ConstantVelocity
from ...updater.kalman import KalmanUpdater, ExtendedKalmanUpdater
from ...predictor.kalman import KalmanPredictor
from ...deleter.time import UpdateTimeDeleter
from ...hypothesiser.distance import DistanceHypothesiser
from ...dataassociator.neighbour import NearestNeighbour
from ...measures import Mahalanobis
from ...types.detection import Detection, TrueDetection
from ...types.hypothesis import SingleHypothesis
from ...types.prediction import Prediction
from ...types.state import GaussianState
from ...types.update import ParticleStateUpdate, Update, GaussianMixtureUpdate, \
    ASDGaussianStateUpdate, EnsembleStateUpdate
from ..simple import (
    SinglePointInitiator, SimpleMeasurementInitiator,
    MultiMeasurementInitiator, GaussianParticleInitiator, GaussianMixtureInitiator,
    ASDGaussianInitiator, EnsembleInitiator,
    NoHistoryMultiMeasurementInitiator
)


@pytest.mark.parametrize(
    'measurement_model',
    [LinearGaussian(2, [0, 1], np.diag([1, 1])),
     CartesianToBearingRange(2, [1, 0], np.diag([0.1, 1]))],
    ids=['linear', 'non-linear'])
def test_spi(measurement_model):
    """Test SinglePointInitiator"""

    # Prior state information
    prior_state = GaussianState(
        np.array([[0], [0]]),
        np.array([[100, 0], [0, 1]]))

    # Create the Kalman updater
    if isinstance(measurement_model, LinearModel):
        kup = KalmanUpdater(measurement_model)
    else:
        kup = ExtendedKalmanUpdater(measurement_model)

    # Define the Initiator
    initiator = SinglePointInitiator(
        prior_state,
        measurement_model)

    # Define 2 detections from which tracks are to be initiated
    timestamp = datetime.datetime.now()
    detections = [Detection(np.array([[4.5], [2.0]]), timestamp),
                  Detection(np.array([[-4.5], [2.0]]), timestamp)]

    # Run the initiator based on the available detections
    tracks = initiator.initiate(detections, timestamp)

    # Ensure same number of tracks are initiated as number of measurements
    # (i.e. 2)
    assert (len(tracks) == 2)

    # Ensure that tracks are initiated correctly
    evaluated_tracks = [False, False]
    for detection in detections:

        hypo = SingleHypothesis(prediction=prior_state, measurement=detection)
        eval_track_state = kup.update(hypo)

        # Compare against both tracks
        for track_idx, track in enumerate(tracks):

            if (np.array_equal(eval_track_state.mean, track.mean)
                    and np.array_equal(eval_track_state.covar, track.covar)):
                evaluated_tracks[track_idx] = True

    # Ensure both tracks have been evaluated
    assert (all(evaluated_tracks))

    assert set(detections) == set(track.state.hypothesis.measurement
                                  for track in tracks)


def test_linear_measurement():
    measurement_model = LinearGaussian(2, [0], np.array([[50]]))
    measurement_initiator = SimpleMeasurementInitiator(
        GaussianState(np.array([[0], [0]]), np.diag([100, 10])),
        measurement_model
    )

    timestamp = datetime.datetime.now()
    detections = [Detection(np.array([[5]]), timestamp),
                  Detection(np.array([[-5]]), timestamp)]

    tracks = measurement_initiator.initiate(detections, timestamp)

    for track in tracks:
        if track.state_vector[0, 0] > 0:
            assert np.array_equal(track.state_vector, np.array([[5], [0]]))
            assert np.array_equal(
                measurement_model.matrix() @ track.state_vector,
                detections[0].state_vector)
            assert track.state.hypothesis.measurement is detections[0]
        else:
            assert np.array_equal(track.state_vector, np.array([[-5], [0]]))
            assert np.array_equal(
                measurement_model.matrix() @ track.state_vector,
                detections[1].state_vector)
            assert track.state.hypothesis.measurement is detections[1]

        assert track.timestamp == timestamp

        assert np.diag([50, 10]) == approx(track.covar)
        assert measurement_model.matrix() @ track.covar @ \
            measurement_model.matrix().T == approx(measurement_model.covar())


@pytest.mark.parametrize("meas_model", (CartesianToBearingRange, Cartesian2DToBearing))
@pytest.mark.parametrize("skip_non_linear", (None, True, False))
def test_nonlinear_measurement(meas_model, skip_non_linear):

    meas_params = [2, [0, 1], np.diag([np.radians(2), 30])]
    measurement_model = meas_model(*meas_params)
    combined_measurement_model = CombinedReversibleGaussianMeasurementModel([measurement_model])

    measurement_initiator = SimpleMeasurementInitiator(
        prior_state=GaussianState(np.array([[0], [0]]), np.diag([100, 10])),
        measurement_model=measurement_model,
        skip_non_reversible=skip_non_linear)
    combined_measurement_initiator = SimpleMeasurementInitiator(
        prior_state=GaussianState(np.array([[0], [0]]), np.diag([100, 10])),
        measurement_model=combined_measurement_model,
        skip_non_reversible=skip_non_linear)

    timestamp = datetime.datetime.now()
    detections = [Detection(np.array([[5, 2]]), timestamp),
                  Detection(np.array([[-5, -2]]), timestamp)]

    if not (isinstance(measurement_model, ReversibleModel) or skip_non_linear):
        with pytest.raises(Exception):
            # Non-reversible and not skipping
            measurement_initiator.initiate(detections, timestamp)
        with pytest.raises(NotImplementedError):
            # Reversible but not implemented
            combined_measurement_initiator.initiate(detections, timestamp)
    elif not isinstance(measurement_model, ReversibleModel) and skip_non_linear:
        # Skipping for non-reversible
        assert len(measurement_initiator.initiate(detections, timestamp)) == 0
        assert len(combined_measurement_initiator.initiate(detections, timestamp)) == 0
    else:
        # Otherwise tracks made
        all_tracks = [measurement_initiator.initiate(detections, timestamp),
                      combined_measurement_initiator.initiate(detections, timestamp)]
        for tracks in all_tracks:
            assert len(tracks) == 2
            for track in tracks:
                assert track.timestamp == timestamp
                jac = measurement_model.jacobian(track.state)
                Ry = jac @ track.covar @ jac.T
                assert Ry == approx(measurement_model.covar())


def test_linear_measurement_non_direct():
    class _LinearMeasurementModel(LinearModel):
        ndim_state = 2
        ndmim_meas = 2
        mapping = (0, 1)

        def matrix(self):
            return np.array([[0, 1], [2, 0]])

        @staticmethod
        def covar():
            return np.diag([10, 50])

        def ndim(self):
            pass

        def pdf(self):
            pass

        def rvs(slef):
            pass

    measurement_model = _LinearMeasurementModel()
    measurement_initiator = SimpleMeasurementInitiator(
        GaussianState(np.array([[0], [0]]), np.diag([100, 10])),
        measurement_model
    )

    timestamp = datetime.datetime.now()
    detections = [Detection(np.array([[5], [2]]), timestamp),
                  Detection(np.array([[-5], [8]]), timestamp)]

    tracks = measurement_initiator.initiate(detections, timestamp)

    for track in tracks:
        if track.state_vector[1, 0] > 0:
            assert np.array_equal(track.state_vector, np.array([[1], [5]]))
            assert np.array_equal(
                measurement_model.matrix() @ track.state_vector,
                detections[0].state_vector)
            assert track.state.hypothesis.measurement is detections[0]
        else:
            assert np.array_equal(track.state_vector, np.array([[4], [-5]]))
            assert np.array_equal(
                measurement_model.matrix() @ track.state_vector,
                detections[1].state_vector)
            assert track.state.hypothesis.measurement is detections[1]

        assert track.timestamp == timestamp

        assert np.diag([12.5, 10]) == approx(track.covar)
        assert measurement_model.matrix() @ track.covar @ \
            measurement_model.matrix().T == approx(measurement_model.covar())


def test_linear_measurement_extra_state_dim():
    class _LinearMeasurementModel(LinearModel):
        ndim_state = 3
        ndmim_meas = 2

        mapping = (0, 2)

        def matrix(self):
            return np.array([[1, 0, 0], [0, 0, 1]])

        @staticmethod
        def covar():
            return np.diag([10, 50])

        def ndim(self):
            pass

        def pdf(self):
            pass

        def rvs(self):
            pass

    measurement_model = _LinearMeasurementModel()
    measurement_initiator = SimpleMeasurementInitiator(
        GaussianState(np.array([[0], [0], [0]]), np.diag([100, 10, 500])),
        measurement_model
    )

    timestamp = datetime.datetime.now()
    detections = [Detection(np.array([[5], [2]]), timestamp),
                  Detection(np.array([[-5], [8]]), timestamp)]

    tracks = measurement_initiator.initiate(detections, timestamp)

    for track in tracks:
        if track.state_vector[0, 0] > 0:
            assert np.array_equal(
                track.state_vector,
                np.array([[5], [0], [2]]))
            assert np.array_equal(
                measurement_model.matrix() @ track.state_vector,
                detections[0].state_vector)
            assert track.state.hypothesis.measurement is detections[0]
        else:
            assert np.array_equal(
                track.state_vector,
                np.array([[-5], [0], [8]]))
            assert np.array_equal(
                measurement_model.matrix() @ track.state_vector,
                detections[1].state_vector)
            assert track.state.hypothesis.measurement is detections[1]

        assert track.timestamp == timestamp

        assert np.diag([10, 10, 50]) == approx(track.covar)
        assert measurement_model.matrix() @ track.covar @ \
            measurement_model.matrix().T == approx(measurement_model.covar())


def create_multi_measurement_initiator(obj_class, **kwargs):
    transition_model = CombinedLinearGaussianTransitionModel(
        (ConstantVelocity(0.05), ConstantVelocity(0.05)))
    measurement_model = LinearGaussian(
        ndim_state=4, mapping=[0, 2], noise_covar=np.diag([10, 10]))

    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)
    hypothesiser = DistanceHypothesiser(
        predictor, updater, measure=Mahalanobis())
    data_associator = NearestNeighbour(hypothesiser)
    deleter = UpdateTimeDeleter(datetime.timedelta(seconds=59))

    obj_kwargs = dict(prior_state=GaussianState([[0], [0], [0], [0]], np.diag([0, 15, 0, 15])),
                      deleter=deleter,
                      data_associator=data_associator,
                      updater=updater,
                      measurement_model=measurement_model)

    obj_kwargs.update(kwargs)
    measurement_initiator = obj_class(**obj_kwargs)

    return measurement_initiator


@pytest.mark.parametrize('updates_only', [False, True])
def test_multi_measurement(updates_only):
    measurement_initiator = create_multi_measurement_initiator(MultiMeasurementInitiator,
                                                               updates_only=updates_only)

    timestamp = datetime.datetime.now()
    first_detections = {Detection(np.array([[5], [2]]), timestamp),
                        Detection(np.array([[-5], [-2]]), timestamp)}

    first_tracks = measurement_initiator.initiate(first_detections, timestamp)
    assert len(first_tracks) == 0
    assert len(measurement_initiator.holding_tracks) == 2

    timestamp = datetime.datetime.now() + datetime.timedelta(seconds=60)
    second_detections = {Detection(np.array([[5], [3]]), timestamp)}

    second_tracks = measurement_initiator.initiate(second_detections, timestamp)

    if updates_only:
        assert len(second_tracks) == 1
    else:
        assert len(second_tracks) == 2
        assert any(isinstance(track.state, Prediction) for track in second_tracks)
    assert any(isinstance(track.state, Update) for track in second_tracks)
    assert len(measurement_initiator.holding_tracks) == 0


def test_no_history_multi_measurement():
    measurement_initiator = create_multi_measurement_initiator(NoHistoryMultiMeasurementInitiator,
                                                               updates_only=False)

    timestamp = datetime.datetime.now()
    first_detections = {Detection(np.array([[5], [2]]), timestamp),
                        Detection(np.array([[-5], [-2]]), timestamp)}

    first_tracks = measurement_initiator.initiate(first_detections, timestamp)
    assert len(first_tracks) == 0
    assert len(measurement_initiator.holding_tracks) == 2

    timestamp = datetime.datetime.now() + datetime.timedelta(seconds=60)
    second_detections = {Detection(np.array([[5], [3]]), timestamp)}

    second_tracks = measurement_initiator.initiate(second_detections, timestamp)
    for track in second_tracks:
        assert len(track) == 1


@pytest.mark.parametrize("measurement_model_class",
                         (CartesianToBearingRange, Cartesian2DToBearing, LinearGaussian))
@pytest.mark.parametrize("skip_non_reversible", (True, False))
def test_skip_in_multi_measurement(measurement_model_class, skip_non_reversible):
    timestamp = datetime.datetime.now()

    if measurement_model_class == Cartesian2DToBearing:
        state_len = 1
    else:
        state_len = 2

    measurement_model = measurement_model_class(ndim_state=2, mapping=(0, 1),
                                                noise_covar=np.diag([1]*state_len))

    det = Detection(state_vector=np.array([[2]*state_len]),
                    timestamp=timestamp,
                    measurement_model=measurement_model
                    )

    interal_initiator = SinglePointInitiator(
        prior_state=GaussianState([[0], [0]], np.diag([15, 15])))

    measurement_initiator = create_multi_measurement_initiator(
        MultiMeasurementInitiator,
        initiator=interal_initiator,
        measurement_model=None,
        skip_non_reversible=skip_non_reversible)

    measurement_initiator.initiate({det}, timestamp)
    holding_tracks = measurement_initiator.holding_tracks

    if isinstance(measurement_model, Cartesian2DToBearing) and skip_non_reversible:
        assert len(holding_tracks) == 0
    else:
        assert len(holding_tracks) == 1
        for track in holding_tracks:
            assert track.timestamp == timestamp


@pytest.mark.parametrize("initiator", [
    SinglePointInitiator(
        GaussianState(np.array([[0]]), np.array([[100]]))
    ),
    SimpleMeasurementInitiator(
        GaussianState(np.array([[0]]), np.array([[100]]))
    ),
], ids=['SinglePoint', 'LinearMeasurement'])
def test_measurement_model(initiator):
    timestamp = datetime.datetime.now()
    dummy_detection = TrueDetection(np.array([0, 0]), timestamp)
    # The SinglePointInitiator will raise an error when the ExtendedKalmanUpdater
    # is called and neither the detection nor the initiator has a measurement
    # model. The SimpleMeasurementInitiator will raise an error in the if/else
    # blocks.
    with pytest.raises(ValueError):
        _ = initiator.initiate({dummy_detection}, timestamp)


@pytest.mark.parametrize("gaussian_initiator", [
    SinglePointInitiator(
        GaussianState(np.array([[0]]), np.array([[100]])),
        LinearGaussian(1, [0], np.array([[1]]))
    ),
    SimpleMeasurementInitiator(
        GaussianState(np.array([[0]]), np.array([[100]])),
        LinearGaussian(1, [0], np.array([[1]]))
    ),
], ids=['SinglePoint', 'LinearMeasurement'])
def test_gaussian_particle(gaussian_initiator):
    particle_initiator = GaussianParticleInitiator(gaussian_initiator)

    timestamp = datetime.datetime.now()
    detections = [Detection(np.array([[5]]), timestamp),
                  Detection(np.array([[-5]]), timestamp)]

    tracks = particle_initiator.initiate(detections, timestamp)

    for track in tracks:
        assert isinstance(track.state, ParticleStateUpdate)

        if track.state.mean > 0:
            assert np.allclose(track.state.mean, np.array([[5]]), atol=0.4)
            assert track.state.hypothesis.measurement is detections[0]
        else:
            assert np.allclose(track.state.mean, np.array([[-5]]), atol=0.4)
            assert track.state.hypothesis.measurement is detections[1]
        assert track.timestamp == timestamp

        assert np.allclose(track.covar, np.array([[1]]), atol=0.4)


@pytest.mark.parametrize("gaussian_initiator", [
    SinglePointInitiator(
        GaussianState(np.array([[0]]), np.array([[100]])),
        LinearGaussian(1, [0], np.array([[1]]))
    ),
    SimpleMeasurementInitiator(
        GaussianState(np.array([[0]]), np.array([[100]])),
        LinearGaussian(1, [0], np.array([[1]]))
    ),
], ids=['SinglePoint', 'LinearMeasurement'])
def test_gaussian_mixture(gaussian_initiator):
    mixture_initiator = GaussianMixtureInitiator(gaussian_initiator)

    timestamp = datetime.datetime.now()
    detections = [Detection(np.array([[5]]), timestamp),
                  Detection(np.array([[-5]]), timestamp)]
    tracks = mixture_initiator.initiate(detections, timestamp)

    for track in tracks:
        assert isinstance(track.state, GaussianMixtureUpdate)

        if track.state.mean > 0:
            assert np.allclose(track.state.mean, np.array([[5]]), atol=0.4)
            assert track.state.hypothesis.measurement is detections[0]
        else:
            assert np.allclose(track.state.mean, np.array([[-5]]), atol=0.4)
            assert track.state.hypothesis.measurement is detections[1]
        assert track.timestamp == timestamp
        assert np.allclose(track.covar, np.array([[1]]), atol=0.4)


@pytest.mark.parametrize("gaussian_initiator", [
    SinglePointInitiator(
        GaussianState(np.array([[0]]), np.array([[100]])),
        LinearGaussian(1, [0], np.array([[1]]))
    ),
    SimpleMeasurementInitiator(
        GaussianState(np.array([[0]]), np.array([[100]])),
        LinearGaussian(1, [0], np.array([[1]]))
    ),
], ids=['SinglePoint', 'LinearMeasurement'])
def test_asd_gaussian(gaussian_initiator):
    timestamp = datetime.datetime.now()
    asd_initiator = ASDGaussianInitiator(initiator=gaussian_initiator)

    detections = [Detection(np.array([[5]]), timestamp),
                  Detection(np.array([[-5]]), timestamp)]
    tracks = asd_initiator.initiate(detections, timestamp)

    for track in tracks:
        assert isinstance(track.state, ASDGaussianStateUpdate)

        if track.state.mean > 0:
            assert np.allclose(track.state.mean, np.array([[5]]), atol=0.4)
            assert track.state.hypothesis.measurement is detections[0]
        else:
            assert np.allclose(track.state.mean, np.array([[-5]]), atol=0.4)
            assert track.state.hypothesis.measurement is detections[1]
        assert track.timestamp == timestamp
        assert np.allclose(track.covar, np.array([[1]]), atol=0.4)


@pytest.mark.parametrize("gaussian_initiator", [
    SinglePointInitiator(
        GaussianState(np.array([[0]]), np.array([[100]])),
        LinearGaussian(1, [0], np.array([[1]]))
    ),
    SimpleMeasurementInitiator(
        GaussianState(np.array([[0]]), np.array([[100]])),
        LinearGaussian(1, [0], np.array([[1]]))
    ),
], ids=['SinglePoint', 'LinearMeasurement'])
def test_ensemble_1d(gaussian_initiator):
    ensemble_initiator = EnsembleInitiator(gaussian_initiator)

    timestamp = datetime.datetime.now()
    detections = [Detection(np.array([[5]]), timestamp),
                  Detection(np.array([[-5]]), timestamp)]
    tracks = ensemble_initiator.initiate(detections, timestamp)

    for track in tracks:
        assert isinstance(track.state, EnsembleStateUpdate)

        if track.state.mean > 0:
            assert np.allclose(np.mean(track.state.mean), np.array([[5]]), atol=0.5)
            assert track.state.hypothesis.measurement is detections[0]
        else:
            assert np.allclose(np.mean(track.state.mean), np.array([[-5]]), atol=0.5)
            assert track.state.hypothesis.measurement is detections[1]
        assert track.timestamp == timestamp
        assert np.allclose(track.covar, np.array([[1]]), atol=0.6)


@pytest.mark.parametrize("gaussian_initiator", [
    SinglePointInitiator(
        GaussianState(np.array([[0], [0]]), np.diag([100, 0])),
        LinearGaussian(2, [0], np.diag([1]))
    ),
    SimpleMeasurementInitiator(
        GaussianState(np.array([[0], [0]]), np.diag([100, 0])),
        LinearGaussian(2, [0], np.diag([1]))
    )
], ids=['SinglePoint', 'LinearMeasurement'])
def test_ensemble_2d(gaussian_initiator):
    ensemble_initiator = EnsembleInitiator(gaussian_initiator,
                                           ensemble_size=100)

    timestamp = datetime.datetime.now()
    detections = [Detection(np.array([[5]]), timestamp),
                  Detection(np.array([[-5]]), timestamp)]
    tracks = ensemble_initiator.initiate(detections, timestamp)

    for track in tracks:
        assert isinstance(track.state, EnsembleStateUpdate)
        if track.state.mean[0] > 0:
            assert np.allclose(track.state.mean, np.array([[5], [0]]), atol=0.5)
            assert track.state.hypothesis.measurement is detections[0]
        else:
            assert np.allclose(track.state.mean, np.array([[-5], [0]]), atol=0.5)
            assert track.state.hypothesis.measurement is detections[1]
        assert track.timestamp == timestamp
        assert np.allclose(track.covar, np.diag([1, 0]), atol=0.6)
