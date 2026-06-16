import datetime

import numpy as np
import pytest

from ..kalman import ExtendedKalmanUpdater
from ..multi import ParallelUpdater, CombineMeasurementUpdater
from ...types.detection import Detection
from ...types.hypothesis import SingleHypothesis
from ...types.prediction import GaussianStatePrediction
from ...types.state import State
from ...models.measurement.linear import LinearGaussian
from ...models.measurement.nonlinear import CartesianToBearingRange


@pytest.fixture(params=['linear', 'nonlinear'])
def single_updater(request):
    return request.param(None)


@pytest.fixture(params=['combine', 'parallel', 'parallel_multip'])
def multi_updater(request):
    if request.param == 'combine':
        return CombineMeasurementUpdater(ExtendedKalmanUpdater(None))
    elif request.param.startswith('parallel'):
        return ParallelUpdater(
            ExtendedKalmanUpdater(None), multiprocessing=request.param.endswith('multip'))


@pytest.fixture(params=['linear', 'nonlinear'])
def models(request):
    if request.param == 'linear':
        return [
            LinearGaussian(4, [0, 2], np.diag([1.5, 1.5])),
            LinearGaussian(4, [0, 2], np.diag([0.5, 0.5]))
        ]
    elif request.param == 'nonlinear':
        return [
            CartesianToBearingRange(
                4, [0, 2], np.diag([0.01, 1.5]), translation_offset=np.array([[10], [0]])),
            CartesianToBearingRange(
                4, [0, 2], np.diag([0.01, 0.5]), translation_offset=np.array([[0], [-5]]))
        ]


@pytest.mark.parametrize('n_dets', [1, 2])
def test_multi_updater(multi_updater, models, n_dets):
    timestamp = datetime.datetime(2025, 4, 3, 12)
    gt = State([5, 0, 5, 0])

    prediction = GaussianStatePrediction(
        [4.6, 0.5, 5.6, -0.5],
        np.diag([1., 0.25, 1., 0.25]),
        timestamp
        )

    detections = [Detection(model.function(gt), timestamp, model) for model in models][:n_dets]

    assert multi_updater.predict_measurement(prediction, models[0]) \
        is multi_updater.updater.predict_measurement(prediction, models[0])
    assert multi_updater.predict_measurement(prediction, models[1]) \
        is multi_updater.updater.predict_measurement(prediction, models[1])

    seq_post = prediction
    for detection in detections:
        seq_post = multi_updater.updater.update(SingleHypothesis(seq_post, detection))

    mul_post = multi_updater.update(
        [SingleHypothesis(prediction, detection) for detection in detections])

    if isinstance(models[0], LinearGaussian):
        rtol = 1e-9  # Linear should be identical...
    else:
        rtol = 1e-3  # Non-linear non-optimal due to linearisation

    assert np.allclose(mul_post.state_vector, seq_post.state_vector, rtol=rtol)


def test_multi_updater_multi_predictions(multi_updater):
    model = LinearGaussian(1, [0], [[1]])
    t = datetime.datetime(2025, 4, 3, 12)
    with pytest.warns(match="More than one prediction"):
        multi_updater.update([
            SingleHypothesis(GaussianStatePrediction([[1]], [[2]], t), Detection([[0]], t, model)),
            SingleHypothesis(GaussianStatePrediction([[1]], [[3]], t), Detection([[0]], t, model)),
        ])
