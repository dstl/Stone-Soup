import datetime
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from ..kernel import AdaptiveKernelKalmanUpdater
from ...kernel import QuadraticKernel
from ...models.measurement.nonlinear import Cartesian2DToBearing
from ...models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from ...predictor.kernel import AdaptiveKernelKalmanPredictor
from ...types.array import StateVectors
from ...types.detection import Detection
from ...types.groundtruth import GroundTruthState
from ...types.hypothesis import SingleHypothesis
from ...types.state import KernelParticleState

timestamp = datetime.datetime.now()
time_diff = datetime.timedelta(seconds=2)  # 2sec
new_timestamp = timestamp + time_diff

number_particles = 5
samples = multivariate_normal.rvs([0, 0, 0, 0],
                                  np.diag([0.01, 0.005, 0.1, 0.5])**2,
                                  size=number_particles)
weights = np.array([1/number_particles]*number_particles)
prior = KernelParticleState(state_vector=StateVectors(samples.T),
                            weight=weights,
                            timestamp=timestamp)
samples = multivariate_normal.rvs([0, 0, 0, 0],
                                  np.diag([0.01, 0.005, 0.1, 0.5])**2,
                                  size=number_particles)
proposal = KernelParticleState(state_vector=StateVectors(samples.T),
                               weight=weights,
                               timestamp=timestamp)

predictor = AdaptiveKernelKalmanPredictor(
    transition_model=CombinedLinearGaussianTransitionModel(
                [ConstantVelocity(noise_diff_coeff=0.1),
                 ConstantVelocity(noise_diff_coeff=0.1)]),
    kernel=QuadraticKernel())
prediction = predictor.predict(prior=prior, proposal=proposal,
                               timestamp=new_timestamp, noise=False)


@pytest.mark.parametrize(
    "kernel, measurement_model, c, ialpha",
    [
        (   # No Kernel
            None,
            Cartesian2DToBearing(
                ndim_state=4,
                mapping=(0, 2),
                noise_covar=np.array([[0.005 ** 2]])),
            1,
            10
        ),
        (   # With Kernel
            QuadraticKernel(),
            None,
            1,
            10
        )
    ],
    ids=["measurement_model",
         "no_measurement_model",
         ]
)
def test_kernel_updater(kernel, measurement_model, c, ialpha):
    gt_state = GroundTruthState([-0.1, 0.001, 0.7, -0.055], timestamp=new_timestamp)
    mm = Cartesian2DToBearing(
                ndim_state=4,
                mapping=(0, 2),
                noise_covar=np.array([[0.005 ** 2]]))
    measurement = Detection(mm.function(gt_state, noise=True),
                            timestamp=gt_state.timestamp,
                            measurement_model=mm)
    updater = AdaptiveKernelKalmanUpdater(
        measurement_model=measurement_model,
        kernel=QuadraticKernel())
    hypothesis = SingleHypothesis(prediction, measurement)

    update = updater.update(hypothesis)
    hypothesis.measurement_prediction = updater.predict_measurement(
        prediction, measurement_model=mm)
    G_yy = updater.kernel(hypothesis.measurement_prediction)
    g_y = updater.kernel(hypothesis.measurement_prediction, hypothesis.measurement)

    Q_AKKF = prediction.kernel_covar @ np.linalg.pinv(
        G_yy @ prediction.kernel_covar + updater.lambda_updater * np.identity(
            len(prediction)))

    updated_weights = np.atleast_2d(prediction.weight).T + Q_AKKF @ (
                g_y - np.atleast_2d(G_yy @ prediction.weight).T)
    updated_covariance = \
        prediction.kernel_covar - Q_AKKF @ G_yy @ prediction.kernel_covar

    # Proposal Calculation
    pred_mean = prediction.state_vector @ np.squeeze(updated_weights)
    pred_covar = np.diag(np.diag(
        prediction.state_vector @ updated_covariance @ prediction.state_vector.T))

    new_state_vector = multivariate_normal.rvs(
        pred_mean, pred_covar, size=len(prediction)
    )
    assert hasattr(update, 'proposal')
    assert measurement.timestamp == gt_state.timestamp
    assert update.hypothesis.measurement.timestamp == gt_state.timestamp
    assert np.allclose(update.state_vector, prediction.state_vector)
    assert np.allclose(
        np.mean(update.proposal, axis=1),
        np.mean(StateVectors(new_state_vector.T), axis=1),
        atol=2)
    assert np.allclose(update.weight, updated_weights.ravel())
    assert np.allclose(update.kernel_covar, updated_covariance)


@pytest.mark.parametrize(
    "updater_class, kernel, measurement_model, c, ialpha",
    [
        (   # No Kernel
            AdaptiveKernelKalmanUpdater,
            None,
            Cartesian2DToBearing(
                ndim_state=4,
                mapping=(0, 2),
                noise_covar=np.array([[0.005 ** 2]])),
            1,
            10
        ),
        (   # With Kernel
            AdaptiveKernelKalmanUpdater,
            QuadraticKernel(),
            Cartesian2DToBearing(
                ndim_state=4,
                mapping=(0, 2),
                noise_covar=np.array([[0.005 ** 2]])),
            1,
            10
        ),
        (   # With Kernel(c=2, ialpha=20)
            AdaptiveKernelKalmanUpdater,
            QuadraticKernel(c=2, ialpha=20),
            Cartesian2DToBearing(
                ndim_state=4,
                mapping=(0, 2),
                noise_covar=np.array([[0.005 ** 2]])),
            2,
            20
        ),
    ],
    ids=["no_kernel",
         "kernel",
         "c2i20"]
)
def test_no_kernel(updater_class, kernel, measurement_model, c, ialpha):
    updater = updater_class(measurement_model=measurement_model,
                            kernel=kernel)
    assert isinstance(updater.kernel, QuadraticKernel)
    assert updater.kernel.c == c
    assert updater.kernel.ialpha == ialpha


def test_no_measurement_model():
    mm = Cartesian2DToBearing(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.array([[0.005 ** 2]]))

    updater = AdaptiveKernelKalmanUpdater(measurement_model=mm,
                                          kernel=QuadraticKernel())
    gt_state = GroundTruthState([-0.1, 0.001, 0.7, -0.055], timestamp=new_timestamp)

    measurement = Detection(mm.function(gt_state, noise=True),
                            timestamp=gt_state.timestamp,
                            measurement_model=mm)
    hypothesis = SingleHypothesis(prediction, measurement)
    hypothesis.measurement_prediction = updater.predict_measurement(
        prediction,
    )
    update = updater.update(hypothesis)
    assert hasattr(update, 'proposal')
    assert measurement.timestamp == gt_state.timestamp
    assert update.hypothesis.measurement.timestamp == gt_state.timestamp
    assert np.allclose(update.state_vector, prediction.state_vector)
