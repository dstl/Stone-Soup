import pytest
import numpy as np
import datetime

from ..detection import SwitchingDetectionSampler, GaussianDetectionParticleSampler
from ..particle import ParticleSampler
from ...models.measurement.nonlinear import CartesianToBearingRange
from ...types.detection import Detection
from ...types.state import State
from ...models.measurement.linear import LinearGaussian


@pytest.mark.parametrize(
    "measurement_model, states, nsamples",
    [
        (
            LinearGaussian(ndim_state=2,
                           mapping=(0, 1),
                           noise_covar=np.eye(2)),  # measurement_model
            [np.array([10, 10]), np.array([20, 20]), np.array([30, 30])],  # states
            20  # nsamples
        ), (
            LinearGaussian(ndim_state=2,
                           mapping=(0, 1),
                           noise_covar=np.eye(2)),  # measurement_model
            [np.array([10, 10]), np.array([20, 20]), np.array([30, 30])],  # states
            20  # nsamples
        ), (
            LinearGaussian(ndim_state=4,
                           mapping=(0, 3),
                           noise_covar=np.eye(2)),  # measurement_model
            [np.array([10, 0, 10, 0]),
             np.array([20, 0, 20, 0]),
             np.array([30, 0, 30, 0])],  # states
            20  # nsamples
        ), (
            CartesianToBearingRange(ndim_state=4,
                                    mapping=(0, 2),
                                    noise_covar=np.diag([0.01, 1])),  # measurement_model
            [np.array([10, 0, 10, 0]),
             np.array([20, 0, 20, 0]),
             np.array([30, 0, 30, 0])],  # states
            20  # nsamples
        )
    ],
    ids=["linear_1_detection", "linear_3_detections", "linear_3_partial_state_detections",
         "nonlinear_3_detections"]
)
def test_gaussian_detection_sampler(measurement_model, states, nsamples):

    timestamp_now = datetime.datetime.now()
    detections = set()
    for ii in range(3):
        detections |= {Detection(measurement_model.function(State(states[ii])),
                                 measurement_model=measurement_model,
                                 timestamp=timestamp_now)}

    sampler = GaussianDetectionParticleSampler(nsamples=nsamples)

    particles = sampler.sample(detections)
    # check number of particles
    assert len(particles) == nsamples
    # check state vector dimensions
    assert particles.state_vector.shape[0] == states[0].shape[0]
    # check timestamp
    assert particles.timestamp == timestamp_now


@pytest.mark.parametrize(
    "states, measurement_model, nsamples",
    [
        (
            [np.array([10, 0, 10, 0])],  # states
            CartesianToBearingRange(ndim_state=4,
                                    mapping=(0, 2),
                                    noise_covar=np.diag([0.01, 1])),  # measurement_model
            20  # nsamples
        ), (
            [np.array([10, 0, 10, 0]),
             np.array([20, 0, 20, 0]),
             np.array([30, 0, 30, 0])],  # states
            CartesianToBearingRange(ndim_state=4,
                                    mapping=(0, 2),
                                    noise_covar=np.diag([0.01, 1])),  # measurement_model
            20  # nsamples
        ), (
            None,  # states
            CartesianToBearingRange(ndim_state=4,
                                    mapping=(0, 2),
                                    noise_covar=np.diag([0.01, 1])),  # measurement_model
            20
        )
    ], ids=["single_detection", "multiple_detections", "no_detections"]
)
def test_detection_sampler(states, measurement_model, nsamples):
    detections = set()
    if states is not None:
        for ii in range(len(states)):
            detections |= {Detection(measurement_model.function(State(states[ii])),
                                     measurement_model=measurement_model)}

    ndim_state = measurement_model.ndim_state

    detection_sampler = GaussianDetectionParticleSampler(nsamples=nsamples)
    backup_sampler = ParticleSampler(distribution_func=np.random.uniform,
                                     params={'low': np.array([-20, 0, -20, 0]),
                                             'high': np.array([20, 0, 20, 0]),
                                             'size': (nsamples, ndim_state)},
                                     ndim_state=ndim_state)

    sampler = SwitchingDetectionSampler(detection_sampler=detection_sampler,
                                        backup_sampler=backup_sampler)

    samples = sampler.sample(detections)

    # check the number of samples
    assert len(samples) == nsamples
    # check the dimensions of samples
    assert samples.state_vector.shape[0] == measurement_model.ndim_state
