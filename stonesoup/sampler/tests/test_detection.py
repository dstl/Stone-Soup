import pytest
import numpy as np

from ..detection import DetectionSampler
from ..particle import ParticleSampler, GaussianDetectionParticleSampler
from ...models.measurement.nonlinear import CartesianToBearingRange
from ...types.detection import Detection
from ...types.state import State


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

    sampler = DetectionSampler(detection_sampler=detection_sampler, backup_sampler=backup_sampler)

    samples = sampler.sample(detections)

    # check the number of samples
    assert len(samples) == nsamples
    # check the dimensions of samples
    assert samples.state_vector.shape[0] == measurement_model.ndim_state
