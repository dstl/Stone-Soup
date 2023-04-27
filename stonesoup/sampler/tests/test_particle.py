import numpy as np
from scipy.stats import multivariate_normal, uniform
import pytest
import datetime

from ..particle import ParticleSampler, GaussianDetectionParticleSampler
from ...types.detection import Detection
from ...types.state import State
from ...models.measurement.nonlinear import CartesianToBearingRange
from ...models.measurement.linear import LinearGaussian


@pytest.mark.parametrize(
    "distribution_func, params, dist_override, params_override, ndim_state, ndim_state_override,"
    "nparts, timestamp",
    [
        (
            np.random.uniform,  # distribution_func
            {'low': -1, 'high': 1, 'size': 20},  # params
            None,  # dist_override
            None,  # params_override
            1,  # ndim_state
            None,  # ndim_state_override
            20,  # nparts
            datetime.datetime.now()  # timestamp
        ), (
            np.random.uniform,  # distribution_func
            {'low': np.array([-1, -2]), 'high': np.array([1, 2]), 'size': (20, 2)},  # params
            None,  # dist_override
            None,  # params_override
            2,  # ndim_state
            None,  # ndim_state_override
            20,  # nparts
            None  # timestamp
        ), (
            np.random.normal,  # distribution_func
            {'loc': 2, 'scale': 1, 'size': 10},  # params
            None,  # dist_override
            None,  # params_override
            1,  # ndim_state
            None,  # ndim_state_override
            10,  # nparts
            None  # timestamp
        ), (
            np.random.multivariate_normal,  # distribution_func
            {'mean': np.array([5, 5]), 'cov': np.eye(2), 'size': 20},  # params
            None,  # dist_override
            None,  # params_override
            2,  # ndim_state
            None,  # ndim_state_override
            20,  # nparts
            datetime.datetime.now()  # timestamp
        ), (
            np.random.multivariate_normal,  # distribution_func
            {'mean': np.array([5, 5]), 'cov': np.eye(2), 'size': 20},  # params
            np.random.uniform,  # dist_override
            {'low': -1, 'high': 1, 'size': 20},  # params_override
            2,  # ndim_state
            1,  # ndim_state_override
            20,  # nparts
            None  # timestamp
        ), (
            np.random.multivariate_normal,  # distribution_func
            {'mean': np.array([5, 5]), 'cov': np.eye(2), 'size': 20},  # params
            np.random.uniform,  # dist_override
            None,  # params_override
            2,  # ndim_state
            1,  # ndim_state_override
            20,  # nparts
            datetime.datetime.now()  # timestamp
        ), (
            np.random.multivariate_normal,  # distribution_func
            {'mean': np.array([5, 5]), 'cov': np.eye(2), 'size': 20},  # params
            np.random.uniform,  # dist_override
            {'low': -1, 'high': 1, 'size': 20},  # params_override
            2,  # ndim_state
            None,  # ndim_state_override
            20,  # nparts
            None  # timestamp
        ), (
            np.random.multivariate_normal,  # distribution_func
            {'mean': np.array([5, 5]), 'cov': np.eye(2), 'size': 20},  # params
            None,  # dist_override
            {'mean': np.array([10, 10]), 'size': 40},  # params_override
            2,  # ndim_state
            None,  # ndim_state_override
            20,  # nparts
            None  # timestamp
        ), (
            uniform.rvs,  # distribution_func
            {'loc': -1, 'scale': 2, 'size': 20},  # params
            None,  # dist_override
            None,  # params_override
            1,  # ndim_state
            None,  # ndim_state_override
            20,  # nparts
            None  # timestamp
        ), (
            multivariate_normal.rvs,  # distribution_func
            {'mean': np.array([5, 5]), 'cov': np.eye(2), 'size': 20},  # params
            None,  # dist_override
            None,  # params_override
            2,  # ndim_state
            None,  # ndim_state_override
            20,  # nparts
            None  # timestamp
        )
    ],
    ids=["numpy_uniform_1d", "numpy_uniform_2d", "numpy_normal", "numpy_multivariate_normal",
         "numpy_multivariate_normal_override", "override_without_params",
         "override_without_ndim_state", "param_update", "scipy_uniform",
         "scipy_multivariate_normal"]
)
def test_sampler(distribution_func, params, dist_override, params_override, ndim_state,
                 ndim_state_override, nparts, timestamp):

    sampler = ParticleSampler(distribution_func=distribution_func,
                              params=params,
                              ndim_state=ndim_state)

    if timestamp is None:
        particles = sampler.sample()
    else:
        particles = sampler.sample(timestamp=timestamp)

    # check number of particles
    assert len(particles) == nparts
    # check shape of state_vector
    assert np.shape(particles.state_vector) == (ndim_state, nparts)
    # check timestamp
    assert particles.timestamp == timestamp

    # check that override works correctly
    if dist_override is not None:
        if params_override is None:
            with pytest.raises(ValueError) as excinfo:
                new_particles = sampler.sample(distribution_func=dist_override,
                                               ndim_state=ndim_state_override)
            assert "New distribution_func provided without params" in str(excinfo.value)
            return

        if ndim_state_override is None:
            with pytest.raises(ValueError) as excinfo:
                new_particles = sampler.sample(distribution_func=dist_override,
                                               params=params_override,
                                               timestamp=timestamp)
            assert "New distribution_func provided without ndim_state" in str(excinfo.value)
            return

        new_particles = sampler.sample(distribution_func=dist_override,
                                       params=params_override,
                                       ndim_state=ndim_state_override,
                                       timestamp=timestamp)
        # check shape of output
        assert np.shape(new_particles.state_vector) == (ndim_state_override,
                                                        params_override['size'])
        # check the timestamp
        assert new_particles.timestamp == timestamp
    elif params_override is not None:
        new_particles = sampler.sample(params=params_override, timestamp=timestamp)

        # check shape of state vector
        assert np.shape(new_particles.state_vector) == (ndim_state, params_override['size'])
        # check timestamp
        assert new_particles.timestamp == timestamp


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
