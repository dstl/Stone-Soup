import numpy as np
from scipy.stats import multivariate_normal, uniform
import pytest
import datetime

from ..particle import ParticleSampler


@pytest.mark.parametrize(
    "distribution_func, params, params_override, ndim_state, nparts, timestamp",
    [
        (
            np.random.uniform,  # distribution_func
            {'low': -1, 'high': 1, 'size': 20},  # params
            None,  # params_override
            1,  # ndim_state
            20,  # nparts
            datetime.datetime.now()  # timestamp
        ), (
            np.random.uniform,  # distribution_func
            {'low': np.array([-1, -2]), 'high': np.array([1, 2]), 'size': (20, 2)},  # params
            None,  # params_override
            2,  # ndim_state
            20,  # nparts
            None  # timestamp
        ), (
            np.random.normal,  # distribution_func
            {'loc': 2, 'scale': 1, 'size': 10},  # params
            None,  # params_override
            1,  # ndim_state
            10,  # nparts
            None  # timestamp
        ), (
            np.random.multivariate_normal,  # distribution_func
            {'mean': np.array([5, 5]), 'cov': np.eye(2), 'size': 20},  # params
            None,  # params_override
            2,  # ndim_state
            20,  # nparts
            datetime.datetime.now()  # timestamp
        ), (
            np.random.multivariate_normal,  # distribution_func
            {'mean': np.array([5, 5]), 'cov': np.eye(2), 'size': 20},  # params
            {'mean': np.array([10, 10]), 'size': 40},  # params_override
            2,  # ndim_state
            20,  # nparts
            None  # timestamp
        ), (
            uniform.rvs,  # distribution_func
            {'loc': -1, 'scale': 2, 'size': 20},  # params
            None,  # params_override
            1,  # ndim_state
            20,  # nparts
            None  # timestamp
        ), (
            multivariate_normal.rvs,  # distribution_func
            {'mean': np.array([5, 5]), 'cov': np.eye(2), 'size': 20},  # params
            None,  # params_override
            2,  # ndim_state
            20,  # nparts
            None  # timestamp
        )
    ],
    ids=["numpy_uniform_1d", "numpy_uniform_2d", "numpy_normal", "numpy_multivariate_normal",
         "param_update", "scipy_uniform", "scipy_multivariate_normal"]
)
def test_sampler(distribution_func, params, params_override, ndim_state, nparts, timestamp):

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
    if params_override is not None:
        new_particles = sampler.sample(params=params_override, timestamp=timestamp)

        # check shape of state vector
        assert np.shape(new_particles.state_vector) == (ndim_state, params_override['size'])
        # check timestamp
        assert new_particles.timestamp == timestamp
