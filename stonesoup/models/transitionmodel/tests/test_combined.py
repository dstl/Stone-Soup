# coding: utf-8
import datetime

import numpy as np

from stonesoup.models.transitionmodel.linear import LinearGaussianTimeInvariant, ConstantVelocity1D, ConstantVelocity2D, Combined


def test_combined(): 
    F = 3*np.eye(3)
    Q = 3*np.eye(3)
    model_1 = LinearGaussianTimeInvariant(transition_matrix=F, covariance_matrix=Q)
    model_2 = ConstantVelocity1D(noise_diff_coeff=3)
    model_3 = ConstantVelocity2D(noise_diff_coeff=1)

    DIM = 9
    
    combined_model = Combined([model_1, model_2, model_3])
    t_delta = datetime.timedelta(0,3)

    x_prior = np.ones([DIM,1])
    x_post = np.ones([DIM,1])

    assert np.array_equal(DIM, combined_model.ndim_state)
    assert np.array_equal((DIM,DIM), combined_model.matrix(time_interval=t_delta).shape)
    assert np.array_equal((DIM,DIM), combined_model.covar(time_interval=t_delta).shape)
    assert np.array_equal((DIM,1), combined_model.function(x_prior, noise=np.random.randn(DIM,1), time_interval=t_delta).shape)
    assert np.array_equal((DIM,1), combined_model.rvs(time_interval=t_delta).shape)
    assert type(combined_model.pdf(x_post, x_prior, time_interval=t_delta)) is np.float64


test_combined()
