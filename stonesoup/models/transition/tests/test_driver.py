from ..base_driver import GaussianDriver
from ....types.array import StateVector, StateVectors, CovarianceMatrix
import numpy as np
import pytest


def test_gaussian_driver():
    # Test invalid dimensions
    mu_W = np.array([[1], [1]])
    sigma_W2 = np.array([[1], [1]])
    with pytest.raises(AttributeError, match='covariance matrix sigma_W2 must be square'):
        GaussianDriver(mu_W=mu_W, sigma_W2=sigma_W2)

    mu_W = np.array([1])
    sigma_W2 = np.eye(2)
    with pytest.raises(AttributeError, match='ndim of mu_W must match sigma_W2'):
        GaussianDriver(mu_W=mu_W, sigma_W2=sigma_W2)

    # Test mean() and covar()
    mu_W = np.array([1])
    sigma_W2 = np.eye(1)
    gd = GaussianDriver(mu_W=mu_W, sigma_W2=sigma_W2)
    dt = 1
    e_gt = lambda dt: dt * np.ones((1, 1))
    e_gt2 = lambda dt: dt * np.eye(1)
    assert(np.allclose(gd.mean(e_gt_func=e_gt, dt=dt), StateVector(mu_W)))
    assert(np.allclose(gd.covar(e_gt_func=e_gt, dt=dt), CovarianceMatrix(sigma_W2)))

    # Test noise sample shape
    mean = gd.mean(e_gt_func=e_gt, dt=dt)
    covar = gd.covar(e_gt_func=e_gt, dt=dt)
    assert(gd.rvs(mean, covar).shape == (1, 1))
    assert(gd.rvs(mean, covar, 3).shape == (1, 3))
    
    # Test noise sample values
    seed = 13
    gd = GaussianDriver(mu_W=mu_W, sigma_W2=sigma_W2, seed=seed)
    rng = np.random.default_rng(seed)
    truth = rng.multivariate_normal(mu_W, sigma_W2, size=5).T
    truth = truth.view(StateVectors)
    assert(np.allclose(gd.rvs(mean, covar, 5), truth))


def test_alpha_stable_driver():
    pass