from typing import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest

from stonesoup.models.base_driver import LevyDriver
from stonesoup.models.driver import GaussianDriver


@pytest.fixture
def std_normal_driver() -> GaussianDriver:
    return GaussianDriver(mu_W=0, sigma_W2=1, seed=0)


@pytest.fixture
def gaussian_driver() -> GaussianDriver:
    return GaussianDriver(mu_W=1.0, sigma_W2=4.0, seed=0)


def test_seed(std_normal_driver: GaussianDriver, gaussian_driver: GaussianDriver) -> None:
    assert std_normal_driver.seed == 0
    assert gaussian_driver.seed == 0


def test_characteristic_function(
    std_normal_driver: GaussianDriver, gaussian_driver: GaussianDriver
) -> None:
    characteristic_func = std_normal_driver.characteristic_func()
    w = 0.0
    result = characteristic_func(w)
    expected = np.exp(-0.5 * w**2)
    assert pytest.approx(expected) == result

    characteristic_func = gaussian_driver.characteristic_func()
    mu = 1.0
    sigma = 2.0
    w = 0.5
    result = characteristic_func(w)
    expected = np.exp(-1j * w * mu - 0.5 * sigma**2 * w**2)
    assert pytest.approx(expected) == result


def test_mean(
    std_normal_driver: GaussianDriver,
    gaussian_driver: GaussianDriver,
    mock_e_ft_func: Callable[..., np.ndarray],
) -> None:
    assert std_normal_driver.mu_W == 0.0
    assert gaussian_driver.mu_W == 1.0
    num_samples = 2
    mean = std_normal_driver.mean(
        mock_e_ft_func, dt=1, mu_W=std_normal_driver.mu_W, num_samples=num_samples
    )
    expected_mean = np.array([[[0.0], [0.0]], [[0.0], [0.0]]])
    assert np.allclose(mean, expected_mean)

    mean = std_normal_driver.mean(
        mock_e_ft_func, dt=1, mu_W=gaussian_driver.mu_W, num_samples=num_samples
    )
    expected_mean = np.array([[[1.0], [2.0]], [[1.0], [2.0]]])
    assert np.allclose(mean, expected_mean)


def test_covar(
    std_normal_driver: GaussianDriver,
    gaussian_driver: GaussianDriver,
    mock_e_ft_func: Callable[..., np.ndarray],
) -> None:
    assert std_normal_driver.mu_W == 0.0
    assert gaussian_driver.mu_W == 1.0
    num_samples = 1
    covar = std_normal_driver.covar(
        mock_e_ft_func, dt=1, sigma_W2=std_normal_driver.sigma_W2, num_samples=num_samples
    )
    expected_covar = np.array([[1.0, 2.0], [2.0, 4.0]])
    assert np.allclose(covar, expected_covar)

    covar = std_normal_driver.covar(
        mock_e_ft_func, dt=1, sigma_W2=gaussian_driver.sigma_W2, num_samples=num_samples
    )
    expected_covar = np.array([[1.0, 2.0], [2.0, 4.0]]) * 4
    assert np.allclose(covar, expected_covar)
