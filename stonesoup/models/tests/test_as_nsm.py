import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gamma
from scipy.stats import kstest, levy_stable, norm

from stonesoup.models.base_driver import NoiseCase
from stonesoup.models.driver import AlphaStableNSMDriver
from stonesoup.types.array import (
    CovarianceMatrices,
    CovarianceMatrix,
    StateVector,
    StateVectors,
)


@pytest.fixture
def alpha() -> float:
    return 1.5


@pytest.fixture(
    scope="function",
    params=[
        NoiseCase.GAUSSIAN_APPROX,
        NoiseCase.PARTIAL_GAUSSIAN_APPROX,
        NoiseCase.TRUNCATED,
    ],
)
def alpha_stable_nsm_driver(
    request: pytest.FixtureRequest,
    seed: int,
    mu_W: float,
    sigma_W2: float,
    expected_jumps_per_sec: int,
    alpha: float,
):
    return AlphaStableNSMDriver(
        seed=seed,
        mu_W=mu_W,
        sigma_W2=sigma_W2,
        c=expected_jumps_per_sec,
        alpha=alpha,
        noise_case=request.param,
    )


def test_mean_covar(alpha_stable_nsm_driver: AlphaStableNSMDriver):
    def ft(dt: int, jtimes: np.array):
        return np.ones_like(jtimes)[..., None, None]

    def e_ft(dt: int):
        return dt * np.ones((1, 1))

    dt = 1
    n_samples = 2
    jsizes, jtimes = alpha_stable_nsm_driver.sample_latents(dt=dt, num_samples=n_samples)
    mean = alpha_stable_nsm_driver.mean(
        jsizes=jsizes, jtimes=jtimes, e_ft_func=e_ft, ft_func=ft, dt=dt
    )
    covar = alpha_stable_nsm_driver.covar(
        jsizes=jsizes, jtimes=jtimes, e_ft_func=e_ft, ft_func=ft, dt=dt
    )
    expected_mean = StateVectors([[[-1.2177414]], [[-1.80693522]]])

    assert isinstance(mean, StateVectors)
    assert np.allclose(mean, expected_mean)

    assert isinstance(covar, CovarianceMatrices)
    if alpha_stable_nsm_driver.noise_case == NoiseCase.GAUSSIAN_APPROX:
        expected_covar = CovarianceMatrices([[[8.04448152]], [[6.70822924]]])
    elif alpha_stable_nsm_driver.noise_case == NoiseCase.PARTIAL_GAUSSIAN_APPROX:
        expected_covar = CovarianceMatrices([[[5.66337994]], [[4.32712766]]])
    elif alpha_stable_nsm_driver.noise_case == NoiseCase.TRUNCATED:
        expected_covar = CovarianceMatrices([[[3.28227836]], [[1.94602608]]])
    else:
        raise RuntimeError(f"Noise case not tested {alpha_stable_nsm_driver.noise_case}")
    assert np.allclose(covar, expected_covar)

    n_samples = 1
    jsizes, jtimes = alpha_stable_nsm_driver.sample_latents(dt=dt, num_samples=n_samples)
    mean = alpha_stable_nsm_driver.mean(
        jsizes=jsizes, jtimes=jtimes, e_ft_func=e_ft, ft_func=ft, dt=dt
    )
    covar = alpha_stable_nsm_driver.covar(
        jsizes=jsizes, jtimes=jtimes, e_ft_func=e_ft, ft_func=ft, dt=dt
    )
    assert isinstance(mean, StateVector)
    assert isinstance(covar, CovarianceMatrix)


def test_sample_latents(
    expected_jumps_per_sec, alpha_stable_nsm_driver: AlphaStableNSMDriver
):
    # Use a seperate instance other than fixture with c =
    dt = 1
    num_samples = 2
    jsizes, jtimes = alpha_stable_nsm_driver.sample_latents(
        dt=dt, num_samples=num_samples
    )
    expected_jsizes = np.array([[1.29327154, 0.98714497], [1.26875021, 0.98568295]])
    expected_jtimes = np.array([[0.54362499, 0.93507242], [0.81585355, 0.0027385]])
    assert jsizes.shape == jtimes.shape
    assert jsizes.shape[0] == expected_jumps_per_sec
    assert jsizes.shape[1] == num_samples
    assert np.allclose(jsizes, expected_jsizes)
    assert np.allclose(jtimes, expected_jtimes)


def raw_abs_moment_gaussian(alpha, mu, sigma):
    def func(x):
        return (np.abs(x) ** alpha) * norm.pdf(x, loc=mu, scale=sigma)

    return quad(func, -6, 6)[0]


def signed_raw_abs_moment_gaussian(alpha, mu, sigma):
    def func(x):
        return (np.sign(x) * np.abs(x) ** alpha) * norm.pdf(x, loc=mu, scale=sigma)

    return quad(func, -6, 6)[0]


def alpha_stable_cdf(alpha, mu, sigma2):
    # Compute scale
    sigma = np.sqrt(sigma2)
    raw_moments = raw_abs_moment_gaussian(alpha, mu, sigma)
    C_alpha = (1 - alpha) / (gamma(2 - alpha) * np.cos(np.pi * alpha / 2))
    scale = (raw_moments / C_alpha) ** (1 / alpha)

    # Compute skew (beta)
    signed_raw_moments = signed_raw_abs_moment_gaussian(alpha, mu, sigma)
    raw_moments = raw_abs_moment_gaussian(alpha, mu, sigma)
    skew = signed_raw_moments / raw_moments

    dist = levy_stable(alpha=alpha, beta=skew, scale=scale)
    return dist.cdf


def test_rvs(alpha_stable_nsm_driver: AlphaStableNSMDriver):
    """Tests if resulting noise distribution is correct"""

    def ft(dt: int, jtimes: np.array):
        return np.ones_like(jtimes)[..., None, None]

    def e_ft(dt: int):
        return dt * np.ones((1, 1))

    alpha_stable_nsm_driver.c = 100
    num_samples = 1000
    dt = 1
    jsizes, jtimes = alpha_stable_nsm_driver.sample_latents(
        dt=dt, num_samples=num_samples
    )
    assert jtimes.shape == jsizes.shape
    assert jtimes.shape == (alpha_stable_nsm_driver.c, num_samples)
    rvs_samples = []
    for i in range(num_samples):
        a = jsizes[:, i: i + 1]
        b = jtimes[:, i: i + 1]
        mean = alpha_stable_nsm_driver.mean(
            jsizes=a, jtimes=b, e_ft_func=e_ft, ft_func=ft, dt=dt
        )
        covar = alpha_stable_nsm_driver.covar(
            jsizes=a, jtimes=b, e_ft_func=e_ft, ft_func=ft, dt=dt
        )
        noise = alpha_stable_nsm_driver.rvs(mean=mean, covar=covar, num_samples=1)
        rvs_samples.append(noise)
    rvs_samples = np.array(rvs_samples)

    threshold = 1e-2
    cdf = alpha_stable_cdf(
        alpha=alpha_stable_nsm_driver.alpha,
        mu=alpha_stable_nsm_driver.mu_W,
        sigma2=alpha_stable_nsm_driver.mu_W,
    )
    results = kstest(rvs_samples, cdf, N=num_samples)

    if alpha_stable_nsm_driver.noise_case == NoiseCase.GAUSSIAN_APPROX:
        assert results.pvalue >= threshold  # 99% CI
    elif alpha_stable_nsm_driver.noise_case == NoiseCase.PARTIAL_GAUSSIAN_APPROX:
        assert results.pvalue >= threshold
    elif alpha_stable_nsm_driver.noise_case == NoiseCase.TRUNCATED:
        assert results.pvalue < threshold
    else:
        raise RuntimeError(f"Noise case not tested {alpha_stable_nsm_driver.noise_case}")
