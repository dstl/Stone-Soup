from ..base_driver import Latents, TruncatedCase, GaussianResidualApproxCase, PartialGaussianResidualApproxCase
from ..driver import GaussianDriver, AlphaStableNSMDriver, TemperedStableNVMDriver, GammaNVMDriver
from ...types.array import StateVector, StateVectors, CovarianceMatrix
from scipy.stats import levy_stable, norm, kstest, uniform
from scipy.special import gamma
import numpy as np
import pytest
import matplotlib.pyplot as plt
from contextlib import nullcontext as does_not_raise
from scipy.integrate import quad
from datetime import timedelta

def test_gaussian_driver():
    # Test mean() and covar()
    mu_W = np.array([1])
    sigma_W2 = np.eye(1)
    gd = GaussianDriver(mu_W=mu_W, sigma_W2=sigma_W2)
    dt = 1
    e_ft = lambda dt: dt * np.ones((1, 1, 1))
    seed = 13
    random_state = np.random.RandomState(seed)
    assert(np.allclose(gd.mean(e_ft_func=e_ft, dt=dt), StateVector(mu_W)))
    assert(np.allclose(gd.covar(e_ft_func=e_ft, dt=dt), CovarianceMatrix(sigma_W2)))

    with pytest.raises(NotImplementedError, match="truncation"):
        gd.c
    
    with pytest.raises(NotImplementedError, match="noise case"):
        gd.noise_case

    # Test noise sample shape
    mean = gd.mean(e_ft_func=e_ft, dt=dt, mu_W=mu_W)
    covar = gd.covar(e_ft_func=e_ft, dt=dt, sigma_W2=sigma_W2, mu_W=mu_W)
    assert(gd.rvs(mean, covar, random_state=random_state).shape == (1, 1))
    assert(gd.rvs(mean, covar, num_samples=3, random_state=random_state).shape == (1, 3))
    
    # Test noise sample values
    gd = GaussianDriver(mu_W=mu_W, sigma_W2=sigma_W2)
    rng = np.random.RandomState(seed)
    truth = rng.multivariate_normal(mu_W, sigma_W2, size=5).T
    truth = truth.view(StateVectors)
    # Reset rng
    rng = np.random.RandomState(seed)
    assert(np.allclose(gd.rvs(mean, covar, num_samples=5, random_state=rng), truth))


def raw_abs_moment_gaussian(alpha, mu, sigma):
  func = lambda x: (np.abs(x) ** alpha) * norm.pdf(x, loc=mu, scale=sigma)
  return quad(func, -6, 6)[0]


def signed_raw_abs_moment_gaussian(alpha, mu, sigma):
  func = lambda x: (np.sign(x) * np.abs(x) ** alpha) * norm.pdf(x, loc=mu, scale=sigma)
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


@pytest.mark.parametrize(
    "noise_case, threshold, expect, ks_result",
    [
        (PartialGaussianResidualApproxCase(), 1e-2, does_not_raise(), True),
        (GaussianResidualApproxCase(), 1e-2, does_not_raise(), True),
    ],
)
def test_as_nsm_driver(noise_case, threshold, expect, ks_result):
    seed = 0
    mu_W = 1
    sigma_W2 = 1
    alpha = 1.5
    c=100
    
    with expect:
        rng = np.random.RandomState(seed)
        asd = AlphaStableNSMDriver(mu_W=mu_W, sigma_W2=sigma_W2, c=c, alpha=alpha, noise_case=noise_case)
        dt = 1
        n_latents=1000
        ft = lambda dt, jtimes: np.ones_like(jtimes)[..., None, None]
        e_ft = lambda dt: dt * np.ones((1, 1))
        l = asd.sample_latents(dt=dt, num_samples=n_latents, random_state=rng)
        latents = Latents(num_samples=n_latents)
        latents.add(asd, *l)

        mean = asd.mean(e_ft_func=e_ft, ft_func=ft, dt=dt, latents=latents)
        covar = asd.covar(e_ft_func=e_ft, ft_func=ft, dt=dt, latents=latents)
        rvs = []
        for i in range(n_latents):
            tmp = asd.rvs(random_state=rng, e_ft_func=e_ft, ft_func=ft, dt=dt, latents=latents, covar=covar[i], mean=mean[i], num_samples=1)
            rvs.append(tmp[0])
        y = np.array(rvs)

        cdf = alpha_stable_cdf(alpha=alpha, mu=mu_W, sigma2=sigma_W2)
        results = kstest(y, cdf, N=n_latents) 
        assert((results.pvalue >= threshold) == ks_result) # 99% CI
 

# def test_ts_nvm_residuals():
#     seed = 0
#     alpha = 0.4
#     gamma = 1.35
#     beta = (gamma ** (1 / alpha)) / 2.
#     noise_case=1
#     mu_W = 0
#     sigma_W2 = 1
#     upper_limit = 200
#     lower_limit = 100
#     noise_case = 2
        
#     asd = TemperedStableNVMDriver(mu_W=mu_W, sigma_W2=sigma_W2, seed=seed, c=upper_limit, alpha=alpha, beta=beta, noise_case=noise_case)
#     dt = 1
#     n_latents=10000
#     ft = lambda dt, jtimes: np.ones_like(jtimes)[..., None, None]
#     e_ft = lambda dt: dt * np.ones((1, 1))
#     sizes, times = asd.sample_latents(dt=dt, num_samples=n_latents)
#     residual_jsizes = sizes[lower_limit:]
#     residual_jtimes = times[lower_limit:]
#     latents = Latents(num_samples=n_latents)
#     latents.add(asd, residual_jsizes, residual_jtimes)

#     mean = asd.mean(e_ft_func=e_ft, ft_func=ft, dt=dt, latents=latents)
#     covar = asd.covar(e_ft_func=e_ft, ft_func=ft, dt=dt, latents=latents)
#     rvs = []
#     for i in range(n_latents):
#         tmp = asd.rvs(e_ft_func=e_ft, ft_func=ft, dt=dt, latents=latents, covar=covar[i], mean=mean[i], num_samples=1)
#         rvs.append(tmp[0])
#     y = np.array(rvs)
#     y = (y - y.mean()) / y.std()
#     results = kstest(y, norm.cdf, N=n_latents) 
#     assert(results.pvalue >= 1e-2)
#     # fig, ax = plt.subplots(nrows=1, ncols=1)

#     # x = np.linspace(norm.ppf(0.001), norm.ppf(0.999), 100)
#     # # ax.plot(x, ls_rvs.pdf(x), 'r-', lw=5, alpha=0.6, label=r'$\alpha$-stable PDF')        
#     # ax.plot(x, norm.pdf(x), 'g-', lw=5, alpha=0.6, label='Standard normal PDF')    
#     # ax.hist(y, density=True, bins='auto', histtype='stepfilled', alpha=0.6)
#     # # ax.set_xlim([x[0], x[-1]])
#     # plt.show()


# def test_ng_nvm_residuals():
    return
    seed = 0
    mu_W = 0
    sigma_W2 = 1
    gamma = np.sqrt(2.)
    beta = (gamma ** 2) / 2.
    nu = 2
    upper_limit = 100
    lower_limit = 10 # c

    for noise_case in [2]:
        vgd =  GammaNVMDriver(mu_W=mu_W, sigma_W2=sigma_W2, seed=seed, c=upper_limit, nu=nu, beta=beta)
        dt = 1
        n_latents=10000
        ft = lambda dt, jtimes: np.ones_like(jtimes)[..., None, None]
        e_ft = lambda dt: dt * np.ones((1, 1))
        sizes, times = vgd.sample_latents(dt=dt, num_samples=n_latents)
        residual_jsizes = sizes[lower_limit:]
        residual_jtimes = times[lower_limit:]
        latents = Latents(num_samples=n_latents)
        latents.add(vgd, residual_jsizes, residual_jtimes)

        mean = vgd.mean(e_ft_func=e_ft, ft_func=ft, dt=dt, latents=latents)
        covar = vgd.covar(e_ft_func=e_ft, ft_func=ft, dt=dt, latents=latents)
        rvs = []
        for i in range(n_latents):
            tmp = vgd.rvs(e_ft_func=e_ft, ft_func=ft, dt=dt, latents=latents, covar=covar[i], mean=mean[i], num_samples=1)
            rvs.append(tmp[0])
        y = np.array(rvs)
        y = (y - y.mean()) / y.std()
        results = kstest(y, norm.cdf, N=n_latents) 
        assert(results.pvalue < 1e-2)
        # fig, ax = plt.subplots(nrows=1, ncols=1)

        # x = np.linspace(norm.ppf(0.001), norm.ppf(0.999), 100)
        # # ax.plot(x, ls_rvs.pdf(x), 'r-', lw=5, alpha=0.6, label=r'$\alpha$-stable PDF')        
        # ax.plot(x, norm.pdf(x), 'g-', lw=5, alpha=0.6, label='Standard normal PDF')    
        # ax.hist(y, density=True, bins='auto', histtype='stepfilled', alpha=0.6)
        # # ax.set_xlim([x[0], x[-1]])
        # plt.show()