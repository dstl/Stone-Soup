import pytest
import numpy as np

from scipy.stats import multivariate_normal

from ..types.array import StateVectors
from ..types.state import KernelParticleState
from ..kernel import (Kernel, MultiplicativeKernel, AdditiveKernel,
                      PolynomialKernel, LinearKernel, QuadraticKernel, QuarticKernel,
                      GaussianKernel)

number_particles = 4
rng = np.random.RandomState(50)
samples = multivariate_normal.rvs([0, 0, 0, 0],
                                  np.diag([0.01, 0.005, 0.1, 0.5])**2,
                                  size=number_particles,
                                  random_state=rng)
prior = KernelParticleState(state_vector=StateVectors(samples.T),
                            weight=np.array([1/number_particles]*number_particles),
                            )
samples = multivariate_normal.rvs([0, 0, 0, 0],
                                  np.diag([0.01, 0.005, 0.1, 0.5])**2,
                                  size=number_particles,
                                  random_state=rng)
proposal = KernelParticleState(state_vector=StateVectors(samples.T),
                               weight=np.array([1/number_particles]*number_particles),
                               )


@pytest.mark.parametrize(
    "kernel_class, output, state1, state2",
    [
        (
            GaussianKernel(),
            StateVectors([[0.12615663, 0.12615343, 0.12602984, 0.11616877],
                          [0.12615343, 0.12615663, 0.12600953, 0.11633711],
                          [0.12602984, 0.12600953, 0.12615663, 0.11659654],
                          [0.11616877, 0.11633711, 0.11659654, 0.12615663]]),
            prior,
            None
        ),
        (
            GaussianKernel(),
            StateVectors([[0.12615332, 0.12615361, 0.12603986, 0.10961454],
                          [0.12615522, 0.12615549, 0.12605789, 0.10955370],
                          [0.12602568, 0.12600645, 0.12583658, 0.10955982],
                          [0.11624686, 0.11614979, 0.11543241, 0.08696717]]),
            prior,
            proposal
        ),
        (
            QuadraticKernel(),
            StateVectors([[1.12546138, 0.89290663, 1.10262155, 0.94640177],
                          [0.89290663, 1.10265966, 0.91273369, 1.05014420],
                          [1.10262155, 0.91273369, 1.08758008, 0.95670073],
                          [0.94640177, 1.05014420, 0.95670073, 1.02467909]]),
            prior,
            None
        ),
        (
            QuadraticKernel(),
            StateVectors([[0.75809725, 1.03096874, 0.98911714, 1.01165201],
                          [1.24660020, 0.97224740, 1.00961491, 0.98879035],
                          [0.79646728, 1.02501790, 0.99032626, 1.00779316],
                          [1.11808890, 0.98623137, 1.00469461, 0.99431764]]),
            prior,
            proposal
        ),
        (
            QuarticKernel(),
            StateVectors([[1.26666331, 0.79728225, 1.21577428, 0.89567631],
                          [0.79728225, 1.21585832, 0.83308279, 1.10280285],
                          [1.21577428, 0.83308279, 1.18283043, 0.91527629],
                          [0.89567631, 1.10280285, 0.91527629, 1.04996723]]),
            prior,
            None
        ),
        (
            QuarticKernel(),
            StateVectors([[0.57471144, 1.06289653, 0.97835271, 1.02343979],
                          [1.55401205, 0.94526501, 1.01932226, 0.97770636],
                          [0.63436013, 1.05066170, 0.98074609, 1.01564705],
                          [1.25012279, 0.97265232, 1.00941125, 0.98866757]]),
            prior,
            proposal
        ),
        (
            GaussianKernel(),
            StateVectors([[0.12615332, 0.12615361, 0.12603986, 0.10961454],
                          [0.12615522, 0.12615549, 0.12605789, 0.10955370],
                          [0.12602568, 0.12600645, 0.12583658, 0.10955982],
                          [0.11624686, 0.11614979, 0.11543241, 0.08696717]]),
            prior.state_vector,
            proposal
        ),
        (
            GaussianKernel(),
            StateVectors([[0.12615332, 0.12615361, 0.12603986, 0.10961454],
                          [0.12615522, 0.12615549, 0.12605789, 0.10955370],
                          [0.12602568, 0.12600645, 0.12583658, 0.10955982],
                          [0.11624686, 0.11614979, 0.11543241, 0.08696717]]),
            prior,
            proposal.state_vector
        ),
        (
            QuadraticKernel(),
            StateVectors([[0.75809725, 1.03096874, 0.98911714, 1.01165201],
                          [1.24660020, 0.97224740, 1.00961491, 0.98879035],
                          [0.79646728, 1.02501790, 0.99032626, 1.00779316],
                          [1.11808890, 0.98623137, 1.00469461, 0.99431764]]),
            prior.state_vector,
            proposal
        ),
        (
            QuadraticKernel(),
            StateVectors([[0.75809725, 1.03096874, 0.98911714, 1.01165201],
                          [1.24660020, 0.97224740, 1.00961491, 0.98879035],
                          [0.79646728, 1.02501790, 0.99032626, 1.00779316],
                          [1.11808890, 0.98623137, 1.00469461, 0.99431764]]),
            prior,
            proposal.state_vector
        ),
        (
            QuarticKernel(),
            StateVectors([[0.57471144, 1.06289653, 0.97835271, 1.02343979],
                          [1.55401205, 0.94526501, 1.01932226, 0.97770636],
                          [0.63436013, 1.05066170, 0.98074609, 1.01564705],
                          [1.25012279, 0.97265232, 1.00941125, 0.98866757]]),
            prior.state_vector,
            proposal
        ),
        (
            QuarticKernel(),
            StateVectors([[0.57471144, 1.06289653, 0.97835271, 1.02343979],
                          [1.55401205, 0.94526501, 1.01932226, 0.97770636],
                          [0.63436013, 1.05066170, 0.98074609, 1.01564705],
                          [1.25012279, 0.97265232, 1.00941125, 0.98866757]]),
            prior,
            proposal.state_vector
        )
    ],
    ids=["gaussian", "gaussian_w_prop",
         "quadratic", "quadratic_w_prop",
         "quartic", "quartic_w_prop",
         "gaussian_prior_as_state_vector", "gaussian_proposal_as_state_vector",
         "quadratic_prior_as_state_vector", "quadratic_proposal_as_state_vector",
         "quartic_prior_as_state_vector", "quartic_proposal_as_state_vector"]
)
def test_kernel(kernel_class, output, state1, state2):
    kernel = kernel_class(state1, state2)
    assert np.allclose(kernel, output)


def test_not_implemented():
    with pytest.raises(TypeError):
        Kernel()


@pytest.mark.parametrize(
    "power",
    [1, 2, 3, 4, 5],
    ids=["1", "2", "3", "4", "5"]
)
def test_multiplicative_kernel(power):
    linear_kernel = LinearKernel()
    linear_kernel_list = [linear_kernel] * power
    multiplicative_kernel = MultiplicativeKernel(kernel_list=linear_kernel_list)
    polynomial_kernel = PolynomialKernel(power=power, c=0, ialpha=1)
    state1 = StateVectors([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    state2 = StateVectors([[2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])
    linear_covar = linear_kernel(state1, state2)**power
    multiplicative_covar = multiplicative_kernel(state1, state2)
    polynomial_covar = polynomial_kernel(state1, state2)

    assert np.allclose(linear_covar, multiplicative_covar)
    assert np.allclose(multiplicative_covar, polynomial_covar)
    assert np.allclose(linear_covar, polynomial_covar)

    if power == 2:
        quadratic_kernel = QuadraticKernel(c=0, ialpha=1)
        quadratic_covar = quadratic_kernel(state1, state2)

        assert np.allclose(linear_covar, quadratic_covar)
        assert np.allclose(multiplicative_covar, quadratic_covar)
        assert np.allclose(polynomial_covar, quadratic_covar)

    if power == 4:
        quartic_kernel = QuarticKernel(c=0, ialpha=1)
        quartic_covar = quartic_kernel(state1, state2)

        assert np.allclose(linear_covar, quartic_covar)
        assert np.allclose(multiplicative_covar, quartic_covar)
        assert np.allclose(polynomial_covar, quartic_covar)


@pytest.mark.parametrize(
    "kernel_class",
    [LinearKernel,
     QuadraticKernel,
     QuarticKernel,
     GaussianKernel],
    ids=["Linear", "Quadratic", "Quartic", "Gaussian"]
)
def test_additive_kernel(kernel_class):
    kernel = kernel_class()
    kernel_list = [kernel] * 2
    additive_kernel = AdditiveKernel(kernel_list=kernel_list)
    state1 = StateVectors([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    state2 = StateVectors([[2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])
    linear_covar = kernel(state1, state2)
    assert np.allclose(linear_covar + linear_covar, additive_kernel(state1, state2))


@pytest.mark.parametrize(
    "kernel_class",
    [LinearKernel,
     QuadraticKernel,
     QuarticKernel,
     GaussianKernel],
    ids=["Linear", "Quadratic", "Quartic", "Gaussian"]
)
def test_kwargs_kernel(kernel_class):
    kwargs_list = [{}, {"parameter": 333}, {"c": 333, "ialpha": 333, "variance": 33}]
    state1 = StateVectors([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    state2 = StateVectors([[2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])
    for kwargs in kwargs_list:
        kernel = kernel_class()
        kernel2 = kernel_class(**{k: v for k, v in kwargs.items() if k in type(kernel).properties})
        covar1 = kernel(state1, state2, **kwargs)
        covar2 = kernel2(state1, state2)
        assert np.allclose(covar1, covar2)
