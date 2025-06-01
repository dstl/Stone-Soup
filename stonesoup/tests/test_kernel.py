import pytest
import numpy as np

from scipy.stats import multivariate_normal

from ..types.array import StateVectors
from ..types.detection import Detection
from ..types.state import KernelParticleState, State
from ..kernel import (Kernel, MultiplicativeKernel, AdditiveKernel,
                      PolynomialKernel, LinearKernel, QuadraticKernel, QuarticKernel,
                      GaussianKernel, TrackKernel, MeasurementKernel)
from ..types.track import Track

number_particles = 5
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
            StateVectors([[0.12615663, 0.11295446, 0.12592767, 0.11836642, 0.09369041],
                          [0.11295446, 0.12615663, 0.11517142, 0.12535121, 0.12044839],
                          [0.12592767, 0.11517142, 0.12615663, 0.12006909, 0.09661766],
                          [0.11836642, 0.12535121, 0.12006909, 0.12615663, 0.11570131],
                          [0.09369041, 0.12044839, 0.09661766, 0.11570131, 0.12615663]]),
            prior,
            None
        ),
        (
            GaussianKernel(),
            StateVectors([[0.12402707, 0.12167346, 0.12302255, 0.12608029, 0.12361090],
                          [0.12109319, 0.12359301, 0.12228308, 0.11447752, 0.12133380],
                          [0.12478203, 0.12284850, 0.12389703, 0.12593714, 0.12423455],
                          [0.12427937, 0.12564349, 0.12496328, 0.11956140, 0.12437180],
                          [0.10619750, 0.11119706, 0.10858109, 0.09589545, 0.10700182]]),
            prior,
            proposal
        ),
        (
            QuadraticKernel(),
            StateVectors([[1.12546138, 0.89290663, 1.10262155, 0.94640177, 0.75809725],
                          [0.89290663, 1.10265966, 0.91273369, 1.05014420, 1.24660020],
                          [1.10262155, 0.91273369, 1.08758008, 0.95670073, 0.79646728],
                          [0.94640177, 1.05014420, 0.95670073, 1.02467909, 1.11808890],
                          [0.75809725, 1.24660020, 0.79646728, 1.11808890, 1.62703956]]),
            prior,
            None
        ),
        (
            QuadraticKernel(),
            StateVectors([[1.03096874, 0.98911714, 1.01165201, 1.10961207, 1.02462261],
                          [0.97224740, 1.00961491, 0.98879035, 0.90523805, 0.97662291],
                          [1.02501790, 0.99032626, 1.00779316, 1.08835364, 1.01659991],
                          [0.98623137, 1.00469461, 0.99431764, 0.95254807, 0.98816824],
                          [0.93606691, 1.02383081, 0.97669194, 0.78687434, 0.95117509]]),
            prior,
            proposal
        ),
        (
            QuarticKernel(),
            StateVectors([[1.26666331, 0.79728225, 1.21577428, 0.89567631, 0.57471144],
                          [0.79728225, 1.21585832, 0.83308279, 1.10280285, 1.55401205],
                          [1.21577428, 0.83308279, 1.18283043, 0.91527629, 0.63436013],
                          [0.89567631, 1.10280285, 0.91527629, 1.04996723, 1.25012279],
                          [0.57471144, 1.55401205, 0.63436013, 1.25012279, 2.64725772]]),
            prior,
            None
        ),
        (
            QuarticKernel(),
            StateVectors([[1.06289653, 0.97835271, 1.02343979, 1.23123896, 1.04985149],
                          [0.94526501, 1.01932226, 0.97770636, 0.81945592, 0.95379232],
                          [1.05066170, 0.98074609, 1.01564705, 1.18451365, 1.03347537],
                          [0.97265232, 1.00941125, 0.98866757, 0.90734782, 0.97647647],
                          [0.87622127, 1.04822954, 0.95392714, 0.61917123, 0.90473404]]),
            prior,
            proposal
        ),
        (
            GaussianKernel(),
            StateVectors([[0.12402707, 0.12167346, 0.12302255, 0.12608029, 0.12361090],
                          [0.12109319, 0.12359301, 0.12228308, 0.11447752, 0.12133380],
                          [0.12478203, 0.12284850, 0.12389703, 0.12593714, 0.12423455],
                          [0.12427937, 0.12564349, 0.12496328, 0.11956140, 0.12437180],
                          [0.10619750, 0.11119706, 0.10858109, 0.09589545, 0.10700182]]),
            prior.state_vector,
            proposal
        ),
        (
            GaussianKernel(),
            StateVectors([[0.12402707, 0.12167346, 0.12302255, 0.12608029, 0.12361090],
                          [0.12109319, 0.12359301, 0.12228308, 0.11447752, 0.12133380],
                          [0.12478203, 0.12284850, 0.12389703, 0.12593714, 0.12423455],
                          [0.12427937, 0.12564349, 0.12496328, 0.11956140, 0.12437180],
                          [0.10619750, 0.11119706, 0.10858109, 0.09589545, 0.10700182]]),
            prior,
            proposal.state_vector
        ),
        (
            QuadraticKernel(),
            StateVectors([[1.03096874, 0.98911714, 1.01165201, 1.10961207, 1.02462261],
                          [0.97224740, 1.00961491, 0.98879035, 0.90523805, 0.97662291],
                          [1.02501790, 0.99032626, 1.00779316, 1.08835364, 1.01659991],
                          [0.98623137, 1.00469461, 0.99431764, 0.95254807, 0.98816824],
                          [0.93606691, 1.02383081, 0.97669194, 0.78687434, 0.95117509]]),
            prior.state_vector,
            proposal
        ),
        (
            QuadraticKernel(),
            StateVectors([[1.03096874, 0.98911714, 1.01165201, 1.10961207, 1.02462261],
                          [0.97224740, 1.00961491, 0.98879035, 0.90523805, 0.97662291],
                          [1.02501790, 0.99032626, 1.00779316, 1.08835364, 1.01659991],
                          [0.98623137, 1.00469461, 0.99431764, 0.95254807, 0.98816824],
                          [0.93606691, 1.02383081, 0.97669194, 0.78687434, 0.95117509]]),
            prior,
            proposal.state_vector
        ),
        (
            QuarticKernel(),
            StateVectors([[1.06289653, 0.97835271, 1.02343979, 1.23123896, 1.04985149],
                          [0.94526501, 1.01932226, 0.97770636, 0.81945592, 0.95379232],
                          [1.05066170, 0.98074609, 1.01564705, 1.18451365, 1.03347537],
                          [0.97265232, 1.00941125, 0.98866757, 0.90734782, 0.97647647],
                          [0.87622127, 1.04822954, 0.95392714, 0.61917123, 0.90473404]]),
            prior.state_vector,
            proposal
        ),
        (
            QuarticKernel(),
            StateVectors([[1.06289653, 0.97835271, 1.02343979, 1.23123896, 1.04985149],
                          [0.94526501, 1.01932226, 0.97770636, 0.81945592, 0.95379232],
                          [1.05066170, 0.98074609, 1.01564705, 1.18451365, 1.03347537],
                          [0.97265232, 1.00941125, 0.98866757, 0.90734782, 0.97647647],
                          [0.87622127, 1.04822954, 0.95392714, 0.61917123, 0.90473404]]),
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
    kernel_covar = kernel_class(state1, state2)
    sv1 = state1.state_vector if isinstance(state1, State) else state1
    print(state2, state2 is None)
    if state2 is not None:
        sv2 = state2.state_vector if isinstance(state2, State) else state2
    else:
        sv2 = sv1
    assert kernel_covar.shape == (sv1.shape[1], sv2.shape[1])
    assert np.allclose(kernel_covar, output)


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

    assert multiplicative_covar.shape == (state1.shape[1], state2.shape[1])
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
    state2 = StateVectors([[2, 2], [3, 3], [4, 4], [5, 5]])
    linear_covar = kernel(state1, state2)
    assert linear_covar.shape == (state1.shape[1], state2.shape[1])
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


@pytest.mark.parametrize(
    "kernel_class",
    [LinearKernel,
     QuadraticKernel,
     QuarticKernel,
     GaussianKernel],
    ids=["Linear", "Quadratic", "Quartic", "Gaussian"]
)
def test_multiple_kwargs(kernel_class):
    kwargs_list = [{"c": 1, "ialpha": 11, "variance": 11},
                   {"c": 2, "ialpha": 22, "variance": 22},
                   {"c": 3, "ialpha": 33, "variance": 33}]
    state1 = StateVectors([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    state2 = StateVectors([[2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])
    kernel = kernel_class()
    for kwargs in kwargs_list:
        print(kernel)
        kernel2 = kernel_class(**{k: v for k, v in kwargs.items() if k in type(kernel).properties})
        covar1 = kernel(state1, state2, **kwargs)
        covar2 = kernel2(state1, state2)
        print(kernel, kernel2)
        assert np.allclose(covar1, covar2)


@pytest.mark.parametrize(
    "kernel_class",
    [LinearKernel,
     QuadraticKernel,
     QuarticKernel,
     GaussianKernel],
    ids=["Linear", "Quadratic", "Quartic", "Gaussian"]
)
def test_track_kernel(kernel_class):
    kernel = kernel_class()
    track_kernel = TrackKernel(kernel)
    track_state1 = State(state_vector=[1, 2, 3, 4])
    track_state2 = State(state_vector=[2, 3, 4, 5])
    state_vectors1 = StateVectors([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    state_vectors2 = StateVectors([[2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])
    track1 = Track([track_state1, track_state1, track_state1])
    track2 = Track([track_state2, track_state2, track_state2])
    for tracks, svs in zip([[track1], [track1, track2]],
                           [[state_vectors1], [state_vectors1, state_vectors2]]):
        track_covar = track_kernel(*tracks)
        sv_covar = kernel(*svs)
        assert np.allclose(track_covar, sv_covar)


@pytest.fixture(scope="module", params=[
    (None, StateVectors([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])),
    ([0], StateVectors([[1, 1, 1]])),
    ([0, 1], StateVectors([[1, 1, 1], [2, 2, 2]])),
    ([0, 1, 2], StateVectors([[1, 1, 1], [2, 2, 2], [3, 3, 3]])),
    ([0, 1, 2, 3], StateVectors([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]))
])
def params(request):
    return request.param


@pytest.mark.parametrize(
    "kernel_class",
    [LinearKernel,
     QuadraticKernel,
     QuarticKernel,
     GaussianKernel],
    ids=["Linear", "Quadratic", "Quartic", "Gaussian"]
)
def test_track_mapping_kernel(kernel_class, params):
    kernel = kernel_class()
    measurement_kernel = MeasurementKernel(kernel, params[0])
    measurement = Detection(state_vector=[1, 2, 3, 4])
    measurements = [measurement, measurement, measurement]
    measurement_covar = measurement_kernel(measurements)
    sv_covar = kernel(params[1])
    assert np.allclose(measurement_covar, sv_covar)


@pytest.mark.parametrize(
    "kernel_class",
    [LinearKernel,
     QuadraticKernel,
     QuarticKernel,
     GaussianKernel],
    ids=["Linear", "Quadratic", "Quartic", "Gaussian"]
)
def test_measurement_kernel(kernel_class):
    kernel = kernel_class()
    measurement_kernel = MeasurementKernel(kernel)
    measurement = Detection(state_vector=[1, 2, 3, 4])
    state_vectors = StateVectors([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    measurements = [measurement, measurement, measurement]
    measurement_covar = measurement_kernel(measurements)
    sv_covar = kernel(state_vectors)
    assert np.allclose(measurement_covar, sv_covar)


@pytest.mark.parametrize(
        "kernel_class,parameters,new_parameters",
        [
            (LinearKernel, dict(), dict()),
            (QuadraticKernel, dict(c=1, ialpha=10), dict(c=2, ialpha=10)),
            (QuarticKernel, dict(c=1, ialpha=10), dict(c=2, ialpha=10)),
            (GaussianKernel, dict(variance=10), dict(variance=20))
        ],
        ids=["Linear", "Quadratic", "Quartic", "Gaussian"]
        )
def test_parameters(kernel_class, parameters, new_parameters):
    kernel = kernel_class()
    for k, v in kernel.parameters.items():
        assert k in parameters.keys()
        assert v == parameters[k]
    kernel.update_parameters(new_parameters)
    for k, v in new_parameters.items():
        assert kernel.parameters[k] == v


@pytest.mark.parametrize(
        "kernel_class,primary_kernel_class,parameters,new_parameters",
        [
            (TrackKernel, LinearKernel, dict(), dict()),
            (TrackKernel, QuadraticKernel, dict(c=1, ialpha=10), dict(c=2, ialpha=10)),
            (MeasurementKernel, QuarticKernel, dict(c=1, ialpha=10), dict(c=2, ialpha=10)),
            (MeasurementKernel, GaussianKernel, dict(variance=10), dict(variance=20))
        ],
        ids=["Linear", "Quadratic", "Quartic", "Gaussian"]
        )
def test_nested_parameters(kernel_class, primary_kernel_class, parameters, new_parameters):
    kernel = primary_kernel_class()
    kernel = kernel_class(kernel)
    for k, v in kernel.parameters.items():
        assert k in parameters.keys()
        assert v == parameters[k]
    kernel.update_parameters(new_parameters)
    for k, v in new_parameters.items():
        assert kernel.parameters[k] == v
