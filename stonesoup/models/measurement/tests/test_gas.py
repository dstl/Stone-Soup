import numpy as np
import numpy.random
import pytest
from scipy.special import erf

from ..gas import IsotropicPlume

from ....types.array import StateVector, StateVectors
from ....types.state import State, ParticleState


def isoplume_h(state_vector, translation_offset):
    x, y, z, q, u, phi, zeta1, zeta2 = state_vector
    dist = np.sqrt((x - translation_offset[0])**2 +
                   (y - translation_offset[1])**2 +
                   (z - translation_offset[2])**2)
    lambda_ = np.sqrt((zeta1*zeta2)/(1 + (u**2*zeta2)/(4*zeta1)))

    conc = q/(4*np.pi*zeta1*dist)*np.exp((-(translation_offset[0]-x)*u*np.cos(phi))/(2*zeta1) +
                                         (-(translation_offset[1]-y)*u*np.sin(phi))/(2*zeta1) +
                                         (-1*dist/lambda_))

    return conc


@pytest.mark.parametrize(
    "state, mapping, translation_offset",
    [
        (
                State(StateVector([30, 40, 1, 5, 4, np.radians(90), 1, 8])),  # state
                range(0, 8),  # mapping
                None,  # translation_offset
        ), (
                State(StateVector([30, 40, 1, 5, 4, np.radians(90), 1, 8])),  # state
                range(0, 8),  # mapping
                np.array([[10], [10], [1]]),  # translation_offset
        ), (
                State(StateVector([30, 40, 1, 5, 4, np.radians(90), 1, 8])),  # state
                None,  # mapping
                np.array([[20], [20], [2]]),  # translation_offset
        ), (
                State(StateVector([30, 40, 1, 5, 4, np.radians(90), 1, 8])),  # state
                range(0, 8),  # mapping
                np.array([[30], [35], [1]]),  # translation_offset
        ), (
                State(StateVector([5, 1, 30, 4, 8, 1, 40, np.radians(90)])),  # state
                [2, 6, 5, 0, 3, 7, 1, 4],  # mapping
                np.array([[30], [35], [1]]),  # translation_offset
        ), (
                ParticleState(
                    StateVectors([np.random.uniform(0, 50, 1000),
                                 np.random.uniform(0, 50, 1000),
                                 np.random.uniform(0, 5, 1000),
                                 np.random.gamma(2, 5, 1000),
                                 np.random.normal(4, 2, 1000),
                                 np.random.normal(np.radians(90), np.radians(10), 1000),
                                 np.random.uniform(1, 3, 1000),
                                 26 + np.random.uniform(0, 2, 1000)]),
                    weight=np.array([1 / 1000] * 1000),
                    timestamp=None),  # state
                range(0, 8),  # mapping
                np.array([[30], [35], [1]]),  # translation_offset
        ), (
                ParticleState(
                    StateVectors([np.random.gamma(2, 5, 1000),
                                  np.random.uniform(1, 3, 1000),
                                  np.random.uniform(0, 50, 1000),
                                  np.random.normal(4, 2, 1000),
                                  26 + np.random.uniform(0, 2, 1000),
                                  np.random.uniform(0, 5, 1000),
                                  np.random.uniform(0, 50, 1000),
                                  np.random.normal(np.radians(90), np.radians(10), 1000)]),
                    weight=np.array([1 / 1000] * 1000),
                    timestamp=None),  # state
                [2, 6, 5, 0, 3, 7, 1, 4],  # mapping
                np.array([[30], [35], [1]]),  # translation_offset
        )
    ],
    ids=["no_trans_offset", "trans_offset1", "trans_offset2", "trans_offset3",
         "with_mapping", "particle_state", "particle_state_with_mapping"]
)
def test_isotropic_plume(state, mapping, translation_offset):

    missed_detection_probability = 0.1
    sensing_threshold = 1e-4
    standard_deviation_percentage = 0.5
    noise = 1e-4

    # test that function is correct
    # no noise
    if mapping:
        model = IsotropicPlume(min_noise=noise,
                               sensing_threshold=sensing_threshold,
                               standard_deviation_percentage=standard_deviation_percentage,
                               missed_detection_probability=0,
                               translation_offset=translation_offset,
                               mapping=mapping)
    else:
        model = IsotropicPlume(min_noise=noise,
                               sensing_threshold=sensing_threshold,
                               standard_deviation_percentage=standard_deviation_percentage,
                               missed_detection_probability=0,
                               translation_offset=translation_offset)

    if translation_offset is None:
        translation_offset = np.array([[0], [0], [0]])
    if mapping is None:
        mapping = range(0, 8)

    nparts = state.state_vector.shape[1]

    # Test no noise
    expected_conc = model.function(state)
    assert np.shape(expected_conc)[0] == model.ndim
    unmapped_state = state.state_vector[mapping, :].view(np.ndarray)
    actual_conc = isoplume_h(unmapped_state, translation_offset)
    assert np.all(np.isclose(expected_conc, actual_conc))

    # Test noise
    expected_conc = model.function(state, noise=True, random_state=1990)

    rng = np.random.RandomState(1990)
    actual_conc += actual_conc * standard_deviation_percentage * rng.normal(size=nparts)
    actual_conc[actual_conc < sensing_threshold] = 0

    assert np.all(np.isclose(expected_conc, actual_conc))

    # Check that logpdf and pdf are working correctly
    conc = State(state_vector=StateVector(expected_conc[0]))
    model.missed_detection_probability = missed_detection_probability
    expected_log_likelihood = model.logpdf(conc, state)
    expected_likelihood = model.pdf(conc, state)

    assert np.all(np.isclose(expected_likelihood.astype(np.float_),
                             np.exp(expected_log_likelihood)))

    pred_conc = isoplume_h(unmapped_state, translation_offset=translation_offset)
    if conc.state_vector[0] <= sensing_threshold:
        actual_likelihood = (1 - missed_detection_probability) * 1 / 2 * (1 + erf(
            (sensing_threshold - pred_conc) / (
                    sensing_threshold * np.sqrt(2)))) + missed_detection_probability
    else:
        sigma = standard_deviation_percentage * pred_conc + noise
        actual_likelihood = 1 / (sigma * np.sqrt(2 * np.pi)) \
            * np.exp(-(conc.state_vector - pred_conc) ** 2 / (2 * sigma ** 2))

    assert np.all(np.isclose(expected_log_likelihood, np.log(actual_likelihood)))

    # Check expected response from model.covar
    with pytest.raises(NotImplementedError) as e:
        model.covar()
    assert 'Covariance for IsotropicPlume is dependant on the '\
        'measurement as well as standard deviation!' in str(e.value)
