import datetime
import pickle

import numpy as np
import pytest
from scipy.spatial import distance
from scipy.stats import multivariate_normal

from .. import measures
from ..measures import ObservationAccuracy
from ..types.array import StateVector, CovarianceMatrix, StateVectors
from ..types.state import GaussianState, State, ParticleState

# Create a time stamp to use for both states
t = datetime.datetime.now()

# Set target ground truth prior
u = StateVector([[10.], [1.], [10.], [1.]])
ui = CovarianceMatrix(np.diag([100., 10., 100., 10.]))

state_u = GaussianState(u, ui, timestamp=t)
stateB_u = State(u, timestamp=t)

v = StateVector([[11.], [10.], [100.], [2.]])
vi = CovarianceMatrix(np.diag([20., 3., 7., 10.]))

state_v = GaussianState(v, vi, timestamp=t)
stateB_v = State(v, timestamp=t)


def test_measure_raise_error():
    with pytest.raises(ValueError) as excinfo:
        measures.Euclidean(mapping2=[0, 3])
    assert "Cannot set mapping2 if mapping is None." in str(excinfo.value)


def test_euclidean():

    measure = measures.Euclidean()
    assert measure(state_u, state_v) == distance.euclidean(u[:, 0], v[:, 0])


def test_euclideanweighted():
    weight = np.array([1, 2, 3, 1])
    measure = measures.EuclideanWeighted(weight)
    assert measure(state_u, state_v) == distance.euclidean(u[:, 0], v[:, 0], weight)
    assert measure(stateB_u, stateB_v) == distance.euclidean(u[:, 0], v[:, 0], weight)


def test_mahalanobis():
    measure = measures.Mahalanobis()
    assert measure(state_u, state_v) == distance.mahalanobis(u[:, 0],
                                                             v[:, 0],
                                                             np.linalg.inv(ui))


def test_hellinger():
    v = StateVector([[11.], [10.], [10.], [2.]])
    state_v = GaussianState(v, vi, timestamp=t)
    measure = measures.GaussianHellinger()
    assert np.isclose(measure(state_u, state_v), 0.940, atol=1e-3)


def test_observation_accuracy():
    measure = ObservationAccuracy()
    for _ in range(5):
        TP = np.random.random()
        TN = 1 - TP
        FP = np.random.random()
        FN = 1 - FP

        u = StateVector([TP, TN])
        v = StateVector([FP, FN])
        U = State(u)
        V = State(v)

        assert measure(u, v) == (min([TP, FP]) + min(TN, FN)) / (max([TP, FP]) + max(TN, FN))
        assert measure(U, V) == (min([TP, FP]) + min(TN, FN)) / (max([TP, FP]) + max(TN, FN))


@pytest.mark.xfail(reason="Singular Matrix with all zero covariances.")
def test_zero_hellinger():
    measure = measures.GaussianHellinger()
    # Set target ground truth prior
    u = StateVector([[10.], [1.], [10.], [1.]])
    ui = CovarianceMatrix(np.diag([0., 0., 0., 0.]))
    state_u = GaussianState(u, ui, timestamp=t)

    v = StateVector([[11.], [10.], [100.], [2.]])
    vi = CovarianceMatrix(np.diag([0., 0., 0., 0.]))
    state_v = GaussianState(v, vi, timestamp=t)
    assert np.isclose(measure(state_u, state_v), 1, atol=1e-3)


def test_squared_hellinger():
    measure = measures.SquaredGaussianHellinger()
    v = StateVector([[11.], [10.], [10.], [2.]])
    state_v = GaussianState(v, vi, timestamp=t)
    assert np.isclose(measure(state_u, state_v), 0.884, atol=1e-3)


@pytest.fixture(params=[np.array, list, tuple], ids=['array', 'list', 'tuple'])
def mapping_type(request):
    return request.param


def test_hellinger_full_mapping(mapping_type):
    mapping = mapping_type(np.arange(len(u)))
    v = StateVector([[11.], [10.], [10.], [2.]])
    state_v = GaussianState(v, vi, timestamp=t)
    measure = measures.GaussianHellinger(mapping=mapping)
    assert np.isclose(measure(state_u, state_v), 0.940, atol=1e-3)
    measure = measures.GaussianHellinger(mapping=mapping, mapping2=mapping)
    assert np.isclose(measure(state_u, state_v), 0.940, atol=1e-3)


def test_hellinger_partial_mapping(mapping_type):
    mapping = mapping_type([0, 1])
    v = StateVector([[11.], [10.], [10.], [2.]])
    state_v = GaussianState(v, vi, timestamp=t)
    measure = measures.GaussianHellinger(mapping=mapping)
    assert np.isclose(measure(state_u, state_v), 0.913, atol=1e-3)
    mapping = np.array([0, 3])
    measure = measures.GaussianHellinger(mapping=mapping)
    assert np.isclose(measure(state_u, state_v), 0.386, atol=1e-3)

    mapping = mapping_type([0, 1])
    measure = measures.GaussianHellinger(mapping=mapping, mapping2=mapping)
    assert np.isclose(measure(state_u, state_v), 0.913, atol=1e-3)
    mapping = np.array([0, 3])
    measure = measures.GaussianHellinger(mapping=mapping, mapping2=mapping)
    assert np.isclose(measure(state_u, state_v), 0.386, atol=1e-3)

    v = StateVector([[11.], [2.], [10.], [10.]])
    state_v = GaussianState(v, vi, timestamp=t)
    mapping = mapping_type([0, 1])
    mapping2 = np.array([0, 3])
    measure = measures.GaussianHellinger(mapping=mapping, mapping2=mapping2)
    assert np.isclose(measure(state_u, state_v), 0.913, atol=1e-3)


def test_mahalanobis_full_mapping(mapping_type):
    mapping = mapping_type(np.arange(len(u)))
    measure = measures.Mahalanobis(mapping=mapping)
    assert measure(state_u, state_v) == distance.mahalanobis(u[:, 0],
                                                             v[:, 0],
                                                             np.linalg.inv(ui))
    measure = measures.Mahalanobis(mapping=mapping, mapping2=mapping)
    assert measure(state_u, state_v) == distance.mahalanobis(u[:, 0],
                                                             v[:, 0],
                                                             np.linalg.inv(ui))


def test_mahalanobis_partial_mapping(mapping_type):
    mapping = mapping_type([0, 1])
    measure = measures.Mahalanobis(mapping=mapping)
    reduced_ui = CovarianceMatrix(np.diag([100, 10]))
    assert measure(state_u, state_v) == \
        distance.mahalanobis([10, 1],
                             [11, 10], np.linalg.inv(reduced_ui))
    mapping = np.array([0, 3])
    reduced_ui = CovarianceMatrix(np.diag([100, 10]))
    measure = measures.Mahalanobis(mapping=mapping)
    assert measure(state_u, state_v) == \
        distance.mahalanobis([10, 1],
                             [11, 2], np.linalg.inv(reduced_ui))

    mapping = mapping_type([0, 1])
    measure = measures.Mahalanobis(mapping=mapping, mapping2=mapping)
    assert measure(state_u, state_v) == \
        distance.mahalanobis([10, 1],
                             [11, 10], np.linalg.inv(reduced_ui))
    mapping = np.array([0, 3])
    measure = measures.Mahalanobis(mapping=mapping, mapping2=mapping)
    assert measure(state_u, state_v) == \
        distance.mahalanobis([10, 1],
                             [11, 2], np.linalg.inv(reduced_ui))

    mapping = mapping_type([0, 1])
    mapping2 = np.array([0, 3])
    measure = measures.Mahalanobis(mapping=mapping, mapping2=mapping2)
    assert measure(state_u, state_v) == \
        distance.mahalanobis([10, 1],
                             [11, 2], np.linalg.inv(reduced_ui))


def test_euclidean_full_mapping(mapping_type):
    mapping = mapping_type(np.arange(len(u)))
    measure = measures.Euclidean(mapping=mapping)
    assert measure(state_u, state_v) == distance.euclidean(u[:, 0], v[:, 0])
    measure = measures.Euclidean(mapping=mapping, mapping2=mapping)
    assert measure(state_u, state_v) == distance.euclidean(u[:, 0], v[:, 0])


def test_euclidean_partial_mapping(mapping_type):
    mapping = mapping_type([0, 1])
    measure = measures.Euclidean(mapping=mapping)
    assert measure(state_u, state_v) == \
        distance.euclidean([10, 1], [11, 10])
    mapping = np.array([0, 3])
    measure = measures.Euclidean(mapping=mapping)
    assert measure(state_u, state_v) == \
        distance.euclidean([10, 1], [11, 2])

    mapping = mapping_type([0, 1])
    measure = measures.Euclidean(mapping=mapping, mapping2=mapping)
    assert measure(state_u, state_v) == \
        distance.euclidean([10, 1], [11, 10])
    mapping = np.array([0, 3])
    measure = measures.Euclidean(mapping=mapping, mapping2=mapping)
    assert measure(state_u, state_v) == \
        distance.euclidean([10, 1], [11, 2])

    mapping = mapping_type([0, 1])
    mapping2 = np.array([0, 3])
    measure = measures.Euclidean(mapping=mapping, mapping2=mapping2)
    assert measure(state_u, state_v) == \
        distance.euclidean([10, 1], [11, 2])


def test_euclideanweighted_full_mapping(mapping_type):
    mapping = mapping_type(np.arange(len(u)))
    weight = np.array([1, 2, 3, 1])
    measure = measures.EuclideanWeighted(weight, mapping=mapping)
    assert measure(state_u, state_v) == distance.euclidean(u[:, 0], v[:, 0], weight)
    measure = measures.EuclideanWeighted(weight, mapping=mapping, mapping2=mapping)
    assert measure(state_u, state_v) == distance.euclidean(u[:, 0], v[:, 0], weight)


def test_euclideanweighted_partial_mapping(mapping_type):
    mapping = mapping_type([0, 1])
    weight = np.array([1, 2])
    measure = measures.EuclideanWeighted(weight, mapping=mapping)
    assert measure(state_u, state_v) == \
        distance.euclidean([10, 1], [11, 10], weight)
    mapping = np.array([0, 3])
    measure = measures.EuclideanWeighted(weight, mapping=mapping)
    assert measure(state_u, state_v) == \
        distance.euclidean([10, 1], [11, 2], weight)

    mapping = mapping_type([0, 1])
    measure = measures.EuclideanWeighted(weight, mapping=mapping, mapping2=mapping)
    assert measure(state_u, state_v) == \
        distance.euclidean([10, 1], [11, 10], weight)
    mapping = np.array([0, 3])
    measure = measures.EuclideanWeighted(weight, mapping=mapping, mapping2=mapping)
    assert measure(state_u, state_v) == \
        distance.euclidean([10, 1], [11, 2], weight)

    mapping = np.array([0, 1])
    mapping2 = np.array([0, 3])
    measure = measures.EuclideanWeighted(weight, mapping=mapping, mapping2=mapping2)
    assert measure(state_u, state_v) == \
        distance.euclidean([10, 1], [11, 2], weight)


@pytest.mark.parametrize(
    'measure,result',
    [
        (measures.Mahalanobis(), distance.mahalanobis(u[:, 0], v[:, 0], np.linalg.inv(ui))),
        (measures.Mahalanobis(state_covar_inv_cache_size=0),
         distance.mahalanobis(u[:, 0], v[:, 0], np.linalg.inv(ui))),
        (measures.SquaredMahalanobis(),
         distance.mahalanobis(u[:, 0], v[:, 0], np.linalg.inv(ui))**2),
    ],
    ids=['Mahalanobis', 'Mahalanobis-no-cache', 'SquaredMahalanobis'],
)
def test_mahalanobis_pickle(measure, result):
    assert measure(state_u, state_v) == pytest.approx(result)
    if measure.state_covar_inv_cache_size > 0:
        assert measure._inv_cov.cache_info().currsize == 1

    measure = pickle.loads(pickle.dumps(measure))
    assert measure(state_u, state_v) == pytest.approx(result)
    if measure.state_covar_inv_cache_size > 0:
        assert measure._inv_cov.cache_info().hits == 0  # Cache not pickled currently
        assert measure._inv_cov.cache_info().currsize == 1


def test_kld():

    measure = measures.KLDivergence()

    part_state_a = ParticleState(
        state_vector=StateVectors(np.random.uniform(np.array([[0], [0], [0], [0]]),
                                                    np.array([[20], [2], [20], [2]]),
                                                    size=(4, 100))))
    part_state_b = ParticleState(
        state_vector=StateVectors(np.random.multivariate_normal(np.ravel(u),
                                                                ui,
                                                                100).T))

    part_state_a.log_weight = multivariate_normal.logpdf((part_state_a.state_vector - u).T, cov=ui)
    part_state_b.log_weight = multivariate_normal.logpdf((part_state_b.state_vector - u).T, cov=ui)

    kld = measure(part_state_a, part_state_b)

    # Verify that KLD is not reversible
    assert kld != measure(part_state_b, part_state_a)

    eval_kld = np.sum(np.exp(part_state_a.log_weight)
                      * (part_state_a.log_weight - part_state_b.log_weight))

    assert np.isclose(kld, eval_kld)

    # Verify that if both distributions are the same then KLD is 0
    assert measure(part_state_a, part_state_a) == 0.
    assert measure(part_state_b, part_state_b) == 0.

    # Check errors
    part_state_c = ParticleState(
        state_vector=StateVectors(np.random.uniform(np.array([[0], [0], [0], [0]]),
                                                    np.array([[20], [2], [20], [2]]),
                                                    size=(4, 101))))
    with pytest.raises(ValueError) as e:
        measure(part_state_a, part_state_c)
    assert f'The input sizes are not compatible ' \
           f'({len(part_state_a)} != {len(part_state_c)})' in str(e.value)

    with pytest.raises(TypeError) as e:
        measure(state_u, state_v)
    assert 'state1 or state2 is not a ParticleState' in str(e.value)
