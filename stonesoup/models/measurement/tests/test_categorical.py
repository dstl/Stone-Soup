# -*- coding: utf-8 -*-

import numpy as np
import pytest

from ...transition.tests.test_categorical import create_random_categorical
from ....models.measurement.categorical import CategoricalMeasurementModel
from ....types.array import Matrix, CovarianceMatrix, StateVectors


def test_time_invariant_observation():
    # test emission matrix error
    with pytest.raises(ValueError, match="Row 0 of emission matrix does not sum to 1"):
        CategoricalMeasurementModel(ndim_state=2,
                                    emission_matrix=Matrix([[1, 1],
                                                       [0, 0]]))

    # test mapping error
    with pytest.raises(ValueError, match="Emission matrix maps from 2 elements of the state "
                                         "space, but the mapping is length 3"):
        CategoricalMeasurementModel(ndim_state=6, emission_matrix=np.eye(2), mapping=[0, 2, 4])

    # test category name error
    with pytest.raises(ValueError, match="2 category names were given for a model which returns "
                                         "vectors of length 3"):
        CategoricalMeasurementModel(ndim_state=6, emission_matrix=np.eye(3), mapping=[0, 2, 4],
                                    category_names=['red', 'blue'])
    with pytest.raises(ValueError, match="4 category names were given for a model which returns "
                                         "vectors of length 3"):
        CategoricalMeasurementModel(ndim_state=6, emission_matrix=np.eye(3), mapping=[0, 2, 4],
                                    category_names=['red', 'blue', 'yellow', 'green'])

    # 3 possible measurement categories, 2 possible hidden categories
    E = Matrix([[0.5, 0.5, 0.0],
                [0.0, 0.5, 0.5]])
    Ecov = CovarianceMatrix(np.diag([0.1, 0.1, 0.1]))

    mapping = [0, 2]

    model = CategoricalMeasurementModel(ndim_state=4,
                                        emission_matrix=E,
                                        emission_covariance=Ecov,
                                        mapping=mapping)

    # test ndim meas
    assert model.ndim_meas == 3

    # test default category names
    assert model.category_names == [0, 1, 2]

    # test conditional probability emission
    for _ in range(3):
        state = create_random_categorical(4)
        # testing noiseless as noise generation uses random sampling
        exp_hp = E.T @ state.state_vector[mapping]
        exp_y = np.log(exp_hp / (1 - exp_hp))
        exp_p = 1 / (1 + np.exp(-exp_y))
        exp_p = exp_p / np.sum(exp_p)
        actual_p = model._cond_prob_emission(state)
        assert np.array_equal(exp_p, actual_p)
        assert np.isclose(np.sum(actual_p), 1)

    with pytest.raises(ValueError, match="Noise is generated via random sampling, and defined "
                                         "noise is not implemented"):
        model._cond_prob_emission(create_random_categorical(4), noise=[1, 2, 3])

    # test function with noise (random sampling at end)
    for _ in range(3):
        state = create_random_categorical(4)
        measurement = model.function(state, noise=True)
        assert len(np.where(measurement == 0)[0]) == 2
        assert len(np.where(measurement == 1)[1]) == 1
        assert len(measurement) == 3

    # test function without noise (no random sampling at end)
    for _ in range(3):
        state = create_random_categorical(4)
        measurement = model.function(state, noise=False)
        assert len(measurement) == 3
        assert np.isclose(np.sum(measurement), 1)

    # test rvs
    for i in range(1, 4):
        rvs = model.rvs(num_samples=i)
        assert isinstance(rvs, StateVectors)
        assert len(rvs.T) == i
        for elem in rvs.T:
            assert len(elem) == 3  # same as model ndim meas

    # test pdf
    for _ in range(3):
        state1 = create_random_categorical(3)  # measurement vector
        state2 = create_random_categorical(4)  # state vector
        exp_Hx = E.T @ state2.state_vector[mapping]
        exp_Hx = exp_Hx / np.sum(exp_Hx)
        exp_value = exp_Hx.T @ state1.state_vector
        actual_value = model.pdf(state1, state2)
        assert np.array_equal(actual_value, exp_value)
