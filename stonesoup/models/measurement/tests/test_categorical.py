# -*- coding: utf-8 -*-

import numpy as np
import pytest

from ...transition.tests.test_categorical import create_random_multinomial
from ....models.measurement.categorical import CategoricalMeasurement
from ....types.array import Matrix, CovarianceMatrix, StateVectors


def test_time_invariant_observation():
    # test emission matrix error
    with pytest.raises(ValueError, match="Row 0 of emission matrix does not sum to 1"):
        CategoricalMeasurement(ndim_state=2,
                               emission_matrix=Matrix([[1, 1],
                                                       [0, 0]]))

    # test mapping error
    with pytest.raises(ValueError, match="Emission matrix maps from 2 elements of the state "
                                         "space, but the mapping is length 3"):
        CategoricalMeasurement(ndim_state=6, emission_matrix=np.eye(2), mapping=[0, 2, 4])

    # 3 possible measurement categories, 2 possible hidden categories
    E = Matrix([[0.5, 0.5, 0.0],
                [0.0, 0.5, 0.5]])
    Ecov = CovarianceMatrix(np.diag([0.1, 0.1, 0.1]))

    mapping = [0, 2]

    model = CategoricalMeasurement(ndim_state=4,
                                   emission_matrix=E,
                                   emission_covariance=Ecov,
                                   mapping=mapping)

    # test ndim meas
    assert model.ndim_meas == 3

    # test conditional probability emission
    for _ in range(3):
        state = create_random_multinomial(4)
        # testing noiseless as noise generation uses random sampling
        exp_value = E.T @ state.state_vector[mapping]
        exp_value = exp_value / np.sum(exp_value)
        actual_value = model._cond_prob_emission(state)
        assert np.array_equal(exp_value, actual_value)
        assert sum(actual_value) == 1

    # test function
    for _ in range(3):
        state = create_random_multinomial(4)
        measurement = model.function(state)
        assert len(np.where(measurement == 0)[0]) == 2
        assert len(np.where(measurement == 1)[1]) == 1
        assert len(measurement) == 3

    # test rvs
    for i in range(1, 4):
        rvs = model.rvs(num_samples=i)
        assert isinstance(rvs, StateVectors)
        assert len(rvs.T) == i
        for elem in rvs.T:
            assert len(elem) == 3  # same as model ndim meas

    # test pdf
    for _ in range(3):
        state1 = create_random_multinomial(3)  # measurement vector
        state2 = create_random_multinomial(4)  # state vector
        exp_Hx = E.T @ state2.state_vector[mapping]
        exp_Hx = exp_Hx / np.sum(exp_Hx)
        exp_value = exp_Hx.T @ state1.state_vector
        actual_value = model.pdf(state1, state2)
        assert np.array_equal(actual_value, exp_value)
