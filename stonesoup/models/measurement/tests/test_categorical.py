# -*- coding: utf-8 -*-

import numpy as np
import pytest

from ....models.measurement.categorical import MarkovianMeasurementModel
from ....types.array import StateVector
from ....types.state import CategoricalState


def test_categorical_measurement_model():
    # 3 hidden categories, 4 measurement categories
    E = np.array([[30, 25, 5],
                  [20, 25, 10],
                  [10, 25, 80],
                  [40, 25, 5]])

    # Test mismatched number of category names
    with pytest.raises(ValueError, match="ndim_meas of 4 does not match number of measurement "
                                         "categories 2"):
        MarkovianMeasurementModel(E, measurement_categories=['red', 'blue'])

    model = MarkovianMeasurementModel(E)

    # Test default category names
    assert model.measurement_categories == ['0', '1', '2', '3']

    # Test normalised
    expected_array = np.array([[3 / 10, 1 / 4, 1 / 20],
                               [2 / 10, 1 / 4, 2 / 20],
                               [1 / 10, 1 / 4, 16 / 20],
                               [4 / 10, 1 / 4, 1 / 20]])
    assert np.allclose(model.emission_matrix, expected_array)

    # Test ndim
    assert model.ndim_state == 3
    assert model.ndim_meas == 4

    state = CategoricalState([80, 10, 10])

    # Test function
    new_vector = model.function(state, noise=False)
    assert isinstance(new_vector, StateVector)
    assert new_vector.shape[0] == 4
    assert np.isclose(np.sum(new_vector), 1)

    # Test mapping
    assert np.array_equal(model.mapping, [0, 1, 2])
