# -*- coding: utf-8 -*-
from datetime import datetime

import numpy as np
import pytest

from ..categorical import SimpleCategoricalInitiator
from ...models.measurement.categorical import CategoricalMeasurementModel
from ...models.transition.tests.test_categorical import create_categorical, \
    create_categorical_matrix
from ...types.detection import CategoricalDetection
from ...types.state import CategoricalState
from ...types.update import CategoricalStateUpdate


@pytest.mark.parametrize(
    'measurement_model',
    [CategoricalMeasurementModel(ndim_state=3,
                                 emission_matrix=create_categorical_matrix(3, 3),
                                 emission_covariance=0.1 * np.eye(3),
                                 mapping=[0, 1, 2]),
     CategoricalMeasurementModel(ndim_state=3,
                                 emission_matrix=create_categorical_matrix(2, 2),
                                 emission_covariance=0.1 * np.eye(3),
                                 mapping=[0, 1]),
     CategoricalMeasurementModel(ndim_state=3,
                                 emission_matrix=create_categorical_matrix(2, 2),
                                 emission_covariance=0.1 * np.eye(3),
                                 mapping=[0, 2]),
     CategoricalMeasurementModel(ndim_state=3,
                                 emission_matrix=create_categorical_matrix(2, 2),
                                 emission_covariance=0.1 * np.eye(3),
                                 mapping=[2, 0])
     ],
    ids=['[0, 1, 2]', '[0, 1]', '[0, 2]', '[2, 0]'])
def test_categorical_initiator(measurement_model):
    now = datetime.now()

    # Prior state information
    prior_state = CategoricalState([1 / 3, 1 / 3, 1 / 3], category_names=['red', 'green', 'blue'])

    ndim_meas = measurement_model.ndim_meas

    measurements = [CategoricalDetection(create_categorical(ndim_meas), timestamp=now,
                                         measurement_model=measurement_model),
                    CategoricalDetection(create_categorical(ndim_meas), timestamp=now)]

    initiator = SimpleCategoricalInitiator(prior_state, measurement_model=measurement_model)

    tracks = initiator.initiate(measurements)

    assert len(tracks) == 2
    for track in tracks:
        assert len(track) == 1
        assert isinstance(track.state, CategoricalStateUpdate)

    assert set(measurements) == set(track.state.hypothesis.measurement for track in tracks)
