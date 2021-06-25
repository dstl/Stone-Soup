# -*- coding: utf-8 -*-
import datetime

import numpy as np
import pytest

from ..categorical import CategoricalSensor
from ...models.measurement.categorical import CategoricalMeasurementModel
from ...models.transition.tests.test_categorical import create_categorical_matrix
from ...types.array import CovarianceMatrix
from ...types.detection import TrueCategoricalDetection
from ...types.groundtruth import GroundTruthState, CategoricalGroundTruthState, \
    GroundTruthPath


@pytest.mark.parametrize('category_names', (['red', 'green', 'blue', 'yellow'], None))
def test_categorical_sensor(category_names):
    # 4 measurement categories, 2 hidden categories
    E = create_categorical_matrix(2, 4)
    Ecov = CovarianceMatrix(np.eye(4))
    mapping = [0, 2]

    model = CategoricalMeasurementModel(emission_matrix=E,
                                        emission_covariance=Ecov,
                                        mapping=mapping)

    # Test category names error
    with pytest.raises(ValueError, match="3 category names were given for a sensor which returns "
                                         "vectors of length 4"):
        CategoricalSensor(measurement_model=model, category_names=['red', 'green', 'blue'])

    sensor = CategoricalSensor(measurement_model=model,
                               category_names=category_names)

    now = datetime.datetime.now()

    # Test ndim state
    assert sensor.ndim_state == 2

    # Test ndim meas
    assert sensor.ndim_meas == 4

    # Test measuring non-categorical error
    with pytest.raises(ValueError, match="Categorical sensor can only observe categorical states"):
        sensor.measure({GroundTruthState([1, 2, 3], timestamp=now)})
    with pytest.raises(ValueError, match="Categorical sensor can only observe categorical states"):
        sensor.measure({GroundTruthPath(GroundTruthState([1, 2, 3], timestamp=now))})

    # Test measure
    truth1 = GroundTruthPath([CategoricalGroundTruthState([0.1, 0.4, 0.5], timestamp=now)])
    truth2 = GroundTruthPath([CategoricalGroundTruthState([0.6, 0.1, 0.3], timestamp=now)])
    truth3 = CategoricalGroundTruthState([0.1, 0.1, 0.8], timestamp=now)
    ground_truths = {truth1, truth2, truth3}

    detections = sensor.measure(ground_truths)

    assert len(detections) == len(ground_truths)

    for detection in detections:
        assert isinstance(detection, TrueCategoricalDetection)
        assert len(detection.state_vector) == 4
        assert np.isclose(np.sum(detection.state_vector), 1)  # is normalised
        assert detection.groundtruth_path in ground_truths
        assert detection.timestamp == now
        assert detection.measurement_model == model
        if category_names:
            assert detection.category_names == category_names
        else:
            assert detection.category_names == [0, 1, 2, 3]  # default category names
