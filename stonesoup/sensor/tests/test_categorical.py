# -*- coding: utf-8 -*-
import datetime

import pytest
import numpy as np

from stonesoup.models.measurement.categorical import CategoricalMeasurementModel
from stonesoup.types.detection import CategoricalDetection, TrueCategoricalDetection
from stonesoup.types.groundtruth import GroundTruthState, CategoricalGroundTruthState, \
    GroundTruthPath
from ..categorical import CategoricalSensor
from ...types.array import CovarianceMatrix


def test_categorical_sensor():
    # generate random emission matrix and normalise
    # 2x4 - 2 state space elements, to 4 measurement space
    E = np.random.rand(2, 4)
    sum_of_rows = E.sum(axis=1)
    E = E / sum_of_rows[:, np.newaxis]
    Ecov = CovarianceMatrix(0.1 * np.eye(4))

    sensor = CategoricalSensor(ndim_state=3, mapping=[0, 2],
                               emission_matrix=E, emission_covariance=Ecov)

    exp_measurement_model = CategoricalMeasurementModel(ndim_state=3, mapping=[0, 2],
                                                        emission_matrix=E,
                                                        emission_covariance=Ecov)

    assert isinstance(sensor.measurement_model, CategoricalMeasurementModel)
    for property, exp_property in zip(sensor.measurement_model._properties,
                                      exp_measurement_model._properties):
        assert property == exp_property

    now = datetime.datetime.now()

    # test measuring non-categorical error
    with pytest.raises(ValueError, match="Categorical sensor can only observe categorical states"):
        sensor.measure({GroundTruthState([1, 2, 3], timestamp=now)})
        sensor.measure({GroundTruthPath(GroundTruthState([1, 2, 3], timestamp=now))})

    truth1 = CategoricalGroundTruthState([0.1, 0.4, 0.5], timestamp=now)
    truth2 = CategoricalGroundTruthState([0.6, 0.1, 0.3], timestamp=now)
    truth3 = CategoricalGroundTruthState([0.1, 0.1, 0.8], timestamp=now)
    ground_truths = {truth1, truth2, truth3}

    detections = sensor.measure(ground_truths)

    assert len(detections) == len(ground_truths)

    for detection in detections:
        assert isinstance(detection, TrueCategoricalDetection)
        assert detection.groundtruth_path in ground_truths
