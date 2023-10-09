from datetime import datetime

import numpy as np

from ..categorical import HMMSensor
from ...models.measurement.categorical import MarkovianMeasurementModel
from ...types.detection import TrueCategoricalDetection
from ...types.groundtruth import CategoricalGroundTruthState, GroundTruthPath


def test_hmm_sensor():
    # 3 hidden categories, 4 measurement categories
    E = np.array([[30, 25, 5],
                  [20, 25, 10],
                  [10, 25, 80],
                  [40, 25, 5]])

    model = MarkovianMeasurementModel(E)

    sensor = HMMSensor(measurement_model=model)

    # Test ndim state
    assert sensor.ndim_state == 3

    # Test ndim meas
    assert sensor.ndim_meas == 4

    # Test measure
    now = datetime.now()
    truth1 = GroundTruthPath([CategoricalGroundTruthState([0.1, 0.4, 0.5], timestamp=now)])
    truth2 = GroundTruthPath([CategoricalGroundTruthState([0.6, 0.1, 0.3], timestamp=now)])
    truth3 = CategoricalGroundTruthState([0.1, 0.1, 0.8], timestamp=now)
    ground_truths = {truth1, truth2, truth3}

    for noise in (True, False):

        detections = sensor.measure(ground_truths, noise=noise)

        assert len(detections) == 3

        for detection in detections:
            assert isinstance(detection, TrueCategoricalDetection)
            assert len(detection.state_vector) == 4
            assert np.isclose(np.sum(detection.state_vector), 1)  # is normalised
            assert detection.groundtruth_path in ground_truths
            assert detection.timestamp == now
            assert detection.measurement_model == model
            assert detection.categories == ['0', '1', '2', '3']

            if noise:
                assert np.count_nonzero(detection.state_vector) == 1  # basis vector
