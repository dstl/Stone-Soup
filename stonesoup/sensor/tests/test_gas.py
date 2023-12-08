import numpy as np
from datetime import datetime
import pytest

from ..gas import GasIntensitySensor
from ...types.groundtruth import GroundTruthState
from ...types.detection import TrueDetection


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
    "position, missed_detection_probability, sensing_threshold, min_noise, "
    "standard_deviation_percentage",
    [
        (
                np.array([[0], [0], [0]]),  # state
                None,  # missed_detection_probability
                None,  # sensing_threshold
                None,  # min_noise
                None,  # standard_deviation_percentage
        ), (
                np.array([[15], [20], [1]]),  # state
                0.3,  # missed_detection_probability
                1e-4,  # sensing_threshold
                5e-4,  # min_noise
                0.4,  # standard_deviation_percentage
        ), (
                np.array([[30], [35], [1]]),  # state
                0.1,  # missed_detection_probability
                1e-4,  # sensing_threshold
                1e-4,  # min_noise
                0.5,  # standard_deviation_percentage
        )
    ],
    ids=["pos1_no_params", "pos2_params", "pos3_params"]
)
def test_gas(position, missed_detection_probability, sensing_threshold, min_noise,
             standard_deviation_percentage):

    start_time = datetime.now()

    if missed_detection_probability is None:
        sensor = GasIntensitySensor()
        assert sensor.missed_detection_probability == 0.1
        assert sensor.standard_deviation_percentage == 0.5
        assert sensor.min_noise == 1e-4
        assert sensor.sensing_threshold == 1e-4
    else:
        sensor = GasIntensitySensor(missed_detection_probability=missed_detection_probability,
                                    sensing_threshold=sensing_threshold,
                                    standard_deviation_percentage=standard_deviation_percentage,
                                    min_noise=min_noise,
                                    position=position)
        assert sensor.missed_detection_probability == missed_detection_probability
        assert sensor.standard_deviation_percentage == standard_deviation_percentage
        assert sensor.min_noise == min_noise
        assert sensor.sensing_threshold == sensing_threshold
        assert np.all(sensor.position == position)

    source_truth = GroundTruthState([30,  # x
                                    40,  # y
                                    1,  # z
                                    5,  # Q
                                    4,  # u
                                    np.radians(90),  # phi
                                    1,  # ci
                                    8],  # cii
                                    timestamp=start_time)

    # Generate noiseless measurement
    measurement = sensor.measure({source_truth}, noise=False)
    measurement = next(iter(measurement))

    assert isinstance(measurement, TrueDetection)
    assert measurement.timestamp == start_time

    # Check measurement
    eval_meas = isoplume_h(source_truth.state_vector.view(np.ndarray), position)
    assert np.equal(measurement.state_vector, eval_meas)

    # Generate noisy measurement
    np.random.seed(1990)
    measurement = sensor.measure({source_truth}, noise=True, random_state=1990)
    measurement = next(iter(measurement))

    rng = np.random.RandomState(1990)
    np.random.seed(1990)
    eval_meas += eval_meas * sensor.standard_deviation_percentage * rng.normal()
    eval_meas[eval_meas < sensor.sensing_threshold] = 0
    eval_meas[:, np.random.uniform() < sensor.missed_detection_probability] = 0

    assert np.equal(measurement.state_vector, eval_meas)
