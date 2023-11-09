import pytest

import numpy as np
from datetime import datetime, timedelta
from ordered_set import OrderedSet

from ...types.array import StateVector
from ...types.state import GaussianState
from ...types.track import Track
from ...sensor.radar import RadarRotatingBearingRange
from ...predictor.kalman import KalmanPredictor
from ...updater.kalman import ExtendedKalmanUpdater
from ...models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                    ConstantVelocity
from ...types.groundtruth import GroundTruthPath, GroundTruthState
from ...types.angle import Angle


@pytest.fixture
def params():
    start_time = datetime.now().replace(microsecond=0)

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.5),
                                                              ConstantVelocity(0.5)])

    time_max = 41
    timesteps = [start_time + timedelta(seconds=k) for k in range(time_max)]
    truths = OrderedSet()
    truth = GroundTruthPath([GroundTruthState([-10, 1, 0, 1], timestamp=timesteps[0])])
    for k in range(1, time_max):
        if k == 11 or k == 31:
            turn = truth[-1]
            turn.state_vector[3] *= -1
            truth.append(GroundTruthState(
                transition_model.function(turn, noise=False, time_interval=timedelta(seconds=1)),
                timestamp=timesteps[k]))
        elif k == 21:
            turn = truth[-1]
            turn.state_vector[1] *= -1
            truth.append(GroundTruthState(
                transition_model.function(turn, noise=False, time_interval=timedelta(seconds=1)),
                timestamp=timesteps[k]))
        else:
            truth.append(GroundTruthState(transition_model.function(truth[-1], noise=False,
                                                                    time_interval=timedelta(
                                                                        seconds=1)),
                                          timestamp=timesteps[k]))
    truths.add(truth)

    sensor_set = set()
    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[0.0001 ** 2, 0],
                              [0, 0.0001 ** 2]]),
        ndim_state=4,
        position=np.array([[0], [0]]),
        rpm=60,
        fov_angle=np.radians(90),
        dwell_centre=StateVector([np.radians(315)]),
        max_range=np.inf,
        resolutions={'dwell_centre': Angle(np.radians(90))}
    )
    sensor.timestamp = start_time
    sensor_set.add(sensor)

    predictor = KalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=None)

    prior = GaussianState([[-10], [1], [0], [1]],
                          np.diag([0.5, 0.5, 0.5, 0.5] + np.random.normal(0, 5e-4, 4)),
                          timestamp=start_time)
    tracks = {Track([prior])}

    return {'start_time': start_time, 'transition_model': transition_model,
            'predictor': predictor, 'updater': updater, 'sensor_set': sensor_set,
            'timesteps': timesteps, 'tracks': tracks, 'truths': truths}
