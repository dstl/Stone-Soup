# -*- coding: utf-8 -*-

import numpy as np
from datetime import datetime, timedelta

from ...types.array import StateVector
from ...types.state import GaussianState
from ...types.track import Track
from ...sensor.radar import RadarRotatingBearingRange
from ...sensor.action.dwell_action import ChangeDwellAction
from ...sensormanager import RandomSensorManager, BruteForceSensorManager
from ...sensormanager.reward import UncertaintyRewardFunction
from ...sensormanager.optimise import OptimizeBruteSensorManager, \
    OptimizeBasinHoppingSensorManager
from ...predictor.kalman import KalmanPredictor
from ...updater.kalman import ExtendedKalmanUpdater
from ...models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                    ConstantVelocity


def test_random_choose_actions():
    time_start = datetime.now()

    dwell_centres = []
    for i in range(3):
        sensors = {RadarRotatingBearingRange(
            position_mapping=(0, 2),
            noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                                  [0, 0.75 ** 2]]),
            ndim_state=4,
            rpm=60,
            fov_angle=np.radians(30),
            dwell_centre=StateVector([0.0]),
            max_range=100,
        )}

        for sensor in sensors:
            sensor.timestamp = time_start
        sensor_manager = RandomSensorManager(sensors)

        chosen_action_configs = sensor_manager.choose_actions({},
                                                              time_start + timedelta(seconds=1))

        assert type(chosen_action_configs) == list

        for chosen_config in chosen_action_configs:
            for sensor, actions in chosen_config.items():
                sensor.add_actions(actions)
                sensor.act(time_start + timedelta(seconds=1))
                dwell_centres.append(sensor.dwell_centre)

                assert isinstance(sensor, RadarRotatingBearingRange)
                assert isinstance(actions[0], ChangeDwellAction)


def test_brute_force_choose_actions():
    time_start = datetime.now()

    tracks = [Track(states=[
        GaussianState([[1], [1], [1], [1]],
                      np.diag([1.5, 0.25, 1.5, 0.25] + np.random.normal(0, 5e-4, 4)),
                      timestamp=time_start),
        GaussianState([[2], [1.5], [2], [1.5]],
                      np.diag([3, 0.5, 3, 0.5] + np.random.normal(0, 5e-4, 4)),
                      timestamp=time_start + timedelta(seconds=1))]),
              Track(states=[
                  GaussianState([[-1], [1], [-1], [1]],
                                np.diag([3, 0.5, 3, 0.5] + np.random.normal(0, 5e-4, 4)),
                                timestamp=time_start),
                  GaussianState([[2], [1.5], [2], [1.5]],
                                np.diag([1.5, 0.25, 1.5, 0.25] + np.random.normal(0, 5e-4, 4)),
                                timestamp=time_start + timedelta(seconds=1))])]

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                              ConstantVelocity(0.005)])
    predictor = KalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=None)
    reward_function = UncertaintyRewardFunction(predictor, updater)

    all_dwell_centres = []

    for i in range(3):
        sensorsA = {RadarRotatingBearingRange(
            position_mapping=(0, 2),
            noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                                  [0, 0.75 ** 2]]),
            position=np.array([[0], [0]]),
            ndim_state=4,
            rpm=60,
            fov_angle=np.radians(30),
            dwell_centre=StateVector([0.0]),
            max_range=np.inf,
        )}

        sensorsB = {RadarRotatingBearingRange(
            position_mapping=(0, 2),
            noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                                  [0, 0.75 ** 2]]),
            position=np.array([[0], [0]]),
            ndim_state=4,
            rpm=60,
            fov_angle=np.radians(30),
            dwell_centre=StateVector([0.0]),
            max_range=np.inf,
        )}

        sensorsC = {RadarRotatingBearingRange(
            position_mapping=(0, 2),
            noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                                  [0, 0.75 ** 2]]),
            position=np.array([[0], [0]]),
            ndim_state=4,
            rpm=60,
            fov_angle=np.radians(30),
            dwell_centre=StateVector([0.0]),
            max_range=np.inf,
        )}

        for sensor_set in [sensorsA, sensorsB, sensorsC]:
            for sensor in sensor_set:
                sensor.timestamp = time_start

        sensor_managerA = BruteForceSensorManager(sensorsA, reward_function)
        sensor_managerB = OptimizeBruteSensorManager(sensorsB, reward_function)
        sensor_managerC = OptimizeBasinHoppingSensorManager(sensorsC,
                                                            reward_function)

        sensor_managers = [sensor_managerA,
                           sensor_managerB,
                           sensor_managerC]

        timesteps = []
        for t in range(3):
            timesteps.append(time_start + timedelta(seconds=t))

        dwell_centres_for_i = []
        for sensor_manager in sensor_managers:
            dwell_centres_over_time = []
            for time in timesteps:
                chosen_action_configs = sensor_manager.choose_actions(tracks, time)

                for chosen_config in chosen_action_configs:
                    for sensor, actions in chosen_config.items():
                        sensor.add_actions(actions)
                        sensor.act(time)
                        dwell_centres_over_time.append(sensor.dwell_centre)

                        assert isinstance(sensor, RadarRotatingBearingRange)
                        assert isinstance(actions[0], ChangeDwellAction)

            dwell_centres_for_i.append(dwell_centres_over_time)

        all_dwell_centres.append(dwell_centres_for_i)
        for t in range(3):
            difference_between_managersAB = dwell_centres_for_i[0][t] - dwell_centres_for_i[1][t]
            difference_between_managersAC = dwell_centres_for_i[0][t] - dwell_centres_for_i[2][t]
            difference_between_managersBC = dwell_centres_for_i[1][t] - dwell_centres_for_i[2][t]
            assert difference_between_managersAB <= np.radians(60)
            assert difference_between_managersAC <= np.radians(60)
            assert difference_between_managersBC <= np.radians(60)

    assert all_dwell_centres[0][0] == all_dwell_centres[1][0] == all_dwell_centres[2][0]
    assert all_dwell_centres[0][1] == all_dwell_centres[1][1] == all_dwell_centres[2][1]
