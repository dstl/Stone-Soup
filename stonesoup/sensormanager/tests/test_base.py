# -*- coding: utf-8 -*-

import numpy as np
import random
from datetime import datetime

from ...base import Base
from ...types.array import StateVector
from ...types.state import State
from ...types.track import Track
from ...sensor.radar import RadarRotatingBearingRange
from ...sensor.action.dwell_action import ChangeDwellAction
from ...sensormanager import RandomSensorManager, BruteForceSensorManager


def test_random_choose_actions():
    time_start = datetime.now()

    sensor = {RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                              [0, 0.75 ** 2]]),
        ndim_state=4,
        rpm=60,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([0.0]),
        max_range=100,
    )}

    sensor_manager = RandomSensorManager(sensor)

    chosen_action_configs = sensor_manager.choose_actions({}, time_start)
    assert type(chosen_action_configs) == list

    for chosen_config in chosen_action_configs:
        for sensor, action in chosen_config.items():
            assert isinstance(sensor, RadarRotatingBearingRange)
            assert isinstance(action[0], ChangeDwellAction)


def test_brute_force_choose_actions():
    time_start = datetime.now()

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

    track = [Track(states=[State(state_vector=[[0]],
                                 timestamp=time_start)])]

    class RewardFunction(Base):

        def calculate_reward(self, config, tracks_list, metric_time):
            config_metric = random.randint(0, 100)
            return config_metric

    reward_function = RewardFunction()

    sensor_manager = BruteForceSensorManager(sensors, reward_function.calculate_reward)

    chosen_action_configs = sensor_manager.choose_actions(track, time_start)

    for chosen_config in chosen_action_configs:
        for sensor, action in chosen_config.items():
            assert isinstance(sensor, RadarRotatingBearingRange)
            assert isinstance(action[0], ChangeDwellAction)
