import numpy as np
from datetime import datetime, timedelta
from ordered_set import OrderedSet
from collections import defaultdict
import copy


from ...types.array import StateVector
from ...types.state import GaussianState
from ...types.track import Track

from ...sensor.action.dwell_action import ChangeDwellAction
from ...sensor.radar import RadarRotatingBearingRange
from ...sensormanager import RandomSensorManager, BruteForceSensorManager, GreedySensorManager

from ...sensormanager.reward import UncertaintyRewardFunction
from ...sensormanager.optimise import OptimizeBruteSensorManager, \
    OptimizeBasinHoppingSensorManager
from ...predictor.kalman import KalmanPredictor
from ...updater.kalman import ExtendedKalmanUpdater
from ...models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from ...hypothesiser.distance import DistanceHypothesiser
from ...measures import Mahalanobis
from ...dataassociator.neighbour import GNNWith2DAssignment


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


def test_uncertainty_based_managers():
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
    reward_function = UncertaintyRewardFunction(predictor, updater, method_sum=False)

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

        sensorsD = {RadarRotatingBearingRange(
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

        for sensor_set in [sensorsA, sensorsB, sensorsC, sensorsD]:
            for sensor in sensor_set:
                sensor.timestamp = time_start

        sensor_managerA = BruteForceSensorManager(sensorsA, reward_function)
        sensor_managerB = OptimizeBruteSensorManager(sensorsB, reward_function)
        sensor_managerC = OptimizeBasinHoppingSensorManager(sensorsC,
                                                            reward_function)
        sensor_managerD = OptimizeBruteSensorManager(sensorsD, reward_function,
                                                     generate_full_output=True,
                                                     finish=True)

        sensor_managers = [sensor_managerA,
                           sensor_managerB,
                           sensor_managerC,
                           sensor_managerD]

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

    assert isinstance(sensor_managerD.get_full_output(), tuple)


def test_greedy_manager(params):
    predictor = params['predictor']
    updater = params['updater']
    sensor_set = params['sensor_set']
    timesteps = params['timesteps']
    tracks = params['tracks']
    truths = params['truths']

    reward_function = UncertaintyRewardFunction(predictor, updater)
    greedysensormanager = GreedySensorManager(sensor_set, reward_function=reward_function)

    hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(),
                                        missed_distance=5)
    data_associator = GNNWith2DAssignment(hypothesiser)

    sensor_history = defaultdict(dict)
    dwell_centres = dict()

    for timestep in timesteps[1:]:
        chosen_actions = greedysensormanager.choose_actions(tracks, timestep)
        measurements = set()
        for chosen_action in chosen_actions:
            for sensor, actions in chosen_action.items():
                sensor.add_actions(actions)
        for sensor in sensor_set:
            sensor.act(timestep)
            sensor_history[timestep][sensor] = copy.copy(sensor)
            dwell_centres[timestep] = sensor.dwell_centre[0][0]
            measurements |= sensor.measure(OrderedSet(truth[timestep] for truth in truths),
                                           noise=False)
        hypotheses = data_associator.associate(tracks,
                                               measurements,
                                               timestep)
        for track in tracks:
            hypothesis = hypotheses[track]
            if hypothesis.measurement:
                post = updater.update(hypothesis)
                track.append(post)
            else:
                track.append(hypothesis.prediction)

    # Double check choose_actions method types are as expected
    assert type(chosen_actions) == list

    for chosen_actions in chosen_actions:
        for sensor, actions in chosen_action.items():
            assert isinstance(sensor, RadarRotatingBearingRange)
            assert isinstance(actions[0], ChangeDwellAction)

    # Check sensor following track as expected
    assert dwell_centres[timesteps[5]] - np.radians(135) < 1e-3
    assert dwell_centres[timesteps[15]] - np.radians(45) < 1e-3
    assert dwell_centres[timesteps[25]] - np.radians(-45) < 1e-3
    assert dwell_centres[timesteps[35]] - np.radians(-135) < 1e-3
