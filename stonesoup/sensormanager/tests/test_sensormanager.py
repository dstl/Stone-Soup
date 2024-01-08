import numpy as np
from datetime import datetime, timedelta
from ordered_set import OrderedSet
from collections import defaultdict
import copy

import pytest

from ...types.array import StateVector, StateVectors
from ...types.state import GaussianState, ParticleState
from ...types.track import Track
from ...sensor.radar import RadarRotatingBearingRange
from ...sensor.action.dwell_action import ChangeDwellAction
from ...sensormanager import RandomSensorManager, BruteForceSensorManager, GreedySensorManager
from ...sensormanager.reward import UncertaintyRewardFunction, ExpectedKLDivergence, \
    MultiUpdateExpectedKLDivergence
from ...sensormanager.optimise import OptimizeBruteSensorManager, \
    OptimizeBasinHoppingSensorManager
from ...predictor.kalman import KalmanPredictor
from ...predictor.particle import ParticlePredictor
from ...updater.kalman import ExtendedKalmanUpdater
from ...updater.particle import ParticleUpdater
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

        assert isinstance(chosen_action_configs, list)

        for chosen_config in chosen_action_configs:
            for sensor, actions in chosen_config.items():
                sensor.add_actions(actions)
                sensor.act(time_start + timedelta(seconds=1))
                dwell_centres.append(sensor.dwell_centre)

                assert isinstance(sensor, RadarRotatingBearingRange)
                assert isinstance(actions[0], ChangeDwellAction)


@pytest.mark.slow
@pytest.mark.parametrize(
    "predictor_obj, updater_obj, reward_function_obj, track1_state1, track1_state2, "
    "track2_state1, track2_state2, error_flag",
    [
        (
            KalmanPredictor,  # predictor_obj
            ExtendedKalmanUpdater,  # updater_obj
            UncertaintyRewardFunction,  # reward_function_obj
            GaussianState([[1], [1], [1], [1]],
                          np.diag([1.5, 0.25, 1.5, 0.25] + np.random.normal(0, 5e-4, 4))),
            GaussianState([[2], [1.5], [2], [1.5]],
                          np.diag([3, 0.5, 3, 0.5] + np.random.normal(0, 5e-4, 4))),
            GaussianState([[-1], [1], [-1], [1]],
                          np.diag([3, 0.5, 3, 0.5] + np.random.normal(0, 5e-4, 4))),
            GaussianState([[2], [1.5], [2], [1.5]],
                          np.diag([1.5, 0.25, 1.5, 0.25] + np.random.normal(0, 5e-4, 4)),
                          ),
            False  # error_flag
        ), (
            ParticlePredictor,  # predictor_obj
            ParticleUpdater,  # updater_obj
            ExpectedKLDivergence,  # reward_function_obj
            ParticleState(state_vector=StateVectors(np.random.multivariate_normal(
                mean=np.array([1, 1, 1, 1]),
                cov=np.diag([1.5, 0.25, 1.5, 0.25]),
                size=100).T),
                          weight=np.array([1/100]*100)),
            ParticleState(state_vector=StateVectors(np.random.multivariate_normal(
                mean=np.array([2, 1.5, 2, 1.5]),
                cov=np.diag([3, 0.5, 3, 0.5]),
                size=100).T),
                          weight=np.array([1/100]*100)),
            ParticleState(state_vector=StateVectors(np.random.multivariate_normal(
                mean=np.array([-1, 1, -1, 1]),
                cov=np.diag([3, 0.5, 3, 0.5]),
                size=100).T),
                          weight=np.array([1/100]*100)),
            ParticleState(state_vector=StateVectors(np.random.multivariate_normal(
                mean=np.array([2, 1.5, 2, 1.5]),
                cov=np.diag([1.5, 0.25, 1.5, 0.25]),
                size=100).T),
                          weight=np.array([1/100]*100)),
            False  # error_flag
        ), (
            ParticlePredictor,  # predictor_obj
            ParticleUpdater,  # updater_obj
            MultiUpdateExpectedKLDivergence,  # reward_function_obj
            ParticleState(state_vector=StateVectors(np.random.multivariate_normal(
                mean=np.array([1, 1, 1, 1]),
                cov=np.diag([1.5, 0.25, 1.5, 0.25]),
                size=100).T),
                          weight=np.array([1/100]*100)),
            ParticleState(state_vector=StateVectors(np.random.multivariate_normal(
                mean=np.array([2, 1.5, 2, 1.5]),
                cov=np.diag([3, 0.5, 3, 0.5]),
                size=100).T),
                          weight=np.array([1/100]*100)),
            ParticleState(state_vector=StateVectors(np.random.multivariate_normal(
                mean=np.array([-1, 1, -1, 1]),
                cov=np.diag([3, 0.5, 3, 0.5]),
                size=100).T),
                          weight=np.array([1/100]*100)),
            ParticleState(state_vector=StateVectors(np.random.multivariate_normal(
                mean=np.array([2, 1.5, 2, 1.5]),
                cov=np.diag([1.5, 0.25, 1.5, 0.25]),
                size=100).T),
                          weight=np.array([1/100]*100)),
            False
        ), (
            KalmanPredictor,  # predictor_obj
            ExtendedKalmanUpdater,  # updater_obj
            ExpectedKLDivergence,  # reward_function_obj
            GaussianState([[1], [1], [1], [1]],
                          np.diag([1.5, 0.25, 1.5, 0.25] + np.random.normal(0, 5e-4, 4))),
            GaussianState([[2], [1.5], [2], [1.5]],
                          np.diag([3, 0.5, 3, 0.5] + np.random.normal(0, 5e-4, 4))),
            GaussianState([[-1], [1], [-1], [1]],
                          np.diag([3, 0.5, 3, 0.5] + np.random.normal(0, 5e-4, 4))),
            GaussianState([[2], [1.5], [2], [1.5]],
                          np.diag([1.5, 0.25, 1.5, 0.25] + np.random.normal(0, 5e-4, 4))),
            True
        ), (
            KalmanPredictor,  # predictor_obj
            ExtendedKalmanUpdater,  # updater_obj
            MultiUpdateExpectedKLDivergence,  # reward_function_obj
            GaussianState([[1], [1], [1], [1]],
                          np.diag([1.5, 0.25, 1.5, 0.25] + np.random.normal(0, 5e-4, 4))),
            GaussianState([[2], [1.5], [2], [1.5]],
                          np.diag([3, 0.5, 3, 0.5] + np.random.normal(0, 5e-4, 4))),
            GaussianState([[-1], [1], [-1], [1]],
                          np.diag([3, 0.5, 3, 0.5] + np.random.normal(0, 5e-4, 4))),
            GaussianState([[2], [1.5], [2], [1.5]],
                          np.diag([1.5, 0.25, 1.5, 0.25] + np.random.normal(0, 5e-4, 4))),
            True
        )
    ],
    ids=['UncertaintySMTest', 'KLDivergenceSMTest', 'MultiUpdateKLDivergenceSMTest',
         'KLDivergenceRaisesTest', 'MultiUpdateKLDivergenceRaisesTest']
)
def test_sensor_managers(predictor_obj, updater_obj, reward_function_obj, track1_state1,
                         track1_state2, track2_state1, track2_state2, error_flag):
    time_start = datetime.now()

    track1_state1.timestamp = time_start
    track2_state1.timestamp = time_start
    track1_state2.timestamp = time_start + timedelta(seconds=1)
    track2_state2.timestamp = time_start + timedelta(seconds=1)

    tracks = [Track(states=[
        track1_state1,
        track1_state2]),
        Track(states=[
            track2_state1,
            track2_state2
        ])]

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                              ConstantVelocity(0.005)])
    predictor = predictor_obj(transition_model)
    updater = updater_obj(measurement_model=None)

    if error_flag:
        # Check that raises function as expected
        with pytest.raises(NotImplementedError) as e:
            reward_function = reward_function_obj(predictor, updater, method_sum=False)
        assert 'Only ParticlePredictor types are currently compatible with this reward function'\
               in str(e.value)
        with pytest.raises(NotImplementedError) as e:
            reward_function = reward_function_obj(None, updater, method_sum=False)
        assert 'Only ParticleUpdater types are currently compatible with this reward function'\
               in str(e.value)
        if reward_function_obj == MultiUpdateExpectedKLDivergence:
            with pytest.raises(ValueError) as e:
                reward_function = reward_function_obj(method_sum=False, updates_per_track=1)
            assert f'updates_per_track = {1}. This reward function only accepts >= 2' in \
                   str(e.value)
        return

    reward_function = reward_function_obj(predictor, updater, method_sum=False)
    if isinstance(reward_function, MultiUpdateExpectedKLDivergence):
        reward_function = reward_function_obj(predictor, updater, method_sum=False,
                                              updates_per_track=3)

    timesteps = []
    for t in range(3):
        timesteps.append(time_start + timedelta(seconds=t + 2))

    sensor_rpm = 5

    sensorsA = {RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                              [0, 0.75 ** 2]]),
        position=np.array([[0], [0]]),
        ndim_state=4,
        rpm=sensor_rpm,
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
        rpm=sensor_rpm,
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
        rpm=sensor_rpm,
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
        rpm=sensor_rpm,
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

    all_dwell_centres = []
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

        all_dwell_centres.append(dwell_centres_over_time)

    for sm in range(3):
        # check that the sensors are not exceeding the maximum speed
        difference_between_t1t2 = \
            np.abs(np.min([all_dwell_centres[sm][0] - all_dwell_centres[sm][1],
                           all_dwell_centres[sm][1] - all_dwell_centres[sm][0]]))
        difference_between_t2t3 = \
            np.abs(np.min([all_dwell_centres[sm][1] - all_dwell_centres[sm][2],
                           all_dwell_centres[sm][2] - all_dwell_centres[sm][1]]))
        assert np.round(difference_between_t1t2, decimals=4) <= \
               np.round(np.radians(sensor_rpm*36/6), decimals=4)
        assert np.round(difference_between_t2t3, decimals=4) <= \
               np.round(np.radians(sensor_rpm*36/6), decimals=4)

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
    assert isinstance(chosen_actions, list)

    for chosen_actions in chosen_actions:
        for sensor, actions in chosen_action.items():
            assert isinstance(sensor, RadarRotatingBearingRange)
            assert isinstance(actions[0], ChangeDwellAction)

    # Check sensor following track as expected
    assert dwell_centres[timesteps[5]] - np.radians(135) < 1e-3
    assert dwell_centres[timesteps[15]] - np.radians(45) < 1e-3
    assert dwell_centres[timesteps[25]] - np.radians(-45) < 1e-3
    assert dwell_centres[timesteps[35]] - np.radians(-135) < 1e-3
