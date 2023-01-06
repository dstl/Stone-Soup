import numpy as np
from datetime import datetime, timedelta
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import utils
import copy
from ordered_set import OrderedSet
from itertools import chain

from ...types.array import StateVector
from ...types.state import GaussianState
from ...types.track import Track
from ...types.groundtruth import GroundTruthPath, GroundTruthState

from ...functions import mod_bearing
from ...hypothesiser.distance import DistanceHypothesiser
from ...measures import Mahalanobis
from ...dataassociator.neighbour import GNNWith2DAssignment
from ...sensor.action.dwell_action import DwellActionsGenerator
from ...sensor.action.dwell_action import ChangeDwellAction
from ...sensor.radar import RadarRotatingBearingRange
from ...sensormanager import RandomSensorManager, BruteForceSensorManager
from ...sensormanager.reward import UncertaintyRewardFunction
from ...sensormanager.optimise import OptimizeBruteSensorManager, \
    OptimizeBasinHoppingSensorManager
from ...sensormanager.reinforcement import ReinforcementSensorManager, BaseEnvironment
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


def test_reinforcement_manager():
    np.random.seed(1990)
    start_time = datetime.now()

    # Generate transition model
    # i.e. fk(xk|xk-1)
    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                              ConstantVelocity(0.005)])

    yps = range(0, 100, 10)  # y value for prior state
    truths = []
    ntruths = 3  # number of ground truths in simulation
    time_max = 50  # timestamps the simulation is observed over

    xdirection = 1
    ydirection = 1

    # Generate ground truths
    for j in range(0, ntruths):
        truth = GroundTruthPath([GroundTruthState([0, xdirection, yps[j], ydirection],
                                                  timestamp=start_time)],
                                id=f"id{j}")

        for k in range(1, time_max):
            truth.append(
                GroundTruthState(
                    transition_model.function(truth[k - 1], noise=True,
                                              time_interval=timedelta(seconds=1)),
                    timestamp=start_time + timedelta(seconds=k)))
        truths.append(truth)

        # alternate directions when initiating tracks
        xdirection *= -1
        if j % 2 == 0:
            ydirection *= -1

    predictor = KalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=None)
    reward_function = UncertaintyRewardFunction(predictor=predictor, updater=updater)
    hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(),
                                        missed_distance=5)
    data_associator = GNNWith2DAssignment(hypothesiser)

    sensorA = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                              [0, 1 ** 2]]),
        ndim_state=4,
        position=np.array([[10], [0]]),
        rpm=60,
        fov_angle=np.radians(45),
        dwell_centre=StateVector([0.0]),
        max_range=np.inf
    )
    sensorA.timestamp = start_time

    priors = []
    xdirection = 1.2
    ydirection = 1.2
    for j in range(0, ntruths):
        priors.append(GaussianState([[0], [xdirection], [yps[j] + 0.1], [ydirection]],
                                    np.diag([0.5, 0.5, 0.5, 0.5] + np.random.normal(0, 5e-4, 4)),
                                    timestamp=start_time))
        xdirection *= -1
        if j % 2 == 0:
            ydirection *= -1

    tracksA = []
    for j, prior in enumerate(priors):
        tracksA.append(Track([prior]))

    class StoneSoupEnv(BaseEnvironment):

        def __init__(self):
            super().__init__()
            # Action size is number of targets
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(), dtype=np.int32, minimum=0, maximum=ntruths - 1, name='action')
            # Observation size is also number of targets
            self.obs_size = ntruths
            self._observation_spec = array_spec.BoundedArraySpec(
                shape=(self.obs_size,), dtype=np.float32, name='observation')
            self._episode_ended = False
            self.max_episode_length = time_max
            self.current_step = 0
            self.start_time = start_time
            # Use deepcopy to prevent the original sensor/tracks being changed
            self.sensor = copy.deepcopy(sensorA)
            self.sensor.timestamp = start_time
            self.tracks = copy.deepcopy(tracksA)

        def _reset(self):
            self._episode_ended = False
            self.current_step = 0
            self.sensor = copy.deepcopy(sensorA)
            self.sensor.timestamp = start_time
            self.tracks = copy.deepcopy(tracksA)
            return ts.restart(np.zeros((self.obs_size,), dtype=np.float32))

        def _step(self, action):

            reward = 0
            if self._episode_ended:
                # The last action ended the episode. Ignore the current action and start
                # a new episode.
                return self.reset()

            uncertainty = []
            for i, target in enumerate(self.tracks):
                # Calculate the bearing of the chosen target from the sensor
                if i == action:
                    x_target = target.state.state_vector[0] - self.sensor.position[0]
                    y_target = target.state.state_vector[2] - self.sensor.position[1]
                    bearing_target = mod_bearing(np.arctan2(y_target, x_target))

                uncertainty.append(np.trace(target.covar))

            current_timestep = self.start_time + timedelta(seconds=self.current_step)
            next_timestep = self.start_time + timedelta(seconds=self.current_step + 1)

            # Create action generator which contains possible actions
            action_generator = DwellActionsGenerator(self.sensor,
                                                     attribute='dwell_centre',
                                                     start_time=current_timestep,
                                                     end_time=next_timestep)

            # Action the environment's sensor to point towards the chosen target
            current_action = [action_generator.action_from_value(bearing_target)]
            config = ({self.sensor: current_action})
            reward += reward_function(config, self.tracks, next_timestep)

            # sorted_uncertainty = sorted(uncertainty)
            # for i in sorted_uncertainty:
            #     if action == uncertainty.index(i):
            #      reward = sorted_uncertainty.index(i)

            self.sensor.add_actions(current_action)
            self.sensor.act(next_timestep)

            # Calculate a measurement from the sensor
            measurement = set()
            measurement |= self.sensor.measure(OrderedSet(truth[current_timestep]
                                                          for truth in truths),
                                               noise=True)
            hypotheses = data_associator.associate(self.tracks,
                                                   measurement,
                                                   current_timestep)

            for track in self.tracks:
                hypothesis = hypotheses[track]
                if hypothesis.measurement:
                    post = updater.update(hypothesis)
                    track.append(post)
                else:
                    track.append(hypothesis.prediction)

            # Set the observation as the prior uncertainty of each target
            observation = np.array(uncertainty, dtype=np.float32)

            self.current_step += 1

            if self.current_step >= self.max_episode_length - 1:
                self._episode_ended = True
                return ts.termination(observation, reward)
            else:
                return ts.transition(observation, reward=reward, discount=1.0)

        @staticmethod
        def generate_action(action):
            """This method is used to convert a tf-agents action into a Stone Soup action"""
            for i, target in enumerate(tracksA):
                if i == action:
                    x_target = target.state.state_vector[0] - sensorA.position[0]
                    y_target = target.state.state_vector[2] - sensorA.position[1]
                    action_bearing = mod_bearing(np.arctan2(y_target, x_target))

            action_generators = DwellActionsGenerator(sensorA,
                                                      attribute='dwell_centre',
                                                      start_time=sensorA.timestamp,
                                                      end_time=sensorA.timestamp +
                                                      timedelta(seconds=1))

            current_action = [action_generators.action_from_value(action_bearing)]
            return current_action

    utils.validate_py_environment(StoneSoupEnv(), episodes=5)
    reinforcementsensormanager = ReinforcementSensorManager({sensorA}, env=StoneSoupEnv())

    # Hyper-parameters
    num_iterations = 10000  # @param {type:"integer"}

    initial_collect_steps = 100
    collect_steps_per_iteration = 1
    replay_buffer_max_length = 100000

    batch_size = 64
    learning_rate = 1e-5
    log_interval = 500

    num_eval_episodes = 10
    eval_interval = 1000
    fc_layer_params = (100, 50)

    # ---- Optional ----
    max_train_reward = 250

    hyper_parameters = {'num_iterations': num_iterations,
                        'initial_collect_steps': initial_collect_steps,
                        'collect_steps_per_iteration': collect_steps_per_iteration,
                        'replay_buffer_max_length': replay_buffer_max_length,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'log_interval': log_interval,
                        'num_eval_episodes': num_eval_episodes,
                        'eval_interval': eval_interval,
                        'fc_layer_params': fc_layer_params,
                        'max_train_reward': max_train_reward}

    reinforcementsensormanager.train(hyper_parameters)

    timesteps = []
    for state in truths[0]:
        timesteps.append(state.timestamp)

    tf_timestep = reinforcementsensormanager.test_env.reset()
    assert tf_timestep[3] == np.array([[0., 0., 0.]], dtype=np.float32)

    from tf_agents.trajectories.time_step import TimeStep

    assert isinstance(tf_timestep, TimeStep)

    for timestep in timesteps[1:]:

        # Generate chosen configuration
        # i.e. {a}k
        # Need to make our own "timestamp" that matches tensorflow time_step_spec
        observation = []
        uncertainty = []
        for target in tracksA:
            # x_target = target.state.state_vector[0] - sensorA.position[0]
            # y_target = target.state.state_vector[2] - sensorA.position[1]
            # bearing_target = mod_bearing(np.arctan2(y_target, x_target))
            uncertainty.append(np.trace(target.covar))

            # observation.append(np.degrees(bearing_target))
            observation.append(np.trace(target.covar))

        # observation = np.array(uncertainty, dtype=np.float32)
        # observation = np.array(observation, dtype=np.float32)

        chosen_actions = reinforcementsensormanager.choose_actions(tracksA, tf_timestep)

        # Create empty dictionary for measurements
        measurementsA = []

        for chosen_action in chosen_actions:
            # chosen_action is a pair of {sensor, action}
            for sensor, actions in chosen_action.items():
                sensor.add_actions(list(chain.from_iterable(actions)))

        sensorA.act(timestep)

        # Observe this ground truth
        # i.e. {z}k
        measurements = sensorA.measure(OrderedSet(truth[timestep] for truth in truths), noise=True)
        measurementsA.extend(measurements)

        hypotheses = data_associator.associate(tracksA,
                                               measurementsA,
                                               timestep)
        for track in tracksA:
            hypothesis = hypotheses[track]
            if hypothesis.measurement:
                post = updater.update(hypothesis)
                track.append(post)
            else:
                track.append(hypothesis.prediction)

        # Propagate environment
        action_step = reinforcementsensormanager.agent.policy.action(tf_timestep)
        tf_timestep = reinforcementsensormanager.test_env.step(action_step.action)
