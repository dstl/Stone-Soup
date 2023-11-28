#!/usr/bin/env python

"""
Reinforcement Learning Sensor Manager
=====================================
This example looks at how to interface a reinforcement learning framework with a Stone Soup sensor manager.
"""

# %%
# Making a Reinforcement Learning Sensor Manager
# ----------------------------------------------
# This example introduces using a Deep Q Network (DQN) reinforcement learning (RL) sensor management algorithm
# in Stone Soup. This is compared to the performance of a brute force algorithm using the same metrics shown in the
# sensor management tutorials. This example is similar to the sensor management tutorials, simulating 3 targets and a
# :class:`~.RadarRotatingBearingRange` sensor which can be actioned to point in different directions.
# 
# Tensorflow-agents is used as the reinforcement learning framework. This is a separate python package that can be found
# at https://github.com/tensorflow/agents. This currently only works on Linux based OSes, or via Windows Subsystem for
# Linux (WSL). See Tensorflow instructions for creating environments (with GPU support if applicable) [#]_.
#
# To run this example, in a clean environment, do  ``pip install stonesoup``, followed by ``pip install
# tf-agents[reverb]``.

# Some general imports and set up
import numpy as np
import random
from datetime import datetime, timedelta

start_time = datetime.now().replace(microsecond=0)

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

# %%
# Generate ground truths
# ----------------------
# Following the methods from previous Stone Soup sensor management tutorials, generate a series of combined linear
# Gaussian transition models and generate ground truths. Each ground truth is offset in the y-direction by 10.
# 
# The number of targets in this simulation is defined by `ntruths` - here there are 3 targets travelling in different
# directions. The time the simulation is observed for is defined by `time_max`.
# 
# We can fix our random number generator in order to probe a particular example repeatedly. To produce random examples,
# comment out the next two lines.

np.random.seed(1990)
random.seed(1990)

# Generate transition model
# i.e. fk(xk|xk-1)
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                          ConstantVelocity(0.005)])

yps = range(0, 100, 10)  # y value for prior state
truths = []
ntruths = 3  # number of ground truths in simulation
time_max = 50  # timestamps the simulation is observed over
timesteps = [start_time + timedelta(seconds=k) for k in range(time_max)]

xdirection = 1
ydirection = 1

# Generate ground truths
for j in range(0, ntruths):
    truth = GroundTruthPath([GroundTruthState([0, xdirection, yps[j], ydirection],
                                              timestamp=start_time)],
                            id=f"id{j}")

    for k in range(1, time_max):
        truth.append(
            GroundTruthState(transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
                             timestamp=start_time + timedelta(seconds=k)))
    truths.append(truth)

    # alternate directions when initiating tracks
    xdirection *= -1
    if j % 2 == 0:
        ydirection *= -1

# %%
# Plot the ground truths. This is done using the :class:`~.Plotterly` class from Stone Soup.

from stonesoup.plotter import AnimatedPlotterly

plotter = AnimatedPlotterly(timesteps, tail_length=1)
plotter.plot_ground_truths(truths, [0, 2])
plotter.fig

# %%
# Create sensors
# --------------
# Create a sensor for each sensor management algorithm. This tutorial uses the
# :class:`~.RadarRotatingBearingRange` sensor. This sensor is an :class:`~.Actionable` so
# is capable of returning the actions it can take at a given time step and can also be given an action to take before
# measuring.
# See the :doc:`Creating an Actionable Sensor Example <Creating_Actionable_Sensor>` for a more
# detailed explanation of actionable sensors.
#
# The :class:`~.RadarRotatingBearingRange` has a dwell centre which is an :class:`~.ActionableProperty`
# so in this case the action is changing the dwell centre to point in a specific direction.
# 

from stonesoup.types.state import StateVector
from stonesoup.sensor.radar.radar import RadarRotatingBearingRange

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

sensorB = RadarRotatingBearingRange(
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
sensorB.timestamp = start_time

# %%
# Create the Kalman predictor and updater
# ---------------------------------------
# Construct a predictor and updater using the :class:`~.KalmanPredictor` and :class:`~.ExtendedKalmanUpdater`
# components from Stone Soup. The :class:`~.ExtendedKalmanUpdater` is used because it can be used for both linear
# and nonlinear measurement models. A hypothesiser and data associator are required for use in both trackers.
# 

from stonesoup.predictor.kalman import KalmanPredictor

predictor = KalmanPredictor(transition_model)

from stonesoup.updater.kalman import ExtendedKalmanUpdater

updater = ExtendedKalmanUpdater(measurement_model=None)
# measurement model is added to detections by the sensor

from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis

hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=5)

from stonesoup.dataassociator.neighbour import GNNWith2DAssignment

data_associator = GNNWith2DAssignment(hypothesiser)

# %%
# Generate Priors
# ----------------------
# First create `ntruths` priors which estimate the targets’ initial states, one for each target. In this example
# each prior is offset by 0.1 in the y direction meaning the position of the track is initially not very accurate. The
# velocity is also systematically offset by +0.2 in both the x and y directions.
# 
# 

from stonesoup.types.state import GaussianState

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

# %%
# Initialise the tracks by creating an empty list and appending the priors generated. This needs to be done separately
# for both sensor manager methods as they will generate different sets of tracks.
# 

from stonesoup.types.track import Track

# Initialise tracks from the RandomSensorManager
tracksA = []
for j, prior in enumerate(priors):
    tracksA.append(Track([prior]))

tracksB = []
for j, prior in enumerate(priors):
    tracksB.append(Track([prior]))

# %%
# Reward function
# ---------------
# A reward function is used to quantify the benefit of sensors taking a particular action or set of actions.
# This can be crafted specifically for an example in order to achieve a particular objective. The function used in
# this example is quite generic but could be substituted for any callable function which returns a numeric
# value that the sensor manager can maximise.
# 
# The :class:`~.UncertaintyRewardFunction` calculates the uncertainty reduction by computing the difference between the
# covariance matrix norms of the prediction, and the posterior assuming a predicted measurement corresponding to that
# prediction.

from stonesoup.sensormanager.reward import UncertaintyRewardFunction

reward_function = UncertaintyRewardFunction(predictor=predictor, updater=updater)

# %%
# Reinforcement Learning
# ----------------------
# Reinforcement learning involves intelligent agents making decisions to maximise a cumulative reward. The agent
# must train in an environment in order to create a policy, which later determines the actions it will take. During
# training, the agent makes decisions and receives rewards, which it uses to optimise the policy.
# 
# .. figure:: ../_static/rl_training.png
#   :width: 800
#   :alt: Illustration of sequential actions and measurements
#
#   Illustration of an RL algorithm taking actions during training. The state and reward it receives are used to
#   determine the best actions.
# 
# Once training has completed, the policy can be exploited to gain rewards.
# 

# %%
# Design Environment
# ------------------
# An environment is needed for the RL agent to learn in. There are resources online for how to design these [#]_.
#
# In this example, the action space is equal to the number of targets in the simulation, so at each time step, the
# sensor can select one target to look at. For the environment, we make a copy of the sensor that we will pass to the
# sensor manager later on. This is so the agent can train in the environment without altering the sensor itself.
# The :class:`~.UncertaintyRewardFunction` is used to calculate the reward obtained for each step in the environment.
# The trace of the covariances for each object is used as the observation for the agent to learn from - it should learn
# to select targets with a larger covariance (higher uncertainty).

from abc import ABC
import numpy as np
import copy
from ordered_set import OrderedSet

from stonesoup.sensor.action.dwell_action import DwellActionsGenerator
from stonesoup.functions import mod_bearing

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import utils


class StoneSoupEnv(py_environment.PyEnvironment, ABC):
    """Example reinforcement learning environment. Environments must contain __init__, _reset,
    _step, and generate_action methods
    """

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
        # Use deepcopy to prevent the original sensor/tracks being changed each time an episode is run
        self.sensor = copy.deepcopy(sensorA)
        self.sensor.timestamp = start_time
        self.tracks = copy.deepcopy(tracksA)

    def action_spec(self):
        """Return action_spec."""
        return self._action_spec

    def observation_spec(self):
        """Return observation_spec."""
        return self._observation_spec

    def _reset(self):
        """Restarts the environment from the first step, resets the initial state
        and observation values, and returns an initial observation
        """
        self._episode_ended = False
        self.current_step = 0
        self.sensor = copy.deepcopy(sensorA)
        self.sensor.timestamp = start_time
        self.tracks = copy.deepcopy(tracksA)
        return ts.restart(np.zeros((self.obs_size,), dtype=np.float32))

    def _step(self, action):
        """Apply action and take one step through environment, and return new time_step.
        """

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

        self.sensor.add_actions(current_action)
        self.sensor.act(next_timestep)

        # Calculate a measurement from the sensor
        measurement = set()
        measurement |= self.sensor.measure(OrderedSet(truth[current_timestep] for truth in truths), noise=True)
        hypotheses = data_associator.associate(self.tracks,
                                               measurement,
                                               current_timestep)

        for track in self.tracks:
            hypothesis = hypotheses[track]
            if hypothesis.measurement:
                post = updater.update(hypothesis)
                track.append(post)
            else:  # When data associator says no detections are good enough, we'll keep the prediction
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
    def generate_action(action, tracks, sensor):
        """This method is used to convert a tf-agents action into a Stone Soup action"""
        for i, target in enumerate(tracks):
            if i == action:
                x_target = target.state.state_vector[0] - sensor.position[0]
                y_target = target.state.state_vector[2] - sensor.position[1]
                action_bearing = mod_bearing(np.arctan2(y_target, x_target))

        action_generators = DwellActionsGenerator(sensor,
                                                  attribute='dwell_centre',
                                                  start_time=sensor.timestamp,
                                                  end_time=sensor.timestamp + timedelta(seconds=1))

        current_action = [action_generators.action_from_value(action_bearing)]
        return current_action


# Validate the environment to ensure that the environment returns the expected specs
train_env = StoneSoupEnv()
utils.validate_py_environment(train_env, episodes=5)

# %%
# RL Sensor Manager
# -----------------
# To be able to use the RL environment we have designed, we need to make a ReinforcementSensorManager class, which
# inherits from :class:`~.SensorManager`.
# 
# We introduce some additional methods that are used by tensorflow-agents: :func:`compute_avg_return`,
# :func:`dense_layer`, and :func:`train`.
# :func:`compute_avg_return` is used to find the average reward by using a given policy. This is used to evaluate the
# training.
# :func:`dense_layer` is used when generating the Q-Network, a neural network model that learns to predict Q-Values.
# :func:`train` is used to generate the policy by running a large number of episodes through the Q-Network to work out
# which actions are best. An episode in RL refers to a single run or instance of the learning process, where the agent
# interacts with the environment.
# 
# We also need to re-define the :func:`choose_actions` method from :class:`~.SensorManager` to be able to interface
# Stone Soup actions with tensorflow-agent actions.

from stonesoup.sensormanager.base import SensorManager
from stonesoup.base import Property
from tf_agents.environments import tf_py_environment


class ReinforcementSensorManager(SensorManager):
    """A sensor manager that employs reinforcement learning algorithms from tensorflow-agents.
    The sensor manager trains on an environment to find an optimal policy, which is then exploited
    to choose actions.
    """
    env: py_environment.PyEnvironment = Property(doc="The environment which the agent learns the policy with.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tf_env = tf_py_environment.TFPyEnvironment(self.env)
        self.test_env = tf_py_environment.TFPyEnvironment(self.env)
        self.agent = None

    @staticmethod
    def compute_avg_return(environment, policy, num_episodes=10):
        """Used to calculate the average reward over a set of episodes.

        Parameters
        ----------
        environment:
            tf-agents environment for evaluating policy on

        policy:
            tf-agents policy for choosing actions in environment

        num_episodes: int
            Number of episodes to sample over

        Returns
        -------
        : int
            average reward calculated over num_episodes

        """
        time_step = None
        episode_return = None
        total_return = 0.0
        for _ in range(num_episodes):
            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    @staticmethod
    def dense_layer(num_units):
        """Method for generating fully connected layers for use in the neural network.

        Parameters
        ----------
        num_units: int
            Number of nodes in dense layer

        Returns
        -------
        : tensorflow dense layer

        """
        # Define a helper function to create Dense layers configured with the right
        # activation and kernel initializer.
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal'))

    def train(self, hyper_parameters):
        """Trains a DQN agent on the specified environment to learn a policy that is later
        used to select actions.

        Parameters
        ----------
        hyper_parameters: dict
            Dictionary containing hyperparameters used in training. See tutorial for
            necessary hyperparameters.

        """
        if self.env is not None:
            self.env.reset()

            train_py_env = self.env
            eval_py_env = self.env
            self.train_env = tf_py_environment.TFPyEnvironment(train_py_env)
            self.eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

            fc_layer_params = hyper_parameters['fc_layer_params']
            action_tensor_spec = tensor_spec.from_spec(self.env.action_spec())
            num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

            # QNetwork consists of a sequence of Dense layers followed by a dense layer
            # with `num_actions` units to generate one q_value per available action as
            # its output.

            dense_layers = [self.dense_layer(num_units) for num_units in fc_layer_params]
            q_values_layer = tf.keras.layers.Dense(
                num_actions,
                activation=None,
                kernel_initializer=tf.keras.initializers.RandomUniform(
                    minval=-0.03, maxval=0.03),
                bias_initializer=tf.keras.initializers.Constant(-0.2))
            q_net = sequential.Sequential(dense_layers + [q_values_layer])

            optimizer = tf.keras.optimizers.Adam(hyper_parameters['learning_rate'])

            train_step_counter = tf.Variable(0)

            self.agent = dqn_agent.DdqnAgent(
                self.train_env.time_step_spec(),
                self.train_env.action_spec(),
                q_network=q_net,
                optimizer=optimizer,
                td_errors_loss_fn=common.element_wise_squared_loss,
                train_step_counter=train_step_counter)

            self.agent.initialize()

            random_policy = random_tf_policy.RandomTFPolicy(self.train_env.time_step_spec(),
                                                            self.train_env.action_spec())

            # See also the metrics module for standard implementations of different metrics.
            # https://github.com/tensorflow/agents/tree/master/tf_agents/metrics

            self.compute_avg_return(self.eval_env, random_policy,
                                    hyper_parameters['num_eval_episodes'])

            table_name = 'uniform_table'
            replay_buffer_signature = tensor_spec.from_spec(
                self.agent.collect_data_spec)
            replay_buffer_signature = tensor_spec.add_outer_dim(
                replay_buffer_signature)

            table = reverb.Table(
                table_name,
                max_size=hyper_parameters['replay_buffer_max_length'],
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                rate_limiter=reverb.rate_limiters.MinSize(1),
                signature=replay_buffer_signature)

            reverb_server = reverb.Server([table])

            replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
                self.agent.collect_data_spec,
                table_name=table_name,
                sequence_length=2,
                local_server=reverb_server)

            rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
                replay_buffer.py_client,
                table_name,
                sequence_length=2)

            py_driver.PyDriver(
                self.env,
                py_tf_eager_policy.PyTFEagerPolicy(
                    random_policy, use_tf_function=True),
                [rb_observer],
                max_steps=hyper_parameters['initial_collect_steps']).run(train_py_env.reset())

            # Dataset generates trajectories with shape [Bx2x...]
            dataset = replay_buffer.as_dataset(
                num_parallel_calls=3,
                sample_batch_size=hyper_parameters['batch_size'],
                num_steps=2).prefetch(3)

            iterator = iter(dataset)

            # (Optional) Optimize by wrapping some code in a graph using TF function.
            self.agent.train = common.function(self.agent.train)

            # Reset the train step.
            self.agent.train_step_counter.assign(0)

            # Evaluate the agent's policy once before training.
            avg_return = self.compute_avg_return(self.eval_env, self.agent.policy,
                                                 hyper_parameters['num_eval_episodes'])
            returns = [avg_return]

            # Reset the environment.
            time_step = train_py_env.reset()

            # Create a driver to collect experience.
            collect_driver = py_driver.PyDriver(
                self.env,
                py_tf_eager_policy.PyTFEagerPolicy(
                    self.agent.collect_policy, use_tf_function=True),
                [rb_observer],
                max_steps=hyper_parameters['collect_steps_per_iteration'])

            for _ in range(hyper_parameters['num_iterations']):
                # Collect a few steps and save to the replay buffer.
                time_step, _ = collect_driver.run(time_step)

                # Sample a batch of data from the buffer and update the agent's network.
                experience, unused_info = next(iterator)
                train_loss = self.agent.train(experience).loss

                step = self.agent.train_step_counter.numpy()

                if step % hyper_parameters['log_interval'] == 0:
                    print('step = {0}: loss = {1}'.format(step, train_loss))

                if step % hyper_parameters['eval_interval'] == 0:
                    # Agent Policy Output
                    avg_return = self.compute_avg_return(self.eval_env, self.agent.policy,
                                                         hyper_parameters['num_eval_episodes'])
                    returns.append(avg_return)
                    print('step = {0}: Average Return = {1}'.format(step, avg_return))
                    if ('max_train_reward' in hyper_parameters) and \
                            (avg_return > hyper_parameters['max_train_reward']):
                        break

            print('\n-----\nTraining complete\n-----')

    def choose_actions(self, tracks, sensors, timestamp, nchoose=1, **kwargs):
        """Returns a chosen [list of] action(s) from the action set for each sensor.
        Chosen action(s) is selected by exploiting the reinforcement learning agent's
        policy that was found during training.

        Parameters
        ----------
        tracks: set of :class:`~Track`
            Set of tracks at given time. Used in reward function.
        sensors: :class:`~Sensor`
            Sensor(s) used for observation
        timestamp: :class:`tf_agents.trajectories.TimeSpec`
            Timestep of environment at current time
        nchoose : int
            Number of actions from the set to choose (default is 1)

        Returns
        -------
        : dict
            The pairs of :class:`~.Sensor`: [:class:`~.Action`] selected
        """

        configs = [dict() for _ in range(nchoose)]
        for sensor_action_assignment in configs:
            for sensor in sensors:
                chosen_actions = []
                action_step = self.agent.policy.action(timestamp)
                action = action_step.action
                stonesoup_action = self.env.generate_action(action, tracks, sensor)
                chosen_actions.append(stonesoup_action)
                sensor_action_assignment[sensor] = chosen_actions

            return configs


# %%
# Create Sensor Managers
# ----------------------
# We initiate our reinforcement learning sensor manager with the environment we have designed
# 

from stonesoup.sensormanager import BruteForceSensorManager

reinforcementsensormanager = ReinforcementSensorManager({sensorA}, env=StoneSoupEnv())
bruteforcesensormanager = BruteForceSensorManager({sensorB}, reward_function=reward_function)

# %%
# Train RL agent
# --------------
# To generate a policy, we need to train the reinforcement learning agent using the environment we created above.
# Some hyperparameters are created that the agent uses to train with.
# 
# To train the agent, the hyperparameters are passed to the train method in the :class:`~.ReinforcementSensorManager`.

import tensorflow as tf
import reverb
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy, random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

num_iterations = 10000

initial_collect_steps = 100
collect_steps_per_iteration = 1
replay_buffer_max_length = 100000

batch_size = 64
learning_rate = 1e-4
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

# %%
# Run the sensor managers
# -----------------------
# The :func:`choose_actions` function requires a time step and a tracks list as inputs.
#
# For both sensor management methods, the chosen actions are added to the sensor and measurements made. Tracks which
# have been observed by the sensor are updated and those that haven’t are predicted forward. These states are appended
# to the tracks list.
#

# %%
# Run reinforcement learning sensor manager
# -----------------------------------------
# To be able to exploit the policy generated by the reinforcement sensor manager, it  must be passed appropriate
# 'timesteps'.
# These are distinct from the timesteps in Stone Soup, and is of the form time_step_spec from tf-agents.

from itertools import chain

sensor_history_A = dict()

timesteps = []
for state in truths[0]:
    timesteps.append(state.timestamp)

tf_timestep = reinforcementsensormanager.test_env.reset()
reinforcementsensormanager.env.reset()
for timestep in timesteps[1:]:

    # Generate chosen configuration
    # i.e. {a}k
    # Need to make our own "timestamp" that matches tensorflow time_step_spec
    observation = []
    uncertainty = []
    for target in tracksA:
        x_target = target.state.state_vector[0] - sensorA.position[0]
        y_target = target.state.state_vector[2] - sensorA.position[1]
        bearing_target = mod_bearing(np.arctan2(y_target, x_target))
        uncertainty.append(np.trace(target.covar))

        # observation.append(np.degrees(bearing_target))
        observation.append(np.trace(target.covar))

    observation = np.array(uncertainty, dtype=np.float32)
    # observation = np.array(observation, dtype=np.float32)

    chosen_actions = reinforcementsensormanager.choose_actions(tracksA, [sensorA], tf_timestep)

    # Create empty dictionary for measurements
    measurementsA = []

    for chosen_action in chosen_actions:
        # chosen_action is a pair of {sensor, action}
        for sensor, actions in chosen_action.items():
            sensor.add_actions(list(chain.from_iterable(actions)))

    sensorA.act(timestep)

    # Store sensor history for plotting
    sensor_history_A[timestep] = copy.copy(sensorA)

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
        else:  # When data associator says no detections are good enough, we'll keep the prediction
            track.append(hypothesis.prediction)

    # Propagate environment
    action_step = reinforcementsensormanager.agent.policy.action(tf_timestep)
    tf_timestep = reinforcementsensormanager.test_env.step(action_step.action)

# %%
# Plot ground truths, tracks and uncertainty ellipses for each target.

import plotly.graph_objects as go
from stonesoup.functions import pol2cart

plotterA = AnimatedPlotterly(timesteps, tail_length=1, sim_duration=10)
plotterA.plot_sensors(sensorA)
plotterA.plot_ground_truths(truths, [0, 2])
plotterA.plot_tracks(tracksA, [0, 2], uncertainty=True, plot_history=False)


def plot_sensor_fov(fig, sensor_history):
    # Plot sensor field of view
    trace_base = len(fig.data)
    fig.add_trace(go.Scatter(mode='lines',
                             line=go.scatter.Line(color='black',
                                                  dash='dash')))

    for frame in fig.frames:
        traces_ = list(frame.traces)
        data_ = list(frame.data)
        x = [0, 0]
        y = [0, 0]
        timestring = frame.name
        timestamp = datetime.strptime(timestring, "%Y-%m-%d %H:%M:%S")

        if timestamp in sensor_history:
            sensor = sensor_history[timestamp]
            for i, fov_side in enumerate((-1, 1)):
                range_ = min(getattr(sensor, 'max_range', np.inf), 100)
                x[i], y[i] = pol2cart(range_,
                                      sensor.dwell_centre[0, 0]
                                      + sensor.fov_angle / 2 * fov_side) \
                             + sensor.position[[0, 1], 0]
        else:
            continue

        data_.append(go.Scatter(x=[x[0], sensor.position[0], x[1]],
                                y=[y[0], sensor.position[1], y[1]],
                                mode="lines",
                                line=go.scatter.Line(color='black',
                                                     dash='dash'),
                                showlegend=False))
        traces_.append(trace_base)
        frame.traces = traces_
        frame.data = data_


plot_sensor_fov(plotterA.fig, sensor_history_A)
plotterA.fig

# %%
# Run brute force sensor manager
# ------------------------------

sensor_history_B = dict()
for timestep in timesteps[1:]:

    # Generate chosen configuration
    # i.e. {a}k
    chosen_actions = bruteforcesensormanager.choose_actions(tracksB, timestep)

    # Create empty dictionary for measurements
    measurementsB = set()

    for chosen_action in chosen_actions:
        for sensor, actions in chosen_action.items():
            sensor.add_actions(actions)

    sensorB.act(timestep)

    # Store sensor history for plotting
    sensor_history_B[timestep] = copy.copy(sensorB)

    # Observe this ground truth
    # i.e. {z}k
    measurementsB |= sensorB.measure(OrderedSet(truth[timestep] for truth in truths), noise=True)

    hypotheses = data_associator.associate(tracksB,
                                           measurementsB,
                                           timestep)
    for track in tracksB:
        hypothesis = hypotheses[track]
        if hypothesis.measurement:
            post = updater.update(hypothesis)
            track.append(post)
        else:  # When data associator says no detections are good enough, we'll keep the prediction
            track.append(hypothesis.prediction)

# %%
# Plot ground truths, tracks and uncertainty ellipses for each target.

plotterB = AnimatedPlotterly(timesteps, tail_length=1, sim_duration=10)
plotterB.plot_sensors(sensorB)
plotterB.plot_ground_truths(truths, [0, 2])
plotterB.plot_tracks(tracksB, [0, 2], uncertainty=True, plot_history=False)
plot_sensor_fov(plotterB.fig, sensor_history_B)
plotterB.fig

# %%
# With a properly trained policy, the :class:`~.ReinforcementSensorManager` performs almost as well as the
# :class:`~.BruteForceSensorManager`. Also, once the policy has been learnt, the time taken to run the
# tracking loop is far smaller for the :class:`~.ReinforcementSensorManager` than for the
# :class:`~.BruteForceSensorManager`, which must re-calculate the best actions each time it is run.

# %%
# Metrics
# -------
# Metrics can be used to compare how well different sensor management techniques are working.
# Full explanations of the OSPA
# and SIAP metrics can be found in the :doc:`Metrics Example <Metrics>`.

from stonesoup.metricgenerator.ospametric import OSPAMetric

ospa_generatorA = OSPAMetric(c=40, p=1,
                             generator_name='ReinforcementSensorManager',
                             tracks_key='tracksA',
                             truths_key='truths')

ospa_generatorB = OSPAMetric(c=40, p=1,
                             generator_name='BruteForceSensorManager',
                             tracks_key='tracksB',
                             truths_key='truths')

from stonesoup.metricgenerator.tracktotruthmetrics import SIAPMetrics
from stonesoup.measures import Euclidean

siap_generatorA = SIAPMetrics(position_measure=Euclidean((0, 2)),
                              velocity_measure=Euclidean((1, 3)),
                              generator_name='ReinforcementSensorManager',
                              tracks_key='tracksA',
                              truths_key='truths')

siap_generatorB = SIAPMetrics(position_measure=Euclidean((0, 2)),
                              velocity_measure=Euclidean((1, 3)),
                              generator_name='BruteForceSensorManager',
                              tracks_key='tracksB',
                              truths_key='truths')

from stonesoup.dataassociator.tracktotrack import TrackToTruth

associator = TrackToTruth(association_threshold=30)

from stonesoup.metricgenerator.uncertaintymetric import SumofCovarianceNormsMetric

uncertainty_generatorA = SumofCovarianceNormsMetric(generator_name='ReinforcementSensorManager',
                                                    tracks_key='tracksA')

uncertainty_generatorB = SumofCovarianceNormsMetric(generator_name='BruteForceSensorManager',
                                                    tracks_key='tracksB')

# %%
# Generate a metrics manager.

from stonesoup.metricgenerator.manager import MultiManager

metric_manager = MultiManager([ospa_generatorA,
                               ospa_generatorB,
                               siap_generatorA,
                               siap_generatorB,
                               uncertainty_generatorA,
                               uncertainty_generatorB],
                              associator=associator)

# %%
# For each time step, data is added to the metric manager on truths and tracks. The metrics themselves can then be
# generated from the metric manager.

metric_manager.add_data({'truths': truths, 'tracksA': tracksA, 'tracksB': tracksB})

metrics = metric_manager.generate_metrics()

# %%
# OSPA metric
# -----------
# First we look at the OSPA metric. This is plotted over time for each sensor manager method.

from stonesoup.plotter import MetricPlotter

fig = MetricPlotter()
fig.plot_metrics(metrics, metric_names=['OSPA distances'])

# %%
# The :class:`~.BruteForceSensorManager` generally results in a smaller OSPA distance
# than the observations of the :class:`~.ReinforcementSensorManager`, reflecting the better tracking performance
# seen in the tracking plots. At some times, the OSPA distance for the :class:`~.ReinforcementSensorManager` is slightly
# lower than for the :class:`~.BruteForceSensorManager`. While it is intuitive to think that the brute force algorithm
# would always perform better, the brute force algorithm will pick the target that is most uncertain, and the
# reinforcement algorithm may pick another target that happens to reduce OSPA distance more.

# %%
# SIAP metrics
# ------------
# Next we look at SIAP metrics. We are only interested in the positional accuracy (PA) and velocity accuracy (VA).
# These metrics can be plotted to show how they change over time.


fig2 = MetricPlotter()
fig2.plot_metrics(metrics, metric_names=['SIAP Position Accuracy at times',
                                         'SIAP Velocity Accuracy at times'])

# %%
# Similar to the OSPA distances the :class:`~.BruteForceSensorManager`
# generally results in both a better positional accuracy and velocity accuracy than the observations
# of the :class:`~.ReinforcementSensorManager`.

# %%
# Uncertainty metric
# ------------------
# Finally we look at the uncertainty metric which computes the sum of covariance matrix norms of each state at each
# time step. This is plotted over time for each sensor manager method.

fig3 = MetricPlotter()
fig3.plot_metrics(metrics, metric_names=['Sum of Covariance Norms Metric'])

# %%
# This metric shows that the uncertainty in the tracks generated by the :class:`~.ReinforcementSensorManager` is a
# little higher than for those generated by the :class:`~.BruteForceSensorManager`. This is also reflected by the
# uncertainty ellipses in the initial plots of tracks and truths.

# %%
# References
# ----------
# .. [#] *https://www.tensorflow.org/install/pip#windows-wsl2*
# .. [#] *https://github.com/tensorflow/agents/blob/master/docs/tutorials/2_environments_tutorial.ipynb*

# sphinx_gallery_thumbnail_number = 3
