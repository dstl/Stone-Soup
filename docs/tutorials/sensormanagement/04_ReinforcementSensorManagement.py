#!/usr/bin/env python
# coding: utf-8

"""
==========================================================
4 - Reinforcement Learning Sensor Management
==========================================================
"""

# %%
# This tutorial introduces using a Deep Q Network (DQN) reinforcement learning sensor management algorithm in
# Stone Soup.
#
# This is compared to the performance of a brute force algorithm using the same metrics as in precious tutorials.
#
# This example is similar to previous examples, simulating 3 targets and a :class:`~.RadarRotatingBearingRange` sensor,
# which can be actioned to point in different directions.
#
# tensorflow-agents is used as the reinforcement learning framework. This currently only works on Linux based OSes, but
# a multi-platform solution may be investigated in the future.

# %%
# Sensor Management example
# -------------------------
# Setup
# ^^^^^
#
# First a simulation must be set up using components from Stone Soup. For this the following imports are required.#

# %%


import numpy as np
import random
from datetime import datetime, timedelta

start_time = datetime.now()

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState


# %%
# Generate ground truths
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Following the methods from previous Stone Soup tutorials, generate a series of combined linear Gaussian transition
# models and generate ground truths. Each ground truth is offset in the y-direction by 10.
#
# The number of targets in this simulation is defined by `ntruths` - here there are 3 targets travelling in different
# directions. The time the simulation is observed for is defined by `time_max`.
#
# We can fix our random number generator in order to probe a particular example repeatedly. This can be undone by
# commenting out the first two lines in the next cell.

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

xdirection = 1
ydirection = 1

# Generate ground truths
for j in range(0, ntruths):
    truth = GroundTruthPath([GroundTruthState([0, xdirection, yps[j], ydirection], timestamp=start_time)],
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

from stonesoup.plotter import Plotterly

# Stonesoup plotter requires sets not lists
truths_set = set(truths)

plotter = Plotterly()
plotter.plot_ground_truths(truths_set, [0, 2])
plotter.fig

# %%
# Create sensors
# ^^^^^^^^^^^^^^
# Create a sensor for each sensor management algorithm. This tutorial uses the
# :class:`~.RadarRotatingBearingRange` sensor. This sensor is an :class:`~.Actionable` so
# is capable of returning the actions it can possibly
# take at a given time step and can also be given an action to take before taking
# measurements.
# See the Creating an Actionable Sensor Example for a more detailed explanation of actionable sensors.
#
# The :class:`~.RadarRotatingBearingRange` has a dwell centre which is an :class:`~.ActionableProperty`
# so in this case the action is changing the dwell centre to point in a specific direction.

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
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Construct a predictor and updater using the :class:`~.KalmanPredictor` and :class:`~.ExtendedKalmanUpdater`
# components from Stone Soup. The :class:`~.ExtendedKalmanUpdater` is used because it can be used for both linear
# and nonlinear measurement models.

from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

from stonesoup.updater.kalman import ExtendedKalmanUpdater
updater = ExtendedKalmanUpdater(measurement_model=None)
# measurement model is added to detections by the sensor

# %%
# Run the Kalman filters
# ^^^^^^^^^^^^^^^^^^^^^^
#
# First create `ntruths` priors which estimate the targets’ initial states, one for each target. In this example
# each prior is offset by 0.5 in the y direction meaning the position of the track is initially not very accurate. The
# velocity is also systematically offset by +0.5 in both the x and y directions.

from stonesoup.types.state import GaussianState

priors = []
xdirection = 1.2
ydirection = 1.2
for j in range(0, ntruths):
    priors.append(GaussianState([[0], [xdirection], [yps[j]+0.1], [ydirection]],
                                np.diag([0.5, 0.5, 0.5, 0.5]+np.random.normal(0,5e-4,4)),
                                timestamp=start_time))
    xdirection *= -1
    if j % 2 == 0:
        ydirection *= -1

# %%
# Initialise the tracks by creating an empty list and appending the priors generated. This needs to be done separately
# for both sensor manager methods as they will generate different sets of tracks.

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
# ^^^^^^^^^^^^^^^
#
# A reward function is used to quantify the benefit of sensors taking a particular action or set of actions.
# This can be crafted specifically for an example in order to achieve a particular objective. The function used in
# this example is quite generic but could be substituted for any callable function which returns a numeric
# value that the sensor manager can maximise.
#
# The :class:`~.UncertaintyRewardFunction` calculates the uncertainty reduction by computing the difference between the
# covariance matrix norms of the
# prediction, and the posterior assuming a predicted measurement corresponding to that prediction.

from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=5)

from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
data_associator = GNNWith2DAssignment(hypothesiser)

from stonesoup.sensormanager.reward import UncertaintyRewardFunction
reward_function = UncertaintyRewardFunction(predictor=predictor, updater=updater)

from ordered_set import OrderedSet


# %%
# Design Environment
# ^^^^^^^^^^^^^^^^^^
#
# An environment is needed for the RL agent to learn in. There are resources online for how to design these [#].
#
# For this tutorial, a pre-designed environment has been created for you to go through.
# In this example, the action space is equal to the number of targets in the simulation, so at each time step, the
# sensor can look at one target.
# The :class:`~.UncertaintyRewardFunction` to calculate the reward obtained for each step in the environment.
# The trace of the covariances for each object is used as the observation for the agent to learn from.

from stonesoup.sensormanager.reinforcement import BaseEnvironment
import numpy as np
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import utils
from stonesoup.sensor.action.dwell_action import DwellActionsGenerator
from stonesoup.functions import mod_bearing
import copy


class StoneSoupEnv(BaseEnvironment):

    def __init__(self):
        super().__init__()
        # Action size is number of targets
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=ntruths-1, name='action')
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
                x_target = target.state.state_vector[0]-self.sensor.position[0]
                y_target = target.state.state_vector[2]-self.sensor.position[1]
                bearing_target = mod_bearing(np.arctan2(y_target, x_target))

            uncertainty.append(np.trace(target.covar))

        current_timestep = self.start_time + timedelta(seconds=self.current_step)
        next_timestep = self.start_time + timedelta(seconds=self.current_step+1)

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

        if self.current_step >= self.max_episode_length-1:
            self._episode_ended = True
            return ts.termination(observation, reward)
        else:
            return ts.transition(observation, reward=reward, discount=1.0)

    @staticmethod
    def generate_action(action):
        """This method is used to convert a tf-agents action into a Stone Soup action"""
        for i, target in enumerate(tracksA):
            if i == action:
                x_target = target.state.state_vector[0]-sensorA.position[0]
                y_target = target.state.state_vector[2]-sensorA.position[1]
                action_bearing = mod_bearing(np.arctan2(y_target, x_target))

        action_generators = DwellActionsGenerator(sensorA,
                                                  attribute='dwell_centre',
                                                  start_time=sensorA.timestamp,
                                                  end_time=sensorA.timestamp+timedelta(seconds=1))

        current_action = [action_generators.action_from_value(action_bearing)]
        return current_action

# Validate the environment to ensure that the environment returns the expected specs
train_env = StoneSoupEnv()
utils.validate_py_environment(train_env, episodes=5)


# %%
# Create Sensor Managers
# ^^^^^^^^^^^^^^^^^^^^^^
#
# We initiate our reinforcement learning sensor manager with the environment we have designed

from stonesoup.sensormanager import ReinforcementSensorManager
reinforcementsensormanager = ReinforcementSensorManager({sensorA}, env=StoneSoupEnv())

from stonesoup.sensormanager import BruteForceSensorManager
bruteforcesensormanager = BruteForceSensorManager({sensorB}, reward_function=reward_function)


# %%
# Train RL agent
# ^^^^^^^^^^^^^^
#
# To generate a policy, we need to train the reinforcement learning agent using the environment we created above.
# Some hyperparameters are created that the agent uses to train with.
#
# To train the agent, the hyperparameters are passed to the train method in the :class:`~.ReinforcementSensorManager`.

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
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# To be able to exploit the policy generated by the reinforcement sensor manager, it  must be passed appropriate
# 'timesteps'.
# These are distinct from the timesteps in Stonesoup, and is of the form time_step_spec from tf-agents.

from itertools import chain

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
        x_target = target.state.state_vector[0]-sensorA.position[0]
        y_target = target.state.state_vector[2]-sensorA.position[1]
        bearing_target = mod_bearing(np.arctan2(y_target, x_target))
        uncertainty.append(np.trace(target.covar))

        # observation.append(np.degrees(bearing_target))
        observation.append(np.trace(target.covar))

    observation = np.array(uncertainty, dtype=np.float32)
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
        else:  # When data associator says no detections are good enough, we'll keep the prediction
            track.append(hypothesis.prediction)

    # Propagate environment
    action_step = reinforcementsensormanager.agent.policy.action(tf_timestep)
    tf_timestep = reinforcementsensormanager.test_env.step(action_step.action)

# %%
# Plot ground truths, tracks and uncertainty ellipses for each target.

plotterA = Plotterly()
plotterA.plot_sensors(sensorA)
plotterA.plot_ground_truths(truths_set, [0, 2])
plotterA.plot_tracks(set(tracksA), [0, 2], uncertainty=True)
plotterA.fig


# %%
# Run brute force sensor manager
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

plotterB = Plotterly()
plotterB.plot_sensors(sensorA)
plotterB.plot_ground_truths(truths_set, [0, 2])
plotterB.plot_tracks(set(tracksB), [0, 2], uncertainty=True)
plotterB.fig

# %%
# With a properly trained policy, the :class:`~.ReinforcementSensorManager` performs almost as well as the
# :class:`~.BruteForceSensorManager`.


# %%
# Metrics
# ^^^^^^^
#
# Metrics can be used to compare how well different sensor management techniques are working.
# Full explanations of the OSPA
# and SIAP metrics can be found in the Metrics Example.

from stonesoup.metricgenerator.ospametric import OSPAMetric
ospa_generator = OSPAMetric(c=40, p=1)

from stonesoup.metricgenerator.tracktotruthmetrics import SIAPMetrics
from stonesoup.measures import Euclidean
siap_generator = SIAPMetrics(position_measure=Euclidean((0, 2)),
                             velocity_measure=Euclidean((1, 3)))


# %%
# The SIAP metrics require an associator to associate tracks to ground truths. This is done using the
# :class:`~.TrackToTruth` associator with an association threshold of 30.
#

from stonesoup.dataassociator.tracktotrack import TrackToTruth
associator = TrackToTruth(association_threshold=30)

# %%
# The OSPA and SIAP metrics don't take the uncertainty of the track into account. The initial plots of the
# tracks and ground truths show by plotting the uncertainty ellipses that there is generally less uncertainty
# in the tracks generated by the :class:`~.BruteForceSensorManager`.
#
# To capture this we can use an uncertainty metric to look at the sum of covariance matrix norms at
# each time step. This gives a representation of the overall uncertainty of the tracking over time.

from stonesoup.metricgenerator.uncertaintymetric import SumofCovarianceNormsMetric
uncertainty_generator = SumofCovarianceNormsMetric()


# %%
# A metric manager is used for the generation of metrics on multiple :class:`~.GroundTruthPath` and
# :class:`~.Track` objects. This takes in the metric generators, as well as the associator required for the
# SIAP metrics.
#
# We must use a different metric manager for each sensor management method. This is because each sensor manager
# generates different track data which is then used in the metric manager.

from stonesoup.metricgenerator.manager import SimpleManager

metric_managerA = SimpleManager([ospa_generator, siap_generator, uncertainty_generator],
                                associator=associator)

metric_managerB = SimpleManager([ospa_generator, siap_generator, uncertainty_generator],
                                associator=associator)


# %%
# For each time step, data is added to the metric manager on truths and tracks. The metrics themselves can then be
# generated from the metric manager.

metric_managerA.add_data(truths, tracksA)
metric_managerB.add_data(truths, tracksB)

metricsA = metric_managerA.generate_metrics()
metricsB = metric_managerB.generate_metrics()


# %%
# OSPA metric
# ^^^^^^^^^^^
#
# First we look at the OSPA metric. This is plotted over time for each sensor manager method.

import matplotlib.pyplot as plt

ospa_metricA = metricsA['OSPA distances']
ospa_metricB = metricsB['OSPA distances']

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot([i.timestamp for i in ospa_metricA.value],
        [i.value for i in ospa_metricA.value],
        label='ReinforcementSensorManager')
ax.plot([i.timestamp for i in ospa_metricB.value],
        [i.value for i in ospa_metricB.value],
        label='BruteForceSensorManager')
ax.set_ylabel("OSPA distance")
ax.set_xlabel("Time")
ax.legend()


# %%
# The :class:`~.BruteForceSensorManager` generally results in a smaller OSPA distance
# than the random observations of the :class:`~.ReinforcementSensorManager`, reflecting the better tracking performance
# seen in the tracking plots.

# %%
# SIAP metrics
# ^^^^^^^^^^^^
#
# Next we look at SIAP metrics. We are only interested in the positional accuracy (PA) and velocity accuracy (VA).
# These metrics can be plotted to show how they change over time.

fig, axes = plt.subplots(2)

times = metric_managerA.list_timestamps()

pa_metricA = metricsA['SIAP Position Accuracy at times']
va_metricA = metricsA['SIAP Velocity Accuracy at times']

pa_metricB = metricsB['SIAP Position Accuracy at times']
va_metricB = metricsB['SIAP Velocity Accuracy at times']

axes[0].set(title='Positional Accuracy', xlabel='Time', ylabel='PA')
axes[0].plot(times, [metric.value for metric in pa_metricA.value],
             label='ReinforcementSensorManager')
axes[0].plot(times, [metric.value for metric in pa_metricB.value],
             label='BruteForceSensorManager')
axes[0].legend()

axes[1].set(title='Velocity Accuracy', xlabel='Time', ylabel='VA')
axes[1].plot(times, [metric.value for metric in va_metricA.value],
             label='ReinforcementSensorManager')
axes[1].plot(times, [metric.value for metric in va_metricB.value],
             label='BruteForceSensorManager')
axes[1].legend()

# %%
# Similar to the OSPA distances the :class:`~.BruteForceSensorManager`
# generally results in both a better positional accuracy and velocity accuracy than the random observations
# of the :class:`~.RandomSensorManager`.
#

# %%
# Uncertainty metric
# ^^^^^^^^^^^^^^^^^^
#
# Finally we look at the uncertainty metric which computes the sum of covariance matrix norms of each state at each
# time step. This is plotted over time for each sensor manager method.

uncertainty_metricA = metricsA['Sum of Covariance Norms Metric']
uncertainty_metricB = metricsB['Sum of Covariance Norms Metric']

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot([i.timestamp for i in uncertainty_metricA.value],
        [i.value for i in uncertainty_metricA.value],
        label='ReinforcementSensorManager')
ax.plot([i.timestamp for i in uncertainty_metricB.value],
        [i.value for i in uncertainty_metricB.value],
        label='BruteForceSensorManager')
ax.set_ylabel("Sum of covariance matrix norms")
ax.set_xlabel("Time")
ax.legend()

# %%
# This metric shows that the uncertainty in the tracks generated by the :class:`~.RandomSensorManager` is much greater
# than for those generated by the :class:`~.BruteForceSensorManager`. This is also reflected by the uncertainty ellipses
# in the initial plots of tracks and truths.
#
#

# %%
# References
# ^^^^^^^^^^
#
# .. [#] *https://github.com/tensorflow/agents/blob/master/docs/tutorials/2_environments_tutorial.ipynb*

# sphinx_gallery_thumbnail_number = 2