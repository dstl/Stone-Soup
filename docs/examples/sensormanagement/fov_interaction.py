#!/usr/bin/env python

"""
===========================================
FOV based Reward Function Sensor Management
===========================================
"""

# %%
# This notebook is designed to showcase the use of Stone Soup for simulating "interesting"
# behaviour between a sensor platform and a target.
# The target is travelling through the scenario with a known FOV, the sensor platform has to
# maintain sensing of the target (i.e., keep the target in the sensor platform's FOV but not
# enter the target's FOV otherwise it would be detected).
#
# Background and notation
# -----------------------
#
# *[Some more info on something useful here]*.

# %%
# Setting Up the Scenario
# -----------------------
# We generate a ground truth of a target following a constant velocity with an amount of
# noise that makes the truth interesting.

import numpy as np
from datetime import datetime, timedelta

from stonesoup.models.transition.linear import (CombinedLinearGaussianTransitionModel,
                                                ConstantVelocity)
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

np.random.seed(1990)

start_time = datetime.now().replace(second=0, microsecond=0)

transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(10), ConstantVelocity(10)])

truth = GroundTruthPath([GroundTruthState([-450, 5, 450, -5], timestamp=start_time)])
duration = 120
timesteps = [start_time]

for k in range(1, duration):
    timesteps.append(start_time+timedelta(seconds=k))
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=timesteps[k]))

# %%
# Visualising the ground truth
# ----------------------------

from stonesoup.plotter import AnimatedPlotterly
plotter = AnimatedPlotterly(timesteps, tail_length=0.3)
plotter.plot_ground_truths(truth, [0, 2])
plotter.fig

# %%
# See, interesting.

# %%
# Creating the Actionable Platform
# --------------------------------
#
# *Next we create the actionable platform itself*. We use a :class:`~.MovingPlatform` (because it
# would be silly to use a :class:`~.FixedPlatform` for a moving platform) with a
# :class:`~.MaxSpeedActionableMovable` movement controller.
# The platform starts alongside the target (as this is not a search problem (yet)).
# We define the platform's movement such that it is capable of moving up to a maximum speed in a
# number of directions (8 cardinal directions (cardinal and ordinal)).
#
# We add a :class:`~.RadarRotatingBearingRange` radar to this platform, which has a field of
# view of 360 degrees, a range of 200, and is capable of rotating its dwell centre by
# 90 degrees each timestep. Similarly a :class:`~.RadarRotatingBearingRange` radar is added to the
# target platform (primarily for use in plotting the target FOV) with a field of view of 360
# degrees and a range of 100.

from stonesoup.platform import MovingPlatform
from stonesoup.movable.max_speed import MaxSpeedActionableMovable
from stonesoup.sensor.radar.radar import RadarRotatingBearingRange
from stonesoup.types.angle import Angle
from stonesoup.types.state import State, StateVector

sensor = RadarRotatingBearingRange(position_mapping=(0, 2),
                                   noise_covar=np.array([[np.radians(5)**2, 0], [0, 10**2]]),
                                   ndim_state=4,
                                   rpm=30,
                                   fov_angle=np.radians(360),
                                   dwell_centre=StateVector([np.radians(90)]),
                                   max_range=200,
                                   resolution=Angle(np.radians(180)))

target_sensor = RadarRotatingBearingRange(position_mapping=(0, 2),
                                          noise_covar=np.array([[np.radians(1)**2, 0], [0, 1**2]]),
                                          ndim_state=4,
                                          rpm=30,
                                          fov_angle=np.radians(360),
                                          dwell_centre=StateVector([np.radians(90)]),
                                          max_range=100,
                                          resolution=Angle(np.radians(180)))

platform = MovingPlatform(movement_controller=MaxSpeedActionableMovable(
    states=[State([[-500], [500]], timestamp=start_time)],
    position_mapping=(0, 1),
    action_mapping=(0, 1),
    resolution=10,
    angle_resolution=np.pi/2,
    max_speed=300),
    sensors=[sensor])

# %%
# Creating a Predictor and Updater
# --------------------------------
#
# Next we need some tracking components: a predictor and an updater.

from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

from stonesoup.updater.kalman import ExtendedKalmanUpdater
updater = ExtendedKalmanUpdater(measurement_model=None)

# %%
# Creating a Sensor Manager
# -------------------------
#
# Now we create a sensor manager, providing it with the sensor, the sensor platform,
# and a reward function.
# In this case the :class:`~.FOVInteractionRewardFunction` and the
# :class:`~.UncertaintyRewardFunction` reward functions are combined using an
# :class:`~.MultiplicativeRewardFunction`.

import copy
from stonesoup.sensor.sensor import Sensor
from stonesoup.types.track import Track
from stonesoup.sensormanager import BruteForceSensorManager
from stonesoup.sensormanager.reward import (
    FOVInteractionRewardFunction, UncertaintyRewardFunction, MultiplicativeRewardFunction)

reward_func_A = FOVInteractionRewardFunction(
    predictor, updater, sensor_fov_radius=sensor.max_range,
    target_fov_radius=target_sensor.max_range)

reward_func_B = UncertaintyRewardFunction(predictor, updater)

reward_func = MultiplicativeRewardFunction([reward_func_A, reward_func_B])

sensormanager = BruteForceSensorManager(sensors={sensor},
                                        platforms={platform},
                                        reward_function=reward_func)

# %%
# Creating a Track
# ----------------

from stonesoup.types.state import GaussianState

prior = GaussianState(truth[0].state_vector,
                      covar=np.diag([0.5, 0.5, 0.5, 0.5] + np.random.normal(0, 5e-4, 4)),
                      timestamp=start_time)

track = Track([prior])

# %%
# Creating a Hypothesiser and Data Associator
# -------------------------------------------
# The final tracking components required are the hypothesiser and data associator.

from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis

hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=5)

from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
data_associator = GNNWith2DAssignment(hypothesiser)

# %%
# Running the Tracking Loop
# -------------------------
#
# At each timestep the sensor manager generates the optimal actions for our sensor and
# platform. The sensor manager's :meth:`~.SensorManager.choose_actions` method is called.

from collections import defaultdict
sensor_history = defaultdict(dict)

measurements = []
for timestep in timesteps[1:]:
    chosen_actions = sensormanager.choose_actions({track}, timestep)
    for chosen_action in chosen_actions:
        for actionable, actions in chosen_action.items():
            actionable.add_actions(actions)
            actionable.act(timestep)
            if isinstance(actionable, Sensor):
                measurement = actionable.measure({truth[timestep]}, noise=True)
    measurements.append(measurement)
    hypotheses = data_associator.associate({track}, measurement, timestep)

    sensor_history[timestep][sensor] = copy.deepcopy(sensor)
    hypothesis = hypotheses[track]

    if hypothesis.measurement:
        post = updater.update(hypothesis)
        track.append(post)
    else:
        track.append(hypothesis.prediction)

# %%
# Plotting
# --------
#
# The FOV-based reward function in an actionable platform is able to follow the target as it moves
# across the action space, keeping it within the sensor's range but outside the FOV of the target.

from stonesoup.plotter import plot_sensor_fov

from stonesoup.platform.base import PathBasedPlatform

plotter = AnimatedPlotterly(timesteps, tail_length=0.3)
plotter.plot_ground_truths(truth, [0, 2])
plotter.plot_tracks(track, mapping=(0, 2))

target_platform = PathBasedPlatform(path=track, sensors=[target_sensor], position_mapping=[0, 2])
target_sensor_history = defaultdict(dict)

for timestep in timesteps[1:]:
    target_platform.move(timestep)
    target_sensor_history[timestep][target_sensor] = copy.deepcopy(target_sensor)
target_sensor_set = {target_sensor}
sensor_set = {sensor}

plot_sensor_fov(plotter.fig, sensor_set, sensor_history)
plot_sensor_fov(plotter.fig, target_sensor_set, target_sensor_history, label="Target FOV",
                color="red")
plotter.fig

# %%
# Summary
# -------
#
# Look it does what we said it would do!
# This was a simple example demonstrating how an actionable platform can use a composite reward
# function to create interesting new interactions.
