#!/usr/bin/env python

"""
Actionable Platforms
====================
This example demonstrates the management of actionable platforms in Stone Soup.
"""

# %%
# Platforms in Stone Soup
# -----------------------
# In Stone Soup, instances of the :class:`~.Platform` class are objects to which one or more
# sensors can be mounted. They provide a means of controlling the position of mounted sensors.
# 
# All platforms in Stone Soup have a movement controller belonging to the class
# :class:`~.Movable`, which determines if and how the platform can move. The default platforms
# and corresponding movement controllers currently implemented in Stone Soup are:
#
# * :class:`~.FixedPlatform`, which has a default movable class of :class:`~.FixedMovable`. The
#   position of these platforms can be manually defined, but otherwise remains fixed.
# * :class:`~.MovingPlatform`, which has a default movable class of :class:`~.MovingMovable`. The
#   position of these platforms is not fixed, but changes according to a predefined
#   :class:`~.TransitionModel`.
# * :class:`~.MultiTransitionMovingPlatform`, which has a default movable class of
#   :class:`~.MultiTransitionMovable`. The same as for :class:`~.MovingPlatform`, but movement is
#   defined by multiple transition models.
#
# Actionable Platforms in Stone Soup
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Actionable platforms work slightly differently to these other platforms. They can be
# instantiated using the previously mentioned :class:`~.FixedPlatform` - what makes them
# 'actionable' is the use of a movement controller with an :class:`~.ActionGenerator`. The
# :class:`~.ActionGenerator` produces objects of class :class:`~.Action` that can be given to a
# :class:`~.SensorManager` to be optimised and acted upon at each timestep.
# 
# Currently, actionable platforms in Stone Soup can be created from a :class:`~.FixedPlatform`
# using the :class:`~.NStepDirectionalGridMovable` movement controller, which allows movement
# across a grid-based action space according to a given step size and number of steps.
# Additional actionable movement controllers will likely be added in the future.
# 
# This example demonstrates the basic usage of actionable platforms. A scenario is created in
# which an :class:`~.NStepDirectionalGridMovable` platform mounted with a
# :class:`~.RadarRotatingBearingRange` sensor is used to track a single moving target that would
# otherwise move out of the sensor's range.

# %%
# Setting Up the Scenario
# -----------------------
# We begin by setting up the scenario. We generate a ground truth to simulate the linear movement
# of a target with a small amount of noise.


import numpy as np
from datetime import datetime, timedelta

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

np.random.seed(1990)

start_time = datetime.now().replace(microsecond=0)

transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.1), ConstantVelocity(0.1)])

truth = GroundTruthPath([GroundTruthState([-450, 5, 450, -5], timestamp=start_time)])
duration = 120
timesteps = [start_time]

for k in range(1, duration):
    timesteps.append(start_time+timedelta(seconds=k))
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=timesteps[k]))


# %%
# Visualising the ground truth.


from stonesoup.plotter import AnimatedPlotterly
plotter = AnimatedPlotterly(timesteps, tail_length=0.3)
plotter.plot_ground_truths(truth, [0, 2])
plotter.fig


# %%
# Creating the Actionable Platform
# --------------------------------
# Next we create the actionable platform itself. To do this we create a :class:`~.FixedPlatform`,
# but change the movement controller to a :class:`~.NStepDirectionalGridMovable`. The platform
# starts alongside the target. We define the platforms movement such that it is capable of moving
# up to two steps at each timestep. As the resolution is set to 1 and the step size 6.25, each
# step corresponds to 6.25 grid cells, each (x=1, y=1) in size. These restrictions will be
# reflected in the list of :class:`~.Action` objects created by the movement controller's
# :class:`~.ActionGenerator`.
# 
# We add a :class:`~.RadarRotatingBearingRange` radar to this platform, which has a field of
# view of 30 degrees, a range of 100 grid cells, and is capable of rotating its dwell centre by
# 180 degrees each timestep.


from stonesoup.platform import FixedPlatform
from stonesoup.movable.grid import NStepDirectionalGridMovable
from stonesoup.sensor.radar.radar import RadarRotatingBearingRange
from stonesoup.types.angle import Angle
from stonesoup.types.state import State, StateVector

sensor = RadarRotatingBearingRange(
    position_mapping=(0, 2),
    noise_covar=np.array([[np.radians(1)**2, 0],
                          [0, 1**2]]),
    ndim_state=4,
    rpm=30,
    fov_angle=np.radians(30),
    dwell_centre=StateVector([np.radians(90)]),
    max_range=100,
    resolution=Angle(np.radians(30)))

platform = FixedPlatform(
    movement_controller=NStepDirectionalGridMovable(states=[State([[-500], [500]],
                                                                  timestamp=start_time)],
                                                    position_mapping=(0, 1),
                                                    resolution=1,
                                                    n_steps=2,
                                                    step_size=6.25,  # 6.25 seems to match target
                                                    action_mapping=(0, 1)),
    sensors=[sensor])


# %%
# Creating a Predictor and Updater
# --------------------------------
# Next we create some standard Stone Soup components required for tracking: a predictor,
# which in this case creates an initial estimate of the target at each timestep according to a
# linear transition model, and an updater, which updates our initial estimate based on our sensor's
# measurements.
# 
# As we are working with a particle filter, we also include a resampler, which occasionally
# regenerates particles according to their weight/likelihood. The particles with higher
# weights are preserved and replicated. A drawback of this approach is particle
# impoverishment, whereby repeatedly resampling higher weight particles results in a lower
# diversity of samples. We therefore include a regulariser to mitigate sample impoverishment by
# slightly moving resampled particles according to a Gaussian kernel if an acceptance
# probability is met.


from stonesoup.resampler.particle import ESSResampler
from stonesoup.regulariser.particle import MCMCRegulariser
from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.updater.particle import ParticleUpdater

resampler = ESSResampler()
regulariser = MCMCRegulariser()
predictor = ParticlePredictor(CombinedLinearGaussianTransitionModel([ConstantVelocity(0.1),
                                                                     ConstantVelocity(0.1)]))
updater = ParticleUpdater(sensor.measurement_model,
                          resampler=resampler,
                          regulariser=regulariser)


# %%
# Creating a Sensor Manager
# -------------------------
# Now we create a sensor manager, giving it our sensor and platform, and a reward function. In
# this case the :class:`~.ExpectedKLDivergence` reward function is used, which chooses actions
# based on the information gained by taking that action.


from stonesoup.sensormanager.reward import ExpectedKLDivergence
from stonesoup.sensormanager import BruteForceSensorManager

reward_updater = ParticleUpdater(measurement_model=None)
reward_func = ExpectedKLDivergence(predictor=predictor, updater=reward_updater)

sensormanager = BruteForceSensorManager(sensors={sensor},
                                        platforms={platform},
                                        reward_function=reward_func)


# %%
# Creating a Track
# ----------------
# We create a prior and use this to initialise a track. The prior consists of 2000 particles for
# each component of our state vector, normally distributed around the ground truth, and each with
# an equal initial weight.


from stonesoup.types.state import StateVectors, ParticleState
from stonesoup.types.track import Track

nparts = 2000
prior = ParticleState(StateVectors([np.random.normal(truth[0].state_vector[0], 10, nparts),
                                    np.random.normal(truth[0].state_vector[1], 1, nparts),
                                    np.random.normal(truth[0].state_vector[2], 10, nparts),
                                    np.random.normal(truth[0].state_vector[3], 1, nparts)]),
                      weight=np.array([1/nparts]*nparts),
                      timestamp=start_time)
prior.parent = prior
track = Track([prior])


# %%
# Creating a Hypothesiser and Data Associator
# -------------------------------------------
# The final components we need before we can begin the tracking loop are a hypothesiser and data
# associator, which, in this case, pair detections with predictions based on their distance.


from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=5)

from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
data_associator = GNNWith2DAssignment(hypothesiser)


# %%
# Running the Tracking Loop
# -------------------------
# Finally we can run the tracking loop.
# 
# At each timestep we use our sensor manager to generate the optimal actions for our sensor and
# platform. When we call the sensor manager's :meth:`~.SensorManager.choose_actions` method, a few
# things are happening in the background. For each actionable we control (including our actionable
# platform), the actionable's :class:`~.ActionGenerator` is used to retrieve all possible
# actions that our sensor or platform could take at that timestep. What happens next will depend
# on the kind of sensor manager used. In our case we chose a :class:`~.BruteForceSensorManager`,
# so every combination of actions between sensor and platform are considered, and each one is
# evaluated by our reward function. The combination of actions resulting in the highest reward
# will be returned, and we then move our actionables accordingly.
# 
# After moving our actionable objects, we take a measurement with our sensor, and update our track
# depending on what we see.


from stonesoup.sensor.sensor import Sensor
import copy
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
                measurement = actionable.measure({truth[timestep]},
                                                 noise=True)
    measurements.append(measurement)
    hypotheses = data_associator.associate({track},
                                           measurement,
                                           timestep)
    
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
# As we can see, the actionable platform is able to follow the target as it moves across the
# action space, keeping it within the sensor's range.


import plotly.graph_objects as go
from stonesoup.functions import pol2cart

plotter.plot_tracks(track, mapping=(0, 2))
plotter.plot_measurements(measurements, mapping=(0, 2))

sensor_set = {sensor}


def plot_sensor_fov(fig_, sensor_set, sensor_history):
    # Plot sensor field of view
    trace_base = len(fig_.data)
    for _ in sensor_set:
        fig_.add_trace(go.Scatter(mode='lines',
                                  line=go.scatter.Line(color='black',
                                                       dash='dash')))

    for frame in fig_.frames:
        traces_ = list(frame.traces)
        data_ = list(frame.data)

        timestring = frame.name
        timestamp = datetime.strptime(timestring, "%Y-%m-%d %H:%M:%S")

        for n_, sensor_ in enumerate(sensor_set):
            x = [0, 0]
            y = [0, 0]

            if timestamp in sensor_history:
                sensor_ = sensor_history[timestamp][sensor_]
                for i, fov_side in enumerate((-1, 1)):
                    range_ = min(getattr(sensor_, 'max_range', np.inf), 100)
                    x[i], y[i] = pol2cart(range_,
                                          sensor_.dwell_centre[0, 0]
                                          + sensor_.fov_angle / 2 * fov_side) \
                        + sensor_.position[[0, 1], 0]
            else:
                continue

            data_.append(go.Scatter(x=[x[0], sensor_.position[0], x[1]],
                                    y=[y[0], sensor_.position[1], y[1]],
                                    mode="lines",
                                    line=go.scatter.Line(color='black',
                                                         dash='dash'),
                                    showlegend=False))
            traces_.append(trace_base + n_)

        frame.traces = traces_
        frame.data = data_


plot_sensor_fov(plotter.fig, sensor_set, sensor_history)
plotter.fig


# %%
# Summary
# -------
# This was a simple example demonstrating how an actionable platform can be used to track a moving
# target.
#
# There are many other scenarios to which actionable platforms could be applied, and the number of
# possible behaviours and applications will only increase as additional classes are introduced.
# Stone Soup also makes it easy to implement additional functionality on your own, for example by
# experimenting with custom reward functions and/or movement controllers.
#
# To see the latest developments for actionable platforms, you can refer to the Stone Soup
# docs for :doc:`Movables <../../stonesoup.movable>` and
# their :doc:`Actions <../../stonesoup.movable.action>`.
