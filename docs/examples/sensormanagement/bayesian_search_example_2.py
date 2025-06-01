#!/usr/bin/env python

"""
Bayesian Search with a Moving Platform
======================================
This example builds on the introduction to Bayesian search given in the first example, found
:doc:`here <bayesian_search_example_1>`.

The paper accompanying this work, '*Open Source Tools for Bayesian Search*' `[1] <#references>`_,
can be found `here <https://doi.org/10.1117/12.3012763>`_.
"""

# %%
# Introduction
# ------------
# In the first example we saw how Bayesian search could be implemented in Stone Soup using a
# single fixed-position rotating sensor. We will now build on this and apply Bayesian search to a
# slightly more complex scenario, which involves controlling a limited-range sensor on a moving
# platform. We will see how Bayesian search can be used to move the platform to the areas most
# likely to contain the target.
#
# As in the first example, we will be using the class :class:`~.ParticleState` to build our search
# space, but this time we will be creating a 2D search grid.

# %%
# Setup
# -----
# We begin with some generic imports, and then set the start time...


import copy
from datetime import datetime, timedelta
import numpy as np
import random

# use fixed seed for random number generators
np.random.seed(123)
random.seed(123)

start_time = datetime(2025, 5, 9, 14, 15)
timesteps = [start_time + timedelta(seconds=k) for k in range(0, 12)]


# %%
# Creating the Platform
# ~~~~~~~~~~~~~~~~~~~~~
# We create the moving platform to which our sensor will be mounted. Rather than defining the
# movement pattern of the platform in advance, we want the platform to automatically choose an
# optimal movement at each timestep. We can use a platform with an
# :class:`~.NStepDirectionalGridMovable` to achieve this.
#
# Platforms with this movement controller can generate a list of actions at each timestep which
# can then be given to a :class:`~.SensorManager`. The sensor manager chooses the action(s) with
# the greatest reward for the current timestep. We define the platform's initial position, and
# limit its movement to a step of three grid cells along either of the grid's axes.


from stonesoup.movable.grid import NStepDirectionalGridMovable
from stonesoup.platform import Platform
from stonesoup.types.state import State, StateVector

initial_loc = StateVector([[4.], [4.]])
initial_state = State(initial_loc, start_time)

platform = Platform(
    movement_controller=NStepDirectionalGridMovable(states=[initial_state],
                                                    position_mapping=(0, 1),
                                                    n_steps=1,
                                                    step_size=3,
                                                    action_mapping=(0, 1),
                                                    ))


# %%
# Creating the Sensor
# ~~~~~~~~~~~~~~~~~~~
# Next we make a sensor which will be mounted to the platform. In this case we use a
# :class:`~.RadarRotatingBearingRange` sensor. The sensor has a 360 degree field of view, with a
# range of 1.6 grid cells.
#
# As before, we set the probability of detection to 0.9.


from stonesoup.sensor.radar.radar import RadarRotatingBearingRange

mounted_sensor = RadarRotatingBearingRange(
    position_mapping=(0, 2),
    noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                          [0, 1 ** 2]]),
    ndim_state=4,
    rpm=0,
    fov_angle=np.radians(360),
    dwell_centre=StateVector([0.0]),
    max_range=1.6,
    resolution=np.radians(360))

mounted_sensor.timestamp = start_time
prob_det = 0.9

platform.add_sensor(mounted_sensor)


# %%
# Creating the Search Grid
# ~~~~~~~~~~~~~~~~~~~~~~~~
# We will now create the environment with which our sensor and platform will interact.
#
# To begin, we create a 15x15 grid. As in the previous tutorial, we will do this using
# :class:`~.ParticleState` objects, whose properties can be used to represent a position and
# probability of target presence.


# creating the positions of the particles that will form our search grid
ymin = 0
xmin = 0
ymax = 14
xmax = 14
nx = 15
ny = 15

xarray = np.linspace(xmin, xmax, nx)
yarray = np.linspace(ymin, ymax, ny)
x, y = np.meshgrid(xarray, yarray)

x_pos = x.flatten()
y_pos = y.flatten()

# %%
# In setting the particle weights here, we are defining our initial probability distribution. In
# this case, the initial distribution will be Gaussian, centred around (x=7, y=8).


from stonesoup.types.state import ParticleState
from stonesoup.types.array import StateVectors
from scipy.stats import multivariate_normal

# parameterise our prior probability distribution
mu = np.array([7, 8])  # mean(x, y)
sigma = np.array([[10, 0], [0, 6]])  # covariance matrix [[var(x), covar(xy)], [covar(xy), var(y)]]

prior = multivariate_normal.pdf(np.vstack((x_pos, y_pos)).T, mu, sigma)

prior = prior / sum(prior)
prior_weights = prior.reshape(ny, nx).flatten()

vel = [0] * len(x_pos)
state_vectors = StateVectors(np.array([x_pos, vel, y_pos, vel]))

prior = ParticleState(state_vector=state_vectors,
                      weight=prior_weights,
                      timestamp=start_time)


# %%
# Creating a Custom Reward Function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# At each timestep, we want our platform to move to the area with the greatest probability of
# containing the target. We create a custom reward function with identical functionality to that
# in the first example.


from stonesoup.sensor.sensor import Sensor


def sumofweightsreward(config, undetectmap, timestamp):
    """
    Takes a configuration, which is a mapping of actionables (sensors, platforms, etc.) to their
    possible actions. For each item in the mapping, creates a copy of the actionable which is used
    to simulate the action (e.g., platform movement). Then takes a measurement with the simulated
    sensors to count the number of particles that would be in the sensor's FOV if the action
    were taken. Returns sum of particles within the sensor's FOV.
    """
    predicted_sensors = set()
    memo = {}
    for actionable, actions in config.items():
        predicted_actionable = copy.deepcopy(actionable, memo)
        predicted_actionable.add_actions(actions)
        predicted_actionable.act(timestamp)

        if isinstance(predicted_actionable, Sensor):
            predicted_sensors.add(predicted_actionable)

    running_tot = 0
    for sensor in predicted_sensors:

        # assume no detection
        for j, particle in enumerate(undetectmap.state_vector):
            pstate = ParticleState(particle)
            weight = undetectmap.weight[j]
            # add to running total if in FOV
            if sensor.is_detectable(pstate):
                running_tot += weight

    return float(running_tot)


# %%
# Creating a Sensor Manager
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Finally, we create a :class:`~.BruteForceSensorManager`, which will evaluate every possible
# movement of our platform using the custom reward function and return the one with the highest
# reward value.


from stonesoup.sensormanager import BruteForceSensorManager

sensormanager = BruteForceSensorManager(platforms={platform},
                                        sensors={mounted_sensor},
                                        reward_function=sumofweightsreward)


# %%
# Running the Scenario
# --------------------
# Aside from the addition of the platform, the search loop here is very similar to that of the
# previous example. At each timestep we move the sensor and platform, make an observation and
# update the particle weights accordingly.


sensor_history = []
sensor_history.append(copy.deepcopy(mounted_sensor))  #  initial state
search_cell_info = [prior]
current_state = prior
prob_found_list = [0]

for timestep in timesteps[1:]:

    # update the search cell states with a new timestamp
    next_state = ParticleState(current_state.state_vector,
                               weight=current_state.weight,
                               timestamp=timestep)

    # get highest rewarded action from sensor manager and move sensor and platform
    chosen_actions = sensormanager.choose_actions(next_state, timestep)
    for chosen_action in chosen_actions:
        for actionable, actions in chosen_action.items():
            actionable.add_actions(actions)
    mounted_sensor.act(timestep)
    platform.move(timestep)

    # updating probability distribution
    weight_in_view = 0

    # for each particle/search cell
    for j, particle in enumerate(next_state.state_vector):

        pstate = ParticleState(particle)
        weight = next_state.weight[j]

        # update particles according to eq. 4 and 5 of paper
        if mounted_sensor.is_detectable(pstate):
            weight_in_view += weight
            # all other particles adjusted according to probability of not finding target in
            # cell j (eq. 5)
            next_state.weight = next_state.weight / (1 - weight * prob_det)
            # then correct the probability for cell j (eq. 4)
            next_state.weight[j] = weight * (1 - prob_det) / (1 - weight * prob_det)

    # updated search cell states become prior for next step
    current_state = next_state

    # store info for plotting
    sensor_history.append(copy.deepcopy(mounted_sensor))
    search_cell_info.append(current_state)

    # update cumulative probability of having found target by now (eq. 6 of paper)
    prob_found = prob_found_list[-1]
    prob_found_list.append(prob_found + (1 - prob_found) * weight_in_view * prob_det)


# %%
# Visualisation
# -------------
# Finally, let's visualise the results...
#
# .. raw:: html
#
#     <details>
#     <summary><a>Click to show/hide plotting functions</a></summary>
#
# .. code-block:: python
#
#     from plotly import graph_objects as go
#     def plot_search_heatmap(plt, x_pos, y_pos, search_cell_history, **kwargs):
#    
#         # initialise heatmap trace
#         trace_base = len(plt.fig.data)
#    
#         heatmap_kwargs = dict(x=[], y=[], z=[], colorscale="YlOrRd", opacity=0.6,
#                               showlegend=True, showscale=False, name="heatmap",
#                               legendgroup="heatmap")
#         heatmap_kwargs.update(kwargs)
#         plt.fig.add_trace(go.Heatmap(heatmap_kwargs))
#    
#         # get number of traces already in plt
#         # add data for each frame
#         for frame in plt.fig.frames:
#        
#             data_ = list(frame.data)
#             traces_ = list(frame.traces)
#        
#             frame_time = datetime.fromisoformat(frame.name)
#        
#             for particle_state in search_cell_history:
#                 if frame_time == particle_state.timestamp:
#                     weights = [float(w) for w in particle_state.weight]
#                     data_.append(go.Heatmap(x=x_pos, y=y_pos, z=weights))
#                     traces_.append(trace_base)
#                 frame.traces = traces_
#                 frame.data = data_
#        
#         return plt
#
#
#     from typing import Collection
#     from stonesoup.types.state import GaussianState
#     from stonesoup.plotter import Plotterly
#
#     def plot_moving_sensor(plt, sensor_history, plot_fov=False, sensor_label="Moving Sensor",
#                             plot_radius=False, resize=True, **kwargs):
#         """Plots the position of a sensor over time. If simulation has multiple sensors, will
#         need to call this function multiple times.
#
#         sensor_history : Collection of :class:`~.Sensor`, ideally a list
#             Sensor information given at each time step
#         sensor_label: str
#             Label to apply to all tracks for legend.
#         \\*\\*kwargs: dict
#             Additional arguments. Defaults are ``marker=dict(symbol='x', color='black')``.
#         """
#
#         # ensure code doesn't break if sensor is only one timestep
#         if not isinstance(sensor_history, Collection):
#             sensor_history = {sensor_history}
#
#         if plot_fov or plot_radius:
#             from stonesoup.functions import pol2cart  # for plotting the sensor
#
#         # we have called a plotting function so update flag (used in _resize)
#         plt.plotting_function_called = True
#
#         # define the layout
#         trace_base = len(plt.fig.data)  # number of traces currently in figure
#         sensor_kwargs = dict(mode='markers', marker=dict(symbol='x', color='black'),
#                             legendgroup=sensor_label, legendrank=50,
#                             name=sensor_label, showlegend=True)
#         sensor_kwargs.update(kwargs)
#
#         plt.fig.add_trace(go.Scatter(sensor_kwargs))  # initialises trace
#
#         # for every frame, if sensor has same timestamp, get its location and add to the data
#
#         for frame in plt.fig.frames:  # the plotting bit
#
#             frame_time = datetime.fromisoformat(frame.name)  # get frame time in correct format
#             traces_ = list(frame.traces)
#             data_ = list(frame.data)
#
#             sensor_xy = np.array([np.inf, np.inf])
#
#             for sensor in sensor_history:
#                 if sensor.timestamp == frame_time:  # if sensor is in current timestep
#                     sensor_xy = np.array(sensor.position[[0, 1], 0])
#
#                     data_.append(go.Scatter(x=[sensor_xy[0]], y=[sensor_xy[1]]))
#                     traces_.append(trace_base)
#
#             frame.traces = traces_
#             frame.data = data_
#
#         if plot_fov:
#
#             # define the layout
#             trace_base = len(plt.fig.data)  # number of traces currently in figure
#             sensor_kwargs = dict(mode='lines', line=dict(dash="dash", color="black"),
#                                 hoverinfo=None, name="sensor fov", showlegend=True,
#                                 legendgroup="sensor fov")
#
#             plt.fig.add_trace(go.Scatter(sensor_kwargs))  # initialises trace
#
#             for frame in plt.fig.frames:
#
#                 frame_time = datetime.fromisoformat(frame.name)  # correct frame time format 
#                 traces_ = list(frame.traces)
#                 data_ = list(frame.data)
#
#                 x = [0, 0]  # for plotting fov if required
#                 y = [0, 0]
#
#                 for sensor in sensor_history:
#                     if sensor.timestamp == frame_time:  # if sensor is in current timestep
#                         for i, fov_side in enumerate((-1, 1)):
#                             range_ = min(getattr(sensor, 'max_range', np.inf), 100)
#
#                             x[i], y[i] = pol2cart(range_, sensor.dwell_centre[0, 0] \
#                                                     + sensor.fov_angle / 2 * fov_side) \
#                                                   + sensor.position[[0, 1], 0]
#
#                         data_.append(go.Scatter(x=[x[0], sensor.position[0], x[1]],
#                                                 y=[y[0], sensor.position[1], y[1]]))
#                         traces_.append(trace_base)
#
#                 frame.traces = traces_
#                 frame.data = data_
#
#         # plot radius of sensor
#         if plot_radius and sensor.max_range != np.inf:
#
#             # define the layout
#             trace_base = len(plt.fig.data)  # number of traces currently in figure
#             sensor_kwargs = dict(mode='lines', line=dict(dash="dash", color="black"),
#                                 hoverinfo=None, showlegend=True, name="sensor radius",
#                                 legendgroup="sensor radius")
#
#             plt.fig.add_trace(go.Scatter(sensor_kwargs))  # initialises trace
#
#             for frame in plt.fig.frames:
#
#                 # get frame time in correct format
#                 frame_time = datetime.fromisoformat(frame.name)
#                 traces_ = list(frame.traces)
#                 data_ = list(frame.data)
#
#                 for sensor in sensor_history:
#
#                     if sensor.timestamp == frame_time:  # if sensor is in current timestep
#                             circle = GaussianState([[sensor.position[0],
#                                                     sensor.position[1]]],
#                                                     np.diag([sensor.max_range ** 2,
#                                                             sensor.max_range ** 2]))
#
#                             points = Plotterly._generate_ellipse_points(circle, [0, 1])
#
#                             data_.append(go.Scatter(x=points[0, :],
#                                                     y=points[1, :]))
#                             traces_.append(trace_base)
#
#                 frame.traces = traces_
#                 frame.data = data_
#         return plt
#
#
#     from stonesoup.plotter import AnimatedPlotterly
#     plt = AnimatedPlotterly(timesteps, width=550)
#     plt = plot_search_heatmap(plt, x_pos, y_pos, search_cell_info)
#     plt = plot_moving_sensor(plt, sensor_history, plot_fov=False, plot_radius=True)
#     plt.fig.update_xaxes(range=[-0.5, 14.5])
#     plt.fig.update_yaxes(range=[-0.5, 14.5])
#
# .. raw:: html
#
#     </details>
#     <br>  
#     <video autoplay loop controls width=100% height="auto">
#       <source src="../../_static/bayesian_search_ex2_plt1.webm" type="video/webm">
#     </video>
#     <br>
#
# | 
#
# As expected, at each timestep the platform and sensor move to the neighbouring group of cells
# that has the highest total probability of containing the target.
#
# Metrics
# -------
# The accompanying paper `[1] <#references>`_ shows how additional metrics can be used to
# benchmark the performance of search algorithms, and to compare reward functions to balance
# target search with tracking.
#
# References
# ----------
# [1] Harris et al. (2024) - Open source tools for Bayesian search - doi:10.1117/12.3012763
