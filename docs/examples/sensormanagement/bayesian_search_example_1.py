#!/usr/bin/env python

"""
Introduction to Bayesian Search - A Simple 1D Example
=====================================================
This scenario provides the simplest example of using Bayesian search in Stone Soup.
 
The paper accompanying this work, '*Open Source Tools for Bayesian Search*' `[1] <#references>`_,
can be found `here <https://doi.org/10.1117/12.3012763>`_.
"""

# %%
# Bayesian Search
# ---------------
# Implementations of sensor management often rely on the assumption that we have knowledge
# of targets' prior states. However, in a real scenario this is unlikely to be the case. It may
# be necessary to first search an environment in order to discover targets before we can start
# tracking them. Bayesian search is one approach to the problem of searching for undetected targets
# `[2] <#references>`_, and there are
# `many examples <https://en.wikipedia.org/wiki/Bayesian_search_theory>`_ of its application in the
# real world.
# 
# Bayesian search makes use of Bayesian statistics to incorporate prior beliefs about targets,
# represented as a probability distributions, into the calculation of optimal search behaviours.
# The process is as follows:
# 
# #.  Apply an initial probability distribution to the search space according to any prior beliefs
#     we hold about the target locations.
# #.  Take the action that corresponds to the maximisation of some parameter of interest (e.g.
#     probability of target detection).
# #.  Update the probability distribution based on what we observe.
# #.  Repeat steps 2 and 3 until the probability of targets remaining undetected drops below a 
#     threshold.
#
# There are many ways of extending this base process. For example, other parameters, such as
# information gain, search effort or physical constraints, could be included in the objective
# function. There is also scope to change how the action space is represented (discretely or
# continuously), and how the process of action selection is performed and optimised (e.g. myopic
# vs. non-myopic, brute force vs. heuristic algorithms, application of machine learning).
#
# When implementing Bayesian search, it is mathematically convenient to divide the search space
# into a number of discrete cells. Each cell can then be assigned a probability of containing a
# target.
# 
# Two equations are used to perform the update step (step 3). Adapting the notation from
# `[1] <#references>`_, we will here use :math:`w_{k}^{C}` and :math:`w_{k}^{¬C}` to represent the
# probability that a cell contains a target at timestep :math:`k` for an observed (:math:`C`) and 
# unobserved (:math:`¬C`) cell, respectively. :math:`p_d` is used to represent the probability of 
# detecting a target when looking in a cell containing a target.
#
# If, after observation, a target is not detected, we update each cell's probabilities such
# that:
#
# .. math::
#           w_k^{C} = w_{k-1}^{C}{\frac {1-p_d} {1-w_{k-1}^{C} p_d}}
#   :label: eq_1
#  
# if a cell is within our field of view, and:
#
# .. math::
#           w_k^{¬C} = {\frac {w_{k-1}^{¬C}} {1-w_{k-1}^{C} p_d}}
#   :label: eq_2
#  
# if it is not.
#
# Assuming an accurate but imperfect method of observation (:math:`0 < p_d < 1`), this has the
# effect of increasing the probability of target existence in unobserved cells, while
# decreasing it in observed cells. Note that for an imperfect observation method, the probability
# of existence will not drop to 0 after a target is not observed in a cell. This is because the
# lack of a detection may be due to sensor noise/error (represented by :math:`p_d`), as opposed
# to the absence of the target.
#
# It should also be noted that this implementation of Bayesian search assumes correct prior
# knowledge of the expected number of targets. We also assume that false alarms are not
# present.

# %%
# Bayesian Search in Stone Soup
# -----------------------------
# Stone Soup's existing sensor management infrastructure can be readily adapted to undertake
# Bayesian search.
# 
# The search space and corresponding probability distribution need to be represented in such a way
# that allows them to interact with the :class:`~.SensorManager`. One way of achieving this is with
# the :class:`~.ParticleState` class. The :class:`~.ParticleState`'s
# :attr:`~.ParticleState.state_vector` attribute can be used to represent the location of search
# cells and the :attr:`~.ParticleState.weight` attribute can be used to represent their probability
# of containing the target. This also allows us to easily check which cells are able to be observed
# by a sensor at any given time using the :meth:`~.RadarRotatingBearingRange.is_detectable` method.
#
# To choose optimal actions, the :class:`~.SensorManager` also requires an objective
# (reward) function adapted to return high reward when the sensor is looking toward regions with a
# higher probability of containing the target.

# %%
# Search Scenario
# ---------------
# To introduce Bayesian search in Stone Soup, we will start with a simple example. We will simulate
# a bearings-only sensor searching for a single stationary target (we will visualise this later).
# We assume that the sensor never locates the target, but we record the probability that the target
# should have been found by a given timestep. We will control the sensor using three
# different search algorithms, one of which will be Bayesian search.
# 
# The idea here is to gain some intuition of the logic and mathematics underpinning Bayesian
# search, before applying it to higher-fidelity scenarios.

# %%
# Initiate Simulation Variables
# -----------------------------
# We begin with some generic imports and simulation variables that will be used throughout the
# notebook.

from copy import copy, deepcopy
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
import time
import random

# use fixed seed for random number generators
np.random.seed(123)
random.seed(123)

# number of timesteps
simulation_length = 16
# number of cells in our search grid
n_cells = 24

start_time = datetime(2025, 5, 9, 14, 15)
timesteps = [start_time + timedelta(seconds=k) for k in range(simulation_length)]


# %%
# Initialise the Sensor
# ---------------------
# In this example we will be using a bearings-only sensor, with a fixed position, a 45 degree field
# of view (FOV) and a maximum rotation speed of 180 degrees per second.
# 
# We also define our probability of detection here, which will be used later when updating the
# probability distribution in our search space at each timestep.


from stonesoup.types.state import StateVector
from stonesoup.sensor.radar.radar import RadarRotatingBearing

# set probability that sensor detects target if target is in cell
prob_det = 0.9

res = 360/n_cells  # each cell covers this angle
sensor_fov = 3 * res  # sensor's FOV spans three cells

# create the sensor
sensor = RadarRotatingBearing(
    position_mapping=(0, 2),
    noise_covar=np.array([[0, 0],
                          [0, 0]]),
    ndim_state=4,
    position=[[0], [0]],
    rpm=30,
    fov_angle=np.radians(sensor_fov),
    dwell_centre=StateVector([np.pi]),
    clutter_model=None,
    resolution=np.radians(res)  # resolution of sensor equal to distance between search cells
) 
sensor.timestamp = start_time


# %%
# Create Custom Reward Function for Sensor Manager
# ------------------------------------------------
# The :class:`~.SensorManager` will be responsible for deciding in which direction our sensor looks
# at each timestep. It does this by assessing the 'benefit' of each possible action according to a
# specified reward function. In this scenario we want to reward actions that allow us to search
# cells with the greatest probability of containing the target. To do this, we define a custom
# reward function for the sensor manager that sums the weights (probabilities) of all particles
# within the sensor's FOV.


def sumofweightsreward(config, undetectmap, timestamp):
    predicted_sensors = set()
    # for each sensor and action in our prospective configuration
    for sensor, actions in config.items():
        # create a copy of the sensor with which to simulate the action
        predicted_sensor = deepcopy(sensor)
        # perform the action
        predicted_sensor.add_actions(actions)
        predicted_sensor.act(timestamp)
        predicted_sensors.add(predicted_sensor)
        
    # total probability of cells within our sensor's FOV
    total_prob = 0
    
    # calcuate the reward for each simulated action
    for sensor in predicted_sensors:
        
        # assume no detection
        
        for j, particle in enumerate(undetectmap.state_vector):
            pstate = ParticleState(particle)
            weight = undetectmap.weight[j]

            # if particle in sensor's FOV, add the probability to running total
            if sensor.is_detectable(pstate):
                total_prob += weight
                
    return float(total_prob)


# %%
# Create Sensor Managers
# ----------------------
# To compare Bayesian search to some other approaches, we will employ three different sensor
# management algorithms:
#
# *  for **Bayesian search** we will use the :class:`~.OptimizeBruteSensorManager`, which uses a
#    brute force search over a defined input grid (see `scipy.optimze.brute()
#    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brute.html#brute>`_) to
#    calculate the probability of target detection corresponding to each action from a subset of
#    those available to the sensor.
#
# We will compare Bayesian search with:
#
# *  a **random search**, which uses the :class:`~.RandomSensorManager` and chooses a random action
#    at each timestep, and
# *  a **sequential search**, which searches every cell in the search space sequentially.


from stonesoup.sensormanager import OptimizeBruteSensorManager, RandomSensorManager

# Bayesian search
sensor1 = deepcopy(sensor)
optbrutesensormanager = OptimizeBruteSensorManager(sensors={sensor1},
                                                   reward_function=sumofweightsreward)

# random search
sensor2 = deepcopy(sensor)
randomsensormanager = RandomSensorManager(sensors={sensor2})

# sequential search
sensor3 = deepcopy(sensor)  # doesn't require a sensor manager


# %%
# Generate Prior Probability Distribution
# ---------------------------------------
# To conduct our search, we need a way of representing where we think the target is. In this
# case the prior probability distribution will be spread around our sensor, which is located at the
# origin. Though possible to represent this continuously, it is more mathematically convenient to
# split our simulation space into discrete cells and populate each cell with a probability of
# target existence. Here, we have chosen to split the search space around the sensor into 24 cells.
#
# To showcase the power of Bayesian search, we must ensure our target prior probability
# distribution is not uniform. By using a uniform distribution, we claim to have no prior knowledge
# of target location - it could be anywhere. This leads to the Bayesian search pattern being the
# same as a heuristic linear sweep. From this, we could naively conclude that Bayesian search is
# the same as using set search patterns.
#
# Ideally, contextual information about the search environment would be used to create a prior
# probability distribution. However, in this case we will simply generate and store a bimodal
# probability distribution for our prior estimate of the target's location.

# find angles from sensor (located at origin) to each cell
angles = np.linspace(0, 2*np.pi, n_cells, endpoint=False)

# create non-uniform prior probability distribution
increasing_vals = np.array([i+0.1 for i in range(n_cells//4)])
prior_weights = np.concatenate((increasing_vals, np.flip(increasing_vals),
                                increasing_vals, np.flip(increasing_vals)), axis=None)

# ensure that it's normalised
prior_weights = prior_weights / np.sum(prior_weights)


# %%
# Initialise Particles
# --------------------
# As mentioned earlier in the example, we need a way of getting the probability distribution to
# interact with a Stone Soup :class:`~.SensorManager`. We need to represent both the search cell
# location and its probability of containing the target at each timestep.
# 
# There's more than one way of doing this in Stone Soup. In the original paper
# `[1] <#references>`_, :class:`~.Track` and :class:`~.Truth` objects were used to achieve this,
# but here we will adopt a slightly more efficient approach, making use of :class:`~.ParticleState`
# objects. The :class:`~.ParticleState` class has the :attr:`.~ParticleState.state_vector`,
# :attr:`.~ParticleState.timestamp`, and :attr:`.~ParticleState.weight` properties - all of which
# can be used to represent our probability distribution at each timestep.


# create an x pos, y pos and zero velocities for each of the 24 cells in our search space
x_pos = np.cos(angles)
y_pos = np.sin(angles)
vel = [0] * n_cells

from stonesoup.types.state import ParticleState
from stonesoup.types.array import StateVectors

# turn these values into state vectors
state_vectors = StateVectors(np.array([x_pos, vel, y_pos, vel]))

# use the state vectors to create a prior consisting of 24 particles
prior = ParticleState(state_vector=state_vectors,
                      weight=prior_weights,
                      timestamp=start_time)


# %%
# Search Loop
# -------------
# The final thing we need to do is define our search loop.
# At each timestep we:
#
# #. Get the action from the sensor manager and move the sensor accordingly.
# #. Make an observation (in this case we assume the target was not found).
# #. Update the probability distribution (based on the target not being found), by looping through
#    each cell ( *j* ) within the sensor's FOV and:
#
#    a. Updating all non-*j* cells (both unobserved cells and other observed cells) to reflect the
#       lack of detection in *j*. In this case, this increases the probability of the target being
#       in non-*j* cells.
#    b. Updating the probability of *j*. In this case, this reduces the probability, as a
#       relatively unlikely missed detection becomes the only way the target can be there.


def search_loop(prior, sensor, sensormanager, timesteps, prob_det, seq_flag=False):
    st = time.time()
    current_state = prior
    search_cell_info = [prior]
    sensor_info = [copy(sensor)]
    prob_found_list = [0]
    
    for timestep in timesteps[1:]:
        
        # update the search cell states with a new timestamp
        next_state = ParticleState(prior.state_vector, weight=current_state.weight,
                                   timestamp=timestep)

        # if running sequential search, perform this now
        if seq_flag:
            sensor.timestamp = timestep
            sensor.dwell_centre = sensor.dwell_centre + sensor.fov_angle
            
        else:
            chosen_actions = sensormanager.choose_actions(next_state, timestep)

            for chosen_action in chosen_actions:
                for sens, actions in chosen_action.items():
                    sens.add_actions(actions)

            sensor.act(timestep)
            
        # add state of sensor into a set for plotting later
        sensor_info.append(copy(sensor))
            
        # bespoke Bayesian search updater for cell probabilities
        weight_in_view = 0
        # for each particle/search cell
        for j, particle in enumerate(next_state.state_vector):
            
            pstate = ParticleState(particle)
            weight = next_state.weight[j]
            
            # update particles according to eq. 1 and 2 above
            if sensor.is_detectable(pstate):
                weight_in_view += weight 
                # all other particles adjusted according to probability of not finding target in
                # cell j (eq.2)
                next_state.weight = next_state.weight/(1-weight*prob_det)
                # then correct the probability for cell j (eq. 1)
                next_state.weight[j] = weight * (1-prob_det)/(1-weight*prob_det)

        # updated search cell states becomes the prior for next time step
        current_state = next_state
        
        # save search cell state
        search_cell_info.append(copy(next_state))
        
        # update probability of finding target by now. Use eq.6 of paper
        prob_found = prob_found_list[-1]
        prob_found_list.append(prob_found + (1-prob_found) * weight_in_view * prob_det)
    
    print(f"Time taken = {time.time() - st}s")
    return sensor_info, search_cell_info, prob_found_list


# %%
# Running the Simulations
# -----------------------
# We now run the search loop for each of our three search patterns: optimised Bayesian search,
# random search and sequential search.


sensor_history_b, search_cell_history_b, probs_b = search_loop(prior, sensor1,
                                                               optbrutesensormanager, timesteps,
                                                               prob_det)
sensor_history_r, search_cell_history_r, probs_r = search_loop(prior, sensor2,
                                                               randomsensormanager, timesteps,
                                                               prob_det)
sensor_history_s, search_cell_history_s, probs_s = search_loop(prior, sensor3, None, timesteps,
                                                               prob_det, seq_flag=True)

# %%
# Visualising the Simulations
# ---------------------------
# Having run our simulations, we can now visualise the outcomes. To do this, we will utilise Stone
# Soup's :class:`~.AnimatedPlotter`.

# %%
# We can now plot the results with the animated plotter.
#
# .. raw:: html
#
#     <details>
#     <summary><a>Click to show/hide plotting functions</a></summary>
#
# .. code-block:: python
#
#     # we will convert the particles into Track objects for visualisation - this allows
#     # us to use the tracks' uncertainty ellipses in conjunction with the animated plotter to
#     # represent the relative probabilities of target existence in each cell.
#
#     from stonesoup.types.track import Track
#     from stonesoup.types.state import GaussianState
#
#     # function that takes particles from our tracking history and converts them to tracks
#     def particles_to_tracks(search_cell_history, sim_length, n_cells, x_pos, y_pos, timesteps):
#    
#         weights = [[float(weight) for weight in cell_group.weight]
#                for cell_group in search_cell_history]
#
#         tracks = [Track(
#             [GaussianState(state_vector=[30*x_pos[i], 0, 30*y_pos[i], 0],
#                             covar=np.diag([400*weights[j][i], 0,
#                                             400*weights[j][i], 0]),
#                             timestamp=timesteps[j])
#                             for j in range(sim_length)])
#                             for i in range(n_cells)]
#    
#         return tracks
#
#     # create a list of tracks for each of our search algorithms
#     tracks_b = particles_to_tracks(search_cell_history_b, simulation_length, n_cells, x_pos,
#                                    y_pos, timesteps)
#     tracks_s = particles_to_tracks(search_cell_history_s, simulation_length, n_cells, x_pos,
#                                    y_pos, timesteps)
#     tracks_r = particles_to_tracks(search_cell_history_r, simulation_length, n_cells, x_pos,
#                                    y_pos, timesteps)
#
#     from plotly import graph_objects as go
#     from typing import Collection
#     from stonesoup.plotter import Plotterly
#
#     def plot_moving_sensor(plt, sensor_history, plot_fov=False, sensor_label="Moving Sensor",
#                     plot_radius=False, resize=True, **kwargs):
#         """Plots the position of a sensor over time. If simulation has multiple sensors, will
#         need to call this function multiple times.
#
#         sensor_history : Collection of :class:`~.Sensor`, ideally a list
#           Sensor information given at each time step
#         sensor_label: str
#           Label to apply to all tracks for legend.
#         \\*\\*kwargs: dict
#           Additional arguments. Defaults are ``marker=dict(symbol='x', color='black')``.
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
#                         legendgroup=sensor_label, legendrank=50,
#                         name=sensor_label, showlegend=True)
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
#                             hoverinfo=None, name="sensor fov", showlegend=True,
#                             legendgroup="sensor fov")
#
#             plt.fig.add_trace(go.Scatter(sensor_kwargs))  # initialises trace
#
#             for frame in plt.fig.frames:
#
#                 frame_time = datetime.fromisoformat(frame.name)  # set frame time correct format
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
#                                                 + sensor.fov_angle / 2 * fov_side) \
#                                         + sensor.position[[0, 1], 0]
#
#                         data_.append(go.Scatter(x=[x[0], sensor.position[0], x[1]],
#                                             y=[y[0], sensor.position[1], y[1]]))
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
#                             hoverinfo=None, showlegend=True, name="sensor radius",
#                             legendgroup="sensor radius")
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
#                         circle = GaussianState([[sensor.position[0],
#                                             sensor.position[1]]],
#                                         np.diag([sensor.max_range ** 2,
#                                                     sensor.max_range ** 2]))
#
#                         points = Plotterly._generate_ellipse_points(circle, [0, 1])
#
#                         data_.append(go.Scatter(x=points[0, :],
#                                             y=points[1, :]))
#                         traces_.append(trace_base)
#
#                 frame.traces = traces_
#                 frame.data = data_
#         return plt
#
#     from stonesoup.plotter import AnimatedPlotterly
#     plt = AnimatedPlotterly(timesteps=timesteps, tail_length=1, sim_duration=6,
#                             width = 600, height = 600, equal_size=True,
#                            title="Optimised brute-force single stationary target search")
#
#     plot_moving_sensor(plt, sensor_history_b, plot_fov=True)
#
#     plt.plot_tracks(tracks_b, [0, 2], uncertainty=True, mode='lines',line=dict(color='red'),
#                     legendgroup=1, name='Dummy tracks', fillcolor='red', opacity=0.3,
#                     showlegend=False)
#
#
# .. raw:: html
#
#     </details>
#     <br>
#     <video autoplay loop controls width=100% height="auto">
#       <source src="../../_static/bayesian_search_ex1_plt1.webm" type="video/webm">
#     </video>
#     <br>

# %%
# | 
# 
# In this plot we see optimised Bayesian search in effect, as the cell probabilities are updated at
# each timestep and the sensor moves to observe the next most likely cell.


prob_found_plot = go.Figure()
prob_found_plot.add_trace(go.Scatter(y=probs_b, name="Optimised Bayesian"))
prob_found_plot.add_trace(go.Scatter(y=probs_s, name="Sequential"))
prob_found_plot.add_trace(go.Scatter(y=probs_r, name="Random"))
prob_found_plot.update_layout(title="Expected probability of finding target vs search effort",
                              xaxis_title="Search effort (no. actions taken)",
                              yaxis_title="Probability of having detected target",
                              xaxis_range=[0, 16])

# %%
# The second plot allows us to compare the performance of the three search strategies. Both
# optimised Bayesian and sequential search outperform the random search by achieving a higher
# probability of having detected the target throughout the scenario. Bayesian search also reaches a
# near conclusive outcome quicker than the sequential method.
#
# This example showcased the benefits of the Bayesian search approach in a relatively simple
# scenario. To see how Bayesian search can be applied in a more complex setting, continue to the
# :doc:`next example <bayesian_search_example_2>`.
#
# References
# ----------
# | [1] Harris et al. (2024) - Open source tools for Bayesian search - doi:10.1117/12.3012763
# | [2] Stone, L. D. (1976) - Theory of optimal search - Elsevier
