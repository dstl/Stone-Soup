"""
Polar Plotting Example
========================
This example demonstrates use of the  :class:`~.PolarPlotterly` plotting class.
:class:`~.PolarPlotterly` uses :func:`plotly.graph_objects.Scatterpolar` to plot ground truths,
detections, and tracks in a polar plotter.

In this example, two airborne platforms are generated in a Cartesian state space. A
:class:`~.RadarBearingRange` sensor is used to convert the Cartesian state space to an
angular one. Angular ground truth is created using measurements without noise. Detections 
(with noise) are also generated.  Both detections and ground truth are plotted in a polar plot.
"""

# %%
# First, include some standard imports and initialise the start time:
from datetime import datetime
from datetime import timedelta

import numpy as np

from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, ConstantVelocity)
from stonesoup.models.transition.linear import KnownTurnRate
from stonesoup.platform.base import MultiTransitionMovingPlatform, MovingPlatform
from stonesoup.plotter import PolarPlotterly, Plotterly
from stonesoup.sensor.radar.radar import RadarBearingRange
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthPath
from stonesoup.types.state import State

# Define the simulation start time
start_time = datetime(2023, 1, 1)

# %%
# Generate Cartesian State Space Data
# -----------------------------------
# Two targets are created:
#  #. Target 1 moves in a ‘C’ shape. First it moves west (negative x), then it starts a slow,
#     long 180-degree turn moving south, until it is moving east (positive x).
#  #. Target 2 moves in a straight line from north to south.

# Create manoeuvre behaviours and durations for our moving platform
straight_level = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.), ConstantVelocity(0.)])

# Configure target 1's turn behaviour
turn_noise_diff_coeffs = np.array([0., 0.])
turn_rate = np.deg2rad(5)  # specified in radians per second
turn_model = KnownTurnRate(turn_noise_diff_coeffs=turn_noise_diff_coeffs, turn_rate=turn_rate)
turning = CombinedLinearGaussianTransitionModel([turn_model])

manoeuvre_list = [straight_level, turning, straight_level]
manoeuvre_times = [timedelta(seconds=12),
                   timedelta(seconds=36),
                   timedelta(seconds=12)]

# %%
# Using the manoeuvres created previously, two platforms are created that operate in Cartesian
# state space.
target_1_initial_state = State(StateVector([[600], [-50], [600], [0]]), start_time)
target_1 = MultiTransitionMovingPlatform(transition_models=manoeuvre_list,
                                         transition_times=manoeuvre_times,
                                         states=target_1_initial_state,
                                         position_mapping=(0, 2),
                                         velocity_mapping=(1, 3),
                                         sensors=None)

target_2_initial_state = State(StateVector([[500], [0], [700], [-24]]), start_time)
target_2 = MovingPlatform(transition_model=straight_level,
                          states=target_2_initial_state,
                          position_mapping=(0, 2),
                          velocity_mapping=(1, 3),
                          sensors=None)

# %%
# Simulate platform movement:
timesteps = [start_time + timedelta(seconds=i) for i in range(60)]

for t in timesteps:
    target_1.move(t)
    target_2.move(t)

# %%
# Display ground truth in Cartesian state space using the standard
# :class:`~.Plotterly` plotting class:
plotter_xy = Plotterly(title="Bird's Eye View of Targets")
mapping = [0, 2]
plotter_xy.plot_ground_truths(target_1, mapping=[0, 2], truths_label="Target 1")
plotter_xy.plot_ground_truths(target_2, mapping=[0, 2], truths_label="Target 2")
plotter_xy.fig

# %%
# Generate Angular State Space Data
# ----------------------------------
# Measure the previously generated cartesian state space using a
# :class:`~.RadarBearingRange` sensor:

# %%
# Create sensor:
sensor = RadarBearingRange(ndim_state=4,
                           position_mapping=[0, 2],
                           noise_covar=np.diag([np.radians(0.06), 50]),
                           position=StateVector([0, 0]))

# %%
# Plot sensor's location on XY plot
plotter_xy.plot_sensors({sensor}, mapping=[0, 1])
plotter_xy.fig

# %%
# Measure each ground truth individually and without noise to create ground truth paths:
angular_ground_truth_1 = GroundTruthPath([sensor.measure({target_1[t]}, noise=False).pop()
                                          for t in timesteps])
angular_ground_truth_2 = GroundTruthPath([sensor.measure({target_2[t]}, noise=False).pop()
                                          for t in timesteps])

# %%
# Generate detections:
detections = []
for t in timesteps:
    detections.extend(sensor.measure({target_1[t], target_2[t]}, noise=True))

# %%
# Time (s) vs Azimuth Angle (Radians)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
mapping = [0]
plotter_az_t_cart = Plotterly(title="Cartesian - Time (s) vs Azimuth Angle (Radians)",
                              xaxis=dict(title=dict(text="Time (seconds)")),
                              yaxis=dict(title=dict(text="Bearing (Radians)"))
                              )
plotter_az_t_cart.plot_ground_truths({angular_ground_truth_1},
                                     mapping=mapping, truths_label="Target 1")
plotter_az_t_cart.plot_ground_truths({angular_ground_truth_2},
                                     mapping=mapping, truths_label="Target 2")
plotter_az_t_cart.plot_measurements(detections, mapping=mapping, convert_measurements=False)
plotter_az_t_cart.fig

# %%
# This plot shows the current method to visualise angular data. Despite the steady motion of
# Target 1 there is a sharp discontinuity between 30s and 31s. The bearing from the sensor to
# Target 1 passes over π and the bearing wraps around to -3.1 radians. This can make the
# visualisation of data unclear. This isn’t an issue in a polar plot and is displayed in the next
# section.

# %%
# Create Polar Plots
# ------------------
# :class:`~.PolarPlotterly` inherits from :class:`~._Plotter` and therefore has all the same
# methods are the other plotters.

# %%
# Azimuth Angle (Degrees) vs Time (s)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
mapping = [0]
plotter_az_t = PolarPlotterly(title="Azimuth Angle (Degrees) vs Time (s)")
plotter_az_t.plot_ground_truths({angular_ground_truth_1}, mapping=mapping, truths_label="Target 1")
plotter_az_t.plot_ground_truths({angular_ground_truth_2}, mapping=mapping, truths_label="Target 2")
plotter_az_t.plot_measurements(detections, mapping=mapping, convert_measurements=False)
plotter_az_t.fig

# %%
# The range component of the polar plot represents time and the angle component represents the
# bearing of the sensor to the targets.

# %%
# Here we can see how the bearing of the target to the sensor changes over time.  Both targets move
# from 45° to 135°. Target 1 moves anti-clockwise around the sensor and target 2 moves clockwise.
# The detections have a good bearing accuracy and do not deviate far from the ground truth.

# %%
# Azimuth Angle (Degrees) vs Range (m)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
plotter_az_r = PolarPlotterly(title="Azimuth Angle (Degrees) vs Range (m)")
mapping = [0, 1]
plotter_az_r.plot_ground_truths({angular_ground_truth_1}, mapping=mapping, truths_label="Target 1")
plotter_az_r.plot_ground_truths({angular_ground_truth_2}, mapping=mapping, truths_label="Target 2")
plotter_az_r.plot_measurements(detections, mapping=mapping, convert_measurements=False)
plotter_az_r.fig

# %%
# The range component of the polar plot represents range and the angle component represents the
# bearing of the sensor to the targets. As a result the target motion looks identical in a polar
# plot and in a cartesian plot.


# %%
# Reference XY Plot
# ^^^^^^^^^^^^^^^^^
# This is similar to the previous x/y plot but also contains the detections.
plotter_xy.plot_measurements(detections, mapping=[0, 2])
plotter_xy.fig

# %%
# **Summary**
#
# Polar plots allow a more intuitive view of plotting angular data. This is especially true when
# a data source moves over the 2π limit and wraps around.
