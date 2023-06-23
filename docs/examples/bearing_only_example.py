#!/usr/bin/env python

"""
Bearings-only tracking
====================================

Nonlinear bearing-only target tracking is a complex problem because the models have only the knowledge of
the direction towards the sensor, leaving a lot of information that needs to be extracted from the measurements
from the sensor.

In this short tutorial we simulate a radar placed on top of a moving platform using an Extended Kalman Filter
to produce the most accurate tracking of the target. In this example we use a distance based data associator
to merge the hypothesis and the measurements from the sensor.

"""

# %%%
# Layout
# ^^^^^^
# The layout of this example follows:
#
# 1) Create the moving platform and the Bearings-Only radar
# 2) Generate the target movements and groundtruths
# 3) Setup the simulation generating measurements and groundtruths
# 4) Run the simulation and create the plots
#

# some general imports
import numpy as np
from matplotlib import pyplot as plt

from datetime import datetime
from datetime import timedelta

# Load Stone Soup materials
from stonesoup.types.state import State, GaussianState
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.models.transition.linear import (CombinedLinearGaussianTransitionModel, ConstantVelocity)
from stonesoup.models.measurement.nonlinear import Cartesian2DToBearing

# Load the filter components
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.tracker.simple import SingleTargetTracker

# set a random seed and start of the simulation
np.random.seed(2001)
start_time = datetime.now()

#%%
# 1) Create the moving platforma and the radar
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Firstly create the initial state of the platform, including the origin point and the
# velocity in x,y of his movements. Then we create a transition motion (in 2D cartesian coordinates)
# of the platform.
# At this point we can create a Radar which receives only the bearing measurements from the targets.

# import the platform to place the sensor
from stonesoup.platform.base import MovingPlatform

# define the platform location, place it in the origin, and its movement. In addition specify the
# mapping of the position and the velocity mapping. This is done in 2D cartesian coordinates.
platform_state_vector = StateVector([[0],[-5], [0],[-7]])
position_mapping = (0,2)
velocity_mapping = (1,3)

# create the initial state (position and time)
platform_state = State(platform_state_vector, start_time)

# Create a platform transition model, lest assume it is moving into a straight line from its origin place
platform_transition_model = CombinedLinearGaussianTransitionModel([
    ConstantVelocity(0.0), ConstantVelocity(0.0)])

# Now we can create the platform with the initial state, the various mapping and the transition model
platform = MovingPlatform(states=platform_state,
                          position_mapping=position_mapping,
                          velocity_mapping=velocity_mapping,
                          transition_model=platform_transition_model)

# At this stage we need to create the sensor, let's import the RadarBearing. This sensor only elaborates the
# bearing measurements from the target, the range is not specified.
from stonesoup.sensor.radar.radar import RadarBearing

# Configure the radar noise, since we are using just a dimension we need to specify only the
# noise associated to the bearing, we assume a bearing accuracy  of +/- 0.025 degrees for each measurments
noise_covar = CovarianceMatrix(np.array(np.diag([np.deg2rad(0.025)**2])))

# This radar needs to be informed of the x and y mapping of the space target.
radar_mapping = (0,2)

# Instantiate the radar
radar = RadarBearing(ndim_state=4,
                     position_mapping=radar_mapping,
                     noise_covar=noise_covar)

# As presented in the other examples we have to attach the sensor on the platform.
platform.add_sensor(radar)
# At this point we can also check the offset rotation or the mounting of the radar in respect to the
# platform as shown in other tutorials.

# %%
# 2) Generate the target movements and groundtruths
# In this case we build a single target ground truth simulator using a simple transition model
# and a known initial target state

# Load the single target ground truth simulator
from stonesoup.simulator.simple import SingleTargetGroundTruthSimulator

# Instantiate the transition model
transition_model = CombinedLinearGaussianTransitionModel([
    ConstantVelocity(1.0), ConstantVelocity(1.0)])

# Define the initial state of the target using a gaussian state specifying the origin point and the
# accuracy of the predictions on the x and y (using the covariance matrix)
initial_target_state = GaussianState([50, 0, 50, 0],
                                     np.diag([1, 1, 1, 1])**2,
                                     timestamp=start_time)

# Setup the groundtruth simulation
groundtruth_simulation = SingleTargetGroundTruthSimulator(
    transition_model=transition_model,
    initial_state = initial_target_state,
    timestep=timedelta(seconds=1),
    number_steps = 100
)

# %%
# 3) Setup the simulation generating measurements and ground truths
# After defining the measuremnt model we will have the needed components to start running our example

# Define the measurement model using a Cartesian to bearing
meas_model = Cartesian2DToBearing(
    ndim_state=4,
    mapping=(0,2),
    noise_covar=noise_covar)

# Import the PlatformDetectionSimulator
from stonesoup.simulator.platform import PlatformDetectionSimulator

sim = PlatformDetectionSimulator(groundtruth=groundtruth_simulation, platforms=[platform])

# Instantiate the filter components
# Create an Unscented Kalman Predictor
predictor = ExtendedKalmanPredictor(transition_model)

# Create an Unscented Kalman Updater
updater = ExtendedKalmanUpdater(measurement_model=None)

# Instantiate the single point initiator
from stonesoup.initiator.simple import SinglePointInitiator

# Define a initiator, given the complexity of the bearing only tracking let's feed the
# same initial state to both the ground truths measurements and tracker.

initiator = SinglePointInitiator(
    prior_state = initial_target_state,
    measurement_model=meas_model)

# %%
# Add the hypothesiser components, we use a distance based hypothesiser using a Malahonobis
# distance to measure the data association between the measurements and the tracks.
# Since in this case we use a single target a simple nearest neighbour will work just fine

# Load the hypothesiser and data associator
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis

hypothesiser = DistanceHypothesiser(predictor, updater,
                                     measure=Mahalanobis(),
                                     missed_distance=5)

from stonesoup.dataassociator.neighbour import NearestNeighbour
data_associator = NearestNeighbour(hypothesiser)

# Instantiate the deleter
deleter = UpdateTimeStepsDeleter(time_steps_since_update=3)

# Build the kalman tracker
kalman_tracker = SingleTargetTracker(
    initiator=initiator,
    deleter=deleter,
    detector=sim,
    data_associator=data_associator,
    updater=updater
)

# %%
# 4) Run the simulation and create the plots
# At this stage we have everything for running the simulation, we have the tracker, the sensor
# measurements and position

kalman_tracks = {}  # Store for plotting later
groundtruth_paths = {}  # Store for plotting later

# Loop for the tracks and the groundtruths
for time, ctracks in kalman_tracker:
    for track in ctracks:
        loc = (track.state_vector[0], track.state_vector[2])
        if track not in kalman_tracks:
            kalman_tracks[track] = []
        kalman_tracks[track].append(loc)
    for truth in groundtruth_simulation.current[1]:
        loc = (truth.state_vector[0], truth.state_vector[2])
        if truth not in groundtruth_paths:
            groundtruth_paths[truth] = []
        groundtruth_paths[truth].append(loc)

# generate the Platform positions
Xp = [state.state_vector[0] for state in platform]
Yp = [state.state_vector[2] for state in platform]

# set up the plotter
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("$East$ (m)")
ax.set_ylabel("$North$ (m)")
ax.set_ylim(-1000, 400) # change eventually in case there is a different model or seed
ax.set_xlim(-900, 100)

for key in groundtruth_paths:
    X = [coord[0] for coord in groundtruth_paths[key]]
    Y = [coord[1] for coord in groundtruth_paths[key]]
    ax.plot(X, Y, color='r', label='GroundTruth')  # Plot true locations in red

for key in kalman_tracks:
    X = [coord[0] for coord in kalman_tracks[key]]
    Y = [coord[1] for coord in kalman_tracks[key]]
    ax.plot(X, Y, color='b', label='Track estimates')  # Plot track estimates in blue

# plot platform location
ax.plot(Xp, Yp, color='y', label='Radar track')
plt.legend()
plt.show()


"""
References
[1] Xiangdong Lin, Thiagalingam Kirubarajan, Yaakov Bar-Shalom, Simon Maskell, 
"Comparison of EKF, pseudomeasurement, and particle filters for a bearing-only target tracking problem", in 
Signal and Data Processing of Small Targets 2002, 2002, vol 4728, pp. 240-250. doi:10.1117/12.478508
[2] Vincent J. Aidala, "Kalman Filter Behavior in Bearing-Only Tracking Applications", IEEE Transactions
on Aerospace Electronic Systems, Vol. 15, January 1979
"""
