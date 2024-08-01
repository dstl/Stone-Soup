#!/usr/bin/env python

"""
Multi-Target Tracking in 3D Using Platform Simulation
=====================================================
In the Stone Soup library, simulations can be set up and run using special
:class:`~.FixedPlatform` and :class:`~.Sensor` objects. Simulated data can be preferable to
real data as the user has more control over the tracking scenario and real
data can be difficult or costly to acquire.
"""

# %%
# Creating the Sensor
# -------------------
# We will begin by importing many relevant packages for the simulation.

from datetime import datetime
from datetime import timedelta
import numpy as np
import random


# Stone Soup imports:
from stonesoup.types.state import State, GaussianState
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, ConstantVelocity)
from stonesoup.models.measurement.nonlinear import \
    CartesianToElevationBearingRange
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.tracker.simple import MultiTargetMixtureTracker
from matplotlib import pyplot as plt

# %%
# We set the start time to be the moment when we begin the simulation; for
# simulations, the actual time doesn't matter, only the time delta between the
# start and the point in question. We also set a random seed to ensure a
# standard outcome. At the end, you can try changing this value to see how the
# stochastic nature of the simulation and tracker can produce very different
# tracking scenarios with the same parameters.
start_time = datetime.now()
np.random.seed(783)
random.seed(783)

# %%
# Create the Stationary Platform
# ------------------------------
# Next, we will create a platform that will hold our radar sensor. In this
# case, the platform is stationary and located at the point (0, 0, 0), though
# in general it need not be.
#
# Define the initial platform position, in this case the origin
platform_state_vector = StateVector([[0], [0], [0]])
position_mapping = (0, 1, 2)

# %%
# Create the initial state (position, time). Notice that the time is set to
# the simulation start time defined earlier
platform_state = State(platform_state_vector, start_time)

# %%
# Create our fixed platform
from stonesoup.platform.base import FixedPlatform
platform = FixedPlatform(
    states=platform_state,
    position_mapping=position_mapping
)

# %%
# Create a Sensor
# ---------------
# Now that our sensor platform has been created, we can create a sensor to
# attach to it. In this case, we will be using a radar that takes measurements
# of range, bearing, and elevation of the targets.
from stonesoup.sensor.radar.radar import RadarElevationBearingRange
from stonesoup.models.clutter import ClutterModel


# %%
# First we create a covariance matrix which is a suitable measurement accuracy
# for the radar sensor. This radar measures range with an accuracy of +/- 25m,
# elevation accuracy +/- 0.15, degrees and bearing accuracy of +/- 0.15
# degrees.
noise_covar = CovarianceMatrix(np.array(np.diag([np.deg2rad(0.15)**2,
                                                 np.deg2rad(0.15)**2,
                                                 25**2])))

# %%
# The radar needs to be informed of where x, y, and z are in the target state
# space. In Stone Soup the states are often of the form [x, vx, y, vy, z, vz].
radar_mapping = (0, 2, 4)

# %%
# A newer feature of the Stone Soup platform simulations are the ability to
# generate clutter directly from the sensors using the :class:`~.ClutterModel`
# class. Using the clutter models, we can simulate realistic clutter
# originating from the measurement model. Clutter is defined in the Cartesian
# plane and converted to the correct measurement types according to the
# sensor. We will now add a clutter model to the radar sensor. This clutter
# model will use a uniform distribution over the defined ranges in each
# dimension.
params = ((-10000, 10000),  # clutter min x and max x
          (-10000, 10000),  # clutter min y and max y
          (8000, 10000))  # clutter min z and max z
clutter_model = ClutterModel(
     clutter_rate=0.5,
     distribution=np.random.default_rng().uniform,
     dist_params=params
)

# %%
# Instantiate the radar and finally, attach the sensor to the stationary platform we defined above.
radar = RadarElevationBearingRange(
    ndim_state=6,
    position_mapping=radar_mapping,
    noise_covar=noise_covar,
    clutter_model=clutter_model
)

platform.add_sensor(radar)

# %%
# Create the Simulation
# ---------------------
# For this example, we wish to have a simulation of multiple airborne targets.
# We will use the :class:`~.MultiTargetGroundTruthSimulator` class to simulate
# the target paths, and then the :class:`~.PlatformDetectionSimulator` class
# to handle the radar simulation.

# %%
# Set a constant velocity transition model for the targets
transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.5), ConstantVelocity(0.5), ConstantVelocity(0.1)])

# %%
# Define the Gaussian State from which new targets are sampled on
# initialisation
initial_target_state = GaussianState(
    state_vector=StateVector([[0], [0], [0], [0], [9000], [0]]),
    covar=CovarianceMatrix(np.diag([2000, 50, 2000, 50, 100, 1]))
)

# %%
# And create the truth simulator for the targets
from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator
groundtruth_sim = MultiTargetGroundTruthSimulator(
    transition_model=transition_model,  # target transition model
    initial_state=initial_target_state,  # add our initial state for targets
    timestep=timedelta(seconds=1),  # time between measurements
    number_steps=120,  # 2 minutes
    birth_rate=0.05,  # 5% chance of a new target being born every second
    death_probability=0.05  # 5% chance of a target being killed
)

# %%
# With our truth data generated and our sensor platform placed, we can now
# construct a simulator to generate measurements of the targets from each
# of the sensors in the simulation; in this case, just the stationary radar.
from stonesoup.simulator.platform import PlatformDetectionSimulator

sim = PlatformDetectionSimulator(
    groundtruth=groundtruth_sim,
    platforms=[platform]
)


# %%
# Set Up the Tracking Algorithm
# -----------------------------
# For this example, we will be using the JPDA algorithm to perform "soft"
# associations of the measurements to the targets. This is necessary as we
# have multiple airborne targets whose paths may intersect - a "hard" or
# "greedy" association algorithm such as the GNN may have issues in these
# cases.

# %%
# First, we create a Kalman predictor using the transition model from the
# target simulation. In real situations, you may not know the actual
# transition model.
from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

# %%
# Next, we define a measurement model for the Kalman updater. Here we have
# altered the noise covariance matrix slightly to make it harder for the
# tracker.
meas_covar = np.diag([np.deg2rad(0.5), np.deg2rad(0.15), 25])
meas_covar_trk = CovarianceMatrix(1.0*np.power(meas_covar, 2))
meas_model = CartesianToElevationBearingRange(
    ndim_state=6,
    mapping=(0, 2, 4),
    noise_covar=meas_covar_trk
)

# %%
# Using the measurement model, we make a Kalman updater which we will pass
# into our JPDA tracker.
from stonesoup.updater.kalman import ExtendedKalmanUpdater
updater = ExtendedKalmanUpdater(measurement_model=meas_model)

# %%
# The hypothesiser will assume that there is a 95% chance to measure any given
# target at any given timestep. In real life, this probability is based on the
# SNR of the target signals. The clutter spatial density of the hypothesiser
# can be changed to check what happens when there is a mismatch between the
# estimated clutter rate and actual clutter rate.
from stonesoup.hypothesiser.probability import PDAHypothesiser
Pd = 0.95  # 95%
hypothesiser = PDAHypothesiser(predictor=predictor,
                               updater=updater,
                               clutter_spatial_density=0.5,
                               prob_detect=Pd)

# %%
# Using the hypothesiser, we can make a data associator. Other MTT algorithms
# may use different association algorithms (like GNN)
from stonesoup.dataassociator.probability import JPDA
data_associator = JPDA(hypothesiser=hypothesiser)

# %%
# We implement a simple deleter algorithm to delete tracks if no measurements
# have fallen within the JPDA gating region in 3 time steps.
deleter = UpdateTimeStepsDeleter(time_steps_since_update=3)

# %%
# We will now set up a track initiator. In real life, targets may enter the
# measurement zone at any time during the collection period, and may leave at
# any point as well. To distinguish new targets from random clutter, we use a
# track initiator. This specific algorithm is a multi-measurement initiator;
# it utilises features of the tracker to initiate and hold tracks temporarily
# within the initiator itself, releasing them to the tracker once there are
# multiple detections associated with them enough to determine that they are
# "sure" tracks. In this case, the tracks are released after 3 appropriate
# detections in a row.
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.dataassociator.neighbour import NearestNeighbour

min_detections = 3  # number of detections required to begin a track
initiator_prior_state = GaussianState(
    state_vector=np.array([[0], [0], [0], [0], [0], [0]]),
    covar=np.diag([0, 10, 0, 10, 0, 10])**2
)

initiator_meas_model = CartesianToElevationBearingRange(
    ndim_state=6,
    mapping=np.array([0, 2, 4]),
    noise_covar=noise_covar
)

initiator = MultiMeasurementInitiator(
    prior_state=initiator_prior_state,
    measurement_model=meas_model,
    deleter=deleter,
    data_associator=NearestNeighbour(hypothesiser),
    updater=updater,
    min_points=min_detections,
    updates_only=True
)

# %%
# Now we are ready to Create a JPDA multi-target tracker.
JPDA_tracker = MultiTargetMixtureTracker(
    initiator=initiator,
    deleter=deleter,
    detector=sim,
    data_associator=data_associator,
    updater=updater
)

# %%
# Run the Simulation and Tracker
# ------------------------------
# Since the JPDA tracker holds the simulation variables, we can easily iterate
# through the tracker. Each time it will update the groundtruth simulation,
# generate detections using our fixed platform and radar, and run the tracking
# algorithm.

# Create lists to hold the information we want to plot later
tracks_plot = set()
tracks_id = set()
groundtruth_plot = set()
detections_plot = set()

# Run the simulation and tracker
for time, ctracks in JPDA_tracker:
    print(time)  # allows us to see the progress of the tracking simulation

    for track in ctracks:
        tracks_plot.add(track)
    for truth in groundtruth_sim.current[1]:
        groundtruth_plot.add(truth)
    for detection in sim.detections:
        detections_plot.add(detection)

# %%
# Plot the Results
# ----------------
# Now that all of the relevant information has been extracted, the results can
# be plotted using the 3D plotting functionality provided by the
# :class:`~.Plotter` class.
from stonesoup.plotter import Plotter, Dimension
plotter = Plotter(Dimension.THREE)
plotter.plot_ground_truths(groundtruth_plot, [0, 2, 4])
plotter.plot_measurements(detections_plot, [0, 2, 4])
plotter.plot_tracks(tracks_plot, [0, 2, 4], uncertainty=False, err_freq=5)

# %%
# We will also make a plot without measurements/clutter to better see the
# tracks.
plotter2 = Plotter(Dimension.THREE)
plotter2.plot_ground_truths(groundtruth_plot, [0, 2, 4])
plotter2.plot_tracks(tracks_plot, [0, 2, 4], uncertainty=True, err_freq=5)

# %%
# Metrics
# -------
# To analyse the tracker performance, we will use the OSPA, SIAP, and
# uncertainty metrics. For each of these metrics, we make a generator
# object which gets put into a metric manager.

# OSPA metric
from stonesoup.metricgenerator.ospametric import OSPAMetric
ospa_generator = OSPAMetric(c=40, p=1,
                            generator_name='OSPA metrics',
                            tracks_key='tracks',
                            truths_key='truths'
                           )

# SIAP metrics
from stonesoup.metricgenerator.tracktotruthmetrics import SIAPMetrics
from stonesoup.measures import Euclidean
SIAPpos_measure = Euclidean(mapping=np.array([0, 2]))
SIAPvel_measure = Euclidean(mapping=np.array([1, 3]))
siap_generator = SIAPMetrics(
    position_measure=SIAPpos_measure,
    velocity_measure=SIAPvel_measure,
    generator_name='SIAP metrics',
    tracks_key='tracks',
    truths_key='truths'
)

# Uncertainty metric
from stonesoup.metricgenerator.uncertaintymetric import \
    SumofCovarianceNormsMetric
uncertainty_generator = SumofCovarianceNormsMetric(generator_name='Uncertainty metric',
                                                   tracks_key='tracks')

# %%
# The metric manager requires us to define an associator. Here we want to
# compare the track estimates with the ground truth.

from stonesoup.dataassociator.tracktotrack import TrackToTruth
associator = TrackToTruth(association_threshold=30)

from stonesoup.metricgenerator.manager import MultiManager
metric_manager = MultiManager(
    [ospa_generator, siap_generator, uncertainty_generator],
    associator=associator
)

# %%
# Since we saved the groundtruth and tracks before, we can easily add them
# to the metric manager now, and then tell it to generate the metrics.
metric_manager.add_data({'truths': groundtruth_plot, 'tracks': tracks_plot})
metrics = metric_manager.generate_metrics()


# %%
# The first metric we will look at is the OSPA metric.
from stonesoup.plotter import MetricPlotter

fig1 = MetricPlotter()
fig1.plot_metrics(metrics, generator_names=['OSPA metrics'])


# %%
# Next are the SIAP metrics. Specifically, we will look at the position and
# velocity accuracy.
fig2 = MetricPlotter()
fig2.plot_metrics(metrics, metric_names=['SIAP Position Accuracy at times',
                                         'SIAP Velocity Accuracy at times'])
fig2.set_fig_title('SIAP metrics')


# %%
# Finally, we will examine a general uncertainty metric. This is calculated as
# the sum of the norms of the covariance matrices of each estimated state.
# Since the sum is not normalized for the number of estimated states, it is
# most important to look at the trends of this graph rather than the values.
fig3 = MetricPlotter()
fig3.plot_metrics(metrics, generator_names=['Uncertainty metric'])
fig3.set_ax_title(['Track uncertainty over time'])