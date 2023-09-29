#!/usr/bin/env python
# coding: utf-8

"""
============================================================================
Comparing Efficient Hypothesis Management (EHM) with probability associators
============================================================================
"""

# In this example, we compare the performances between efficient hypothesis management
# (EHM) and standard joint probabilistic data association.
# The problem we face when dealing with multi-target tracking is the potential
# association of measurements to prediction, which can be ambiguous and
# that could lead into a combinatorial explosion. To reduce the computational
# cost of the operations a number of algorithms have been developed to
# match measurements and predictions to tracks. One of this methods is the
# efficient hypothesis management, explained in details in [1]_, [2]_ and under
# patent in [3]_; this algorithm improves the joint probability data association,
# which is a brute force approach, with improved capability of hypothesis matching
# and rejection with, significantly, cost reduction.
#
# As it stands, there is not a Stone Soup implementation yet, however we
# base these comparisons on the publicly available, under patent license agreement,
# Python package PyEHM developed by ` Dr. Lyudmil Vladimirov `_.
# .. _Lyudmil Vladimirov: https://github.com/sglvladi/pyehm
#
#
# This example follows the usual setup:
# 1) Generate a simple multi-target scenario simulation;
# 2) Prepare the trackers components with the different data associators;
# 3) Run the trackers to collect the tracks;
# 4) Compare the trackers performances;
#

# %%
# 1) Generate a simple multi-target scenario simulation;
# ------------------------------------------------------
# To start with this example we align a typical
# case of multi-target scenario with some level of
# clutter. Then, we set up the various components
# of the trackers. To use this example there is the
# need to install the independent package PyEHM
# using "pip install pyehm".

# %%
# General imports
# ^^^^^^^^^^^^^^^

import numpy as np
from datetime import datetime, timedelta
from copy import deepcopy

# Load the pyehm plugins
from pyehm.plugins.stonesoup import JPDAWithEHM, JPDAWithEHM2

# %%
# Stone Soup Imports
#

from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import GaussianState
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, ConstantVelocity)

# %%
# Simulation parameters setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

np.random.seed(1908)  # set the random seed for the simulation
simulation_start_time = datetime.now().replace(microsecond=0)  # simulation start

# initial state of all targets
initial_state_mean = StateVector([0, 0, 0, 0])
initial_state_covariance = CovarianceMatrix(np.diag([5, 0.5, 5, 0.5]))
timestep_size = timedelta(seconds=1)
number_of_steps = 50  # number of time-steps
birth_rate = 0.25   # probability of new target to appear
death_probability = 0.01  # 5% probability of target to disappear

# setup the initial state of the simulation
initial_state = GaussianState(state_vector=initial_state_mean,
                              covar=initial_state_covariance,
                              timestamp=simulation_start_time)

# create the targets transition model
transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.05), ConstantVelocity(0.05)])

# Put this all together in a multi-target simulator.
from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator
groundtruth_sim = MultiTargetGroundTruthSimulator(
    transition_model=transition_model,
    initial_state=initial_state,
    timestep=timestep_size,
    number_steps=number_of_steps,
    birth_rate=birth_rate,
    death_probability=death_probability)

# Load the measurements model
from stonesoup.models.measurement.linear import LinearGaussian

# initialise the measurement model
measurement_model_covariance = np.diag([0.5, 0.5])
measurement_model = LinearGaussian(4,
                                   [0, 2],
                                   measurement_model_covariance)

# probability of detection
probability_detection = 0.99

# %%
# Generate clutter
# ^^^^^^^^^^^^^^^^

clutter_area = np.array([[-1, 1], [-1, 1]])*30
surveillance_area = ((clutter_area[0][1]-clutter_area[0][0])*
                     (clutter_area[1][1]-clutter_area[1][0]))
clutter_rate = 1.2
clutter_spatial_density = clutter_rate/surveillance_area

# Instantiate the detector simulator
from stonesoup.simulator.simple import SimpleDetectionSimulator

detection_sim = SimpleDetectionSimulator(
    groundtruth=groundtruth_sim,
    measurement_model=measurement_model,
    detection_probability=probability_detection,
    meas_range=clutter_area,
    clutter_rate=clutter_rate)

# To make a 1 to 1 comparison between different trackers we have
# to feed the same detections to each trackers, so we have to
# copy the detection simulations.
from itertools import tee
detection, *detection_sims = tee(detection_sim, 4)

# %%
# 2) Prepare the trackers components with the different data associators;
# -----------------------------------------------------------------------
# We have setup the multi-target scenario, we instantiate all
# the relevant tracker components. We consider a
# :class:`~.UnscentedKalmanPredictor` and :class:`~.UnscentedKalmanUpdater`
# components for the tracker. Then, for the data association we
# use the :class:`~.JPDA` data associator implementation
# present in Stone Soup and the JPDA PyEHM implementation to
# gather relevant comparisons. Please note that we have to
# create multiple copies of the same detector simulator
# to provide each tracker with the same set of detections for
# a fairer comparison.

# %%
# Stone Soup tracker components
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Load the various Kalman components as the predictor and
# updater
from stonesoup.predictor.kalman import UnscentedKalmanPredictor
from stonesoup.updater.kalman import UnscentedKalmanUpdater

# Instantiate the components
predictor = UnscentedKalmanPredictor(transition_model)
updater = UnscentedKalmanUpdater(measurement_model)

# Load the Initiator, Deleter and compose the trackers
from stonesoup.deleter.time import UpdateTimeStepsDeleter
deleter = UpdateTimeStepsDeleter(3)

from stonesoup.initiator.simple import MultiMeasurementInitiator

# Load the probabilistic data Associator and the tracker
from stonesoup.dataassociator.neighbour import GlobalNearestNeighbour
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.dataassociator.probability import JPDA
from stonesoup.tracker.simple import MultiTargetMixtureTracker

# %%
# Design the trackers
# ^^^^^^^^^^^^^^^^^^^

# Lets start with the standard JPDA
initiator = MultiMeasurementInitiator(
    prior_state=GaussianState(np.array([0, 0, 0, 0]),
                              np.diag([5, 0.5, 5, 0.5]) ** 2,
                              timestamp=simulation_start_time),
    measurement_model=None,
    deleter=deleter,
    data_associator=GlobalNearestNeighbour(PDAHypothesiser(predictor=predictor,
                                         updater=updater,
                                         clutter_spatial_density=clutter_spatial_density,
                                         prob_detect=probability_detection)),
    updater=updater,
    min_points=2)

# Tracker
JPDA_tracker = MultiTargetMixtureTracker(
    initiator=initiator,
    deleter=deleter,
    detector=detection_sims[0],
    data_associator=JPDA(PDAHypothesiser(predictor=predictor,
                                         updater=updater,
                                         clutter_spatial_density=clutter_spatial_density,
                                         prob_detect=probability_detection)),
    updater=updater)

# Now lets load the EHMJPDA, if you note the initiator is the same
EHM_initiator = MultiMeasurementInitiator(
    prior_state=GaussianState(np.array([0, 0, 0, 0]),
                              np.diag([5, 0.5, 5, 0.5]) ** 2,
                              timestamp=simulation_start_time),
    measurement_model=None,
    deleter=deleter,
    data_associator=GlobalNearestNeighbour(PDAHypothesiser(predictor=predictor,
                                         updater=updater,
                                         clutter_spatial_density=clutter_spatial_density,
                                         prob_detect=probability_detection,)),
    updater=updater,
    min_points=2)

# In this tracker we use the JPDA with EHM
EHM1_tracker = MultiTargetMixtureTracker(
    initiator=EHM_initiator,
    deleter=deleter,
    detector=detection_sims[1],
    data_associator=JPDAWithEHM(PDAHypothesiser(predictor=predictor,
                                         updater=updater,
                                         clutter_spatial_density=clutter_spatial_density,
                                         prob_detect=probability_detection)),
    updater=updater)

# Copy the same initiator for EHM
EHM2_initiator = deepcopy(EHM_initiator)

# This tracker uses the the second implementation
# of EHM.
EHM2_tracker = MultiTargetMixtureTracker(
    initiator=EHM2_initiator,
    deleter=deleter,
    detector=detection_sims[2],
    data_associator=JPDAWithEHM2(PDAHypothesiser(predictor=predictor,
                                                updater=updater,
                                                clutter_spatial_density=clutter_spatial_density,
                                                prob_detect=probability_detection)),
    updater=updater)


# %%
# 3) Run the trackers to generate the tracks;
# -------------------------------------------
# We have instantiated the three versions of the
# trackers, one with the brute force JPDA hypothesis
# management, one with the EHM implementation [1]_ and
# one with the EHM2 implementation  [2]_.
# Now we can run the trackers and gather the
# final tracks as well as the detections,
# clutter and define a metric plotter to evaluate
# the track accuracy using the metric manager.
# As the three methods will use the same hypothesis
# we will obtain the same tracks, we verify such
# claim by comparing the OSPA metric between
# each hyphotesiser.
# To measure the significant difference in
# computing time we measure the time while running the
# three different trackers.

# %%
# Stone Soup Metrics imports
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

# Instantiate the metrics tracker
from stonesoup.metricgenerator.basicmetrics import BasicMetrics

basic_JPDA = BasicMetrics(generator_name='basic_JPDA', tracks_key='JPDA_tracks',
                          truths_key='truths')
EHM1 = BasicMetrics(generator_name='EHM1', tracks_key='EHM1_tracks',
                    truths_key='truths')
EHM2 = BasicMetrics(generator_name='EHM2', tracks_key='EHM2_tracks',
                    truths_key='truths')

# Compare the generated tracks to verify they obtain the same
# accuracy
from stonesoup.metricgenerator.ospametric import OSPAMetric

ospa_JPDA_EHM1 = OSPAMetric(c=40, p=1, generator_name='OSPA_JPDA-EHM1',
                            tracks_key='JPDA_tracks', truths_key='EHM1_tracks')
ospa_JPDA_EHM2 = OSPAMetric(c=40, p=1, generator_name='OSPA_JPDA-EHM2',
                           tracks_key='JPDA_tracks', truths_key='EHM2_tracks')

# Define the track data associator
from stonesoup.dataassociator.tracktotrack import TrackToTruth

associator = TrackToTruth(association_threshold=30)

# Load the plotter
from stonesoup.metricgenerator.plotter import TwoDPlotter

plot_generator_JPDA = TwoDPlotter([0, 2], [0, 2], [0, 2], uncertainty=True, tracks_key='JPDA_tracks',
                                 truths_key='truths', detections_key='detections',
                                 generator_name='JPDA_plot')
plot_generator_EHM1 = TwoDPlotter([0, 2], [0, 2], [0, 2], uncertainty=True, tracks_key='EHM1_tracks',
                                truths_key='truths', detections_key='detections',
                                generator_name='EHM1_plot')

plot_generator_EHM2 = TwoDPlotter([0, 2], [0, 2], [0, 2], uncertainty=True, tracks_key='EHM2_tracks',
                                truths_key='truths', detections_key='detections',
                                generator_name='EHM2_plot')

# Load the multi-manager
from stonesoup.metricgenerator.manager import MultiManager

# Load all the relevant components of the plots in the metric manager
metric_manager = MultiManager([basic_JPDA,
                               EHM1,
                               EHM2,
                               ospa_JPDA_EHM1,
                               ospa_JPDA_EHM2,
                               plot_generator_JPDA,
                               plot_generator_EHM1,
                               plot_generator_EHM2
                               ], associator)

# %%
# Run simulation
# ^^^^^^^^^^^^^^

# Now lets plot the various tracker results
JPDA_tracks = set()
EHM1_tracks = set()
EHM2_tracks = set()
groundtruths = set()
detections_set = set()

# measure the time
start_time = datetime.now()
for time, ctracks in JPDA_tracker:
    JPDA_tracks.update(ctracks)
    detections_set.update(detection_sim.detections)
jpda_time = datetime.now() - start_time

groundtruths = groundtruth_sim.groundtruth_paths

start_time = datetime.now()
for time, etracks in EHM1_tracker:
    EHM1_tracks.update(etracks)
ehm1_time = datetime.now() - start_time

start_time = datetime.now()
for time, etracks in EHM2_tracker:
    EHM2_tracks.update(etracks)
ehm2_time = datetime.now() - start_time

# Add the various tracks to the metric manager
metric_manager.add_data({'JPDA_tracks': JPDA_tracks}, overwrite=False)
metric_manager.add_data({'truths': groundtruths,
                        'detections': detections_set}, overwrite=False)
metric_manager.add_data({'EHM1_tracks': EHM1_tracks}, overwrite=False)
metric_manager.add_data({'EHM2_tracks': EHM2_tracks}, overwrite=False)


# %%
# 4) Compare the trackers performances;
# -------------------------------------
# We have setup the trackers as well as
# the metric manager now, to conclude this
# tutorial we show the results of the
# computing time needed for each tracker,
# the overall tracks generated and
# the differences between the tracks,
# if any. We start showing the time
# performances of the different trackers.

print('Comparisons between the trackers performances')
print(f'JPDA computing time: {jpda_time} seconds')
print(f'EHM1 computing time: {ehm1_time} seconds, {np.round((jpda_time/ehm1_time-1)*100, 2)} % quicker than JPDA')
print(f'EHM2 computing time: {ehm2_time} seconds, {np.round((jpda_time/ehm2_time-1)*100, 2)} % quicker than JPDA')

# Load the plotter package to plot the
# detections, tracks and detections.
from stonesoup.plotter import Plotterly

plotter = Plotterly()

plotter.plot_ground_truths(groundtruths, [0, 2])
plotter.plot_measurements(detections_set, [0, 2])
plotter.plot_tracks(JPDA_tracks, [0, 2], line= dict(color='orange'),
                    track_label='JPDA tracks')
plotter.plot_tracks(EHM1_tracks, [0, 2], line= dict(color='green', dash='dot'),
                    track_label='EHM1 tracks')
plotter.plot_tracks(EHM2_tracks, [0, 2], line= dict(color='red', dash='dot'),
                    track_label='EHM2 tracks')
plotter.fig

# %%
# Show the metrics
# ^^^^^^^^^^^^^^^^

# Now we process the metrics
metrics = metric_manager.generate_metrics()

# Load the metric plotter
from stonesoup.plotter import MetricPlotter
graph = MetricPlotter()

graph.plot_metrics(metrics, generator_names=['OSPA_JPDA-EHM1',
                                             'OSPA_JPDA-EHM2'],
                   color=['orange',  'blue'])

# update y-axis label and title, other subplots are displaying auto-generated title and labels
graph.axes[0].set(ylabel='OSPA metrics', title='OSPA distances over time between JPDA and EHMs tracks')
graph.fig

# %%
# Conclusion
# ----------
# In this example we have shown how the
# performances of the tracker changes
# by employing or not a efficient management
# system. We measure a significant improvement (depending on
# the number of simulation steps, number of tracks and
# clutter rate) in the computation time in using EHM approach
# compared to the brute force JPDA. In the overall tracks
# we don't measure significant differences between the
# various trackers.

# %%
# References
# ----------
# [1] Maskell, S., Briers, M. and Wright, R., 2004, August. Fast mutual exclusion.
# In Signal and Data Processing of Small Targets 2004 (Vol. 5428, pp. 526-536).
# International Society for Optics and Photonics
# [2] Horridge, P. and Maskell, S., 2006, July. Real-time tracking of hundreds of
# targets with efficient exact JPDAF implementation. In 2006 9th International
# Conference on Information Fusion (pp. 1-8). IEEE
# [3] Maskell, S., 2003, July. Signal Processing with Reduced Combinatorial
# Complexity. Patent Reference:0315349.1
#
