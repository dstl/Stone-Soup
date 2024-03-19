#!/usr/bin/env python
# coding: utf-8

"""
==========================================
Extended Object Tracking Example (MTT-EOT)
==========================================
"""

# %%
# Extended object tracking (EOT) is a research topic that deals with tracking algorithms where
# the dimensions of the target have a non-negligible role in the detection generation.
# Advancements in the resolution capabilities of sensors, such as radars and lidars, allow the
# collection of more signals from the same object at the same instant, or direct detection of the
# shape (or part of) the target.
# This is important in many research areas such as self-driving vehicles.
#
# There are a plethora of definitions regarding EOT, in general we can summarise the main points
# as:
#
#   - there are multiple measurements coming from the same target at the same timestamp;
#   - the measurements can contain a target shape;
#   - the target state has information of the position, kinematics and shape;
#   - tracking algorithms can use clustered measurements or shape approximation to perform the
#     tracking.
#
# In literature [#]_, [#]_ there are several approaches for this problem and relevant applications
# to deal with various approximations and modelling.
# In this example, we consider the case where we collect multiple detections per target, the
# detections are sampled from an ellipse, used as an approximation of the target extent, then we
# use a simple clustering algorithm to identify the centroid of the distribution of detections
# that will be used for tracking.
#
# This example will consider multiple targets with a dynamic model of nearly constant velocity,
# a non-negligible clutter level, we do not consider any bounded environments (i.e., roads) for
# the targets, and we allow collisions to happen.
#
# This example follows this structure:
#
#   1. Describe the targets ground truths;
#   2. Collect the measurements from the targets;
#   3. Prepare the tracking algorithm and run the clustering for the detections;
#   4. Run the tracker and visualise the results.

# %%
# 1. Describe the targets ground truths;
# --------------------------------------
# We consider three targets moving with nearly constant velocity.
# The ground truth states include the metadata describing the size and orientation of the
# target. The target shape is described by the 'length' and 'width', semi-major and semi-minor
# axis of the ellipse, and the 'orientation'. The 'orientation' parameter is
# inferred by the state velocity components at each instant, while we assume that the size
# parameters are kept constant during the simulation.

# %%
# General imports
# ^^^^^^^^^^^^^^^
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import uniform, invwishart, multivariate_normal
from scipy.cluster.vq import kmeans

# %%
# Stone Soup components
# ^^^^^^^^^^^^^^^^^^^^^
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.state import GaussianState, State
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.detection import Detection, Clutter

# Simulation setup
start_time = datetime.now().replace(microsecond=0)
np.random.seed(1908)  # fix the seed
num_steps = 65  # simulation steps
rng = np.random.default_rng()  # random number generator for number of detections

# Define the transition model
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),
                                                          ConstantVelocity(0.05)])

# Instantiate the metadata and starting location for the targets
target_state1 = GaussianState(np.array([-50, 0.05, 70, 0.01]),
                              np.diag([5, 0.5, 5, 0.5]),
                              timestamp=start_time)

metadata_tg1 = {'length': 10,
                'width': 5,
                'orientation': np.arctan(
                    target_state1.state_vector[3]/target_state1.state_vector[1])}

target_state2 = GaussianState(np.array([0, 0.05, 20, 0.01]),
                              np.diag([5, 0.5, 5, 0.5]),
                              timestamp=start_time)

metadata_tg2 = {'length': 20,
                'width': 10,
                'orientation': np.arctan(
                    target_state2.state_vector[3]/target_state2.state_vector[1])}

target_state3 = GaussianState(np.array([50, 0.05, -30, 0.01]),
                              np.diag([5, 0.5, 5, 0.5]),
                              timestamp=start_time)

metadata_tg3 = {'length': 8,
                'width': 3,
                'orientation': np.arctan(
                    target_state3.state_vector[3]/target_state3.state_vector[1])}

# Collect the target and metadata states
targets = [target_state1, target_state2, target_state3]
metadatas = [metadata_tg1, metadata_tg2, metadata_tg3]

# ground truth sets
truths = set()

# loop over the targets
for itarget in range(len(targets)):

    # initialise the truth
    truth = GroundTruthPath(GroundTruthState(targets[itarget].state_vector,
                                             timestamp=start_time,
                                             metadata=metadatas[itarget]))

    for k in range(1, num_steps):  # loop over the timesteps
        # Evaluate the new state
        new_state = transition_model.function(truth[k-1],
                                              noise=True,
                                              time_interval=timedelta(seconds=5))

        # create a new dictionary from the old metadata and evaluate the new orientation
        new_metadata = {'length': truth[k - 1].metadata['length'],
                        'width': truth[k - 1].metadata['width'],
                        'orientation': np.arctan2(new_state[3], new_state[1])}

        truth.append(GroundTruthState(new_state,
                                      timestamp=start_time + timedelta(seconds=5*k),
                                      metadata=new_metadata))

    truths.add(truth)

# %%
# 2. Collect the measurements from the targets;
# ---------------------------------------------
# We have the trajectories of the targets, we can specify the measurement model. In this example
# we consider a :class:`~.LinearGaussian` measurement model. For this application we adopt a
# different approach from other examples, for each target state we create an oriented shape,
# centred in the ground-truth x-y location, and from it, we draw a number of points.
#
# In detail, at each timestep we evaluate the orientation of the ellipse from the velocity state
# of each target, then we randomly select between 1 and 10 points, assuming at least a detection
# per timestamp. The sampling of an elliptic distribution is done using an Inverse-Wishart
# distribution. We use these sampled points to generate target detections.
#
# We generate scans which contain both the detections from the targets and clutter measurements.

# Define the measurement model
measurement_model = LinearGaussian(ndim_state=4,
                                   mapping=(0, 2),
                                   noise_covar=np.diag([25, 25]))

# create a series of scans to collect the measurements and clutter
scans = []
for k in range(num_steps):
    measurement_set = set()

    # iterate for each case
    for truth in truths:

        # current state
        current = truth[k]

        # Identify how many detections to obtain
        sampling_points = rng.integers(low=1, high=10, size=1)

        # Centre of the distribution
        mean_centre = np.array([current.state_vector[0],
                                current.state_vector[2]])

        # covariance of the distribution
        covar = np.diag([current.metadata['length'], current.metadata['width']])

        # rotation matrix of the ellipse
        rotation = np.array([[np.cos(current.metadata['orientation']),
                              -np.sin(current.metadata['orientation'])],
                             [np.sin(current.metadata['orientation']),
                              np.cos(current.metadata['orientation'])]])

        rot_covar = np.dot(rotation, np.dot(covar, rotation.T))

        # use the elliptic covariance matrix
        covariance_matrix = invwishart.rvs(df=3, scale=rot_covar)

        # Sample points
        sample_point = np.atleast_2d(multivariate_normal.rvs(mean_centre,
                                                             covariance_matrix,
                                                             size=sampling_points))

        for ipoint in range(len(sample_point)):
            point = State(np.array([sample_point[ipoint, 0], current.state_vector[1],
                                    sample_point[ipoint, 1], current.state_vector[3]]))

            # Collect the measurement
            measurement = measurement_model.function(point, noise=True)

            # add the measurement on the measurement set
            measurement_set.add(Detection(state_vector=measurement,
                                          timestamp=current.timestamp,
                                          measurement_model=measurement_model))

        # Clutter detections
        truth_x = current.state_vector[0]
        truth_y = current.state_vector[2]

        for _ in range(np.random.poisson(5)):
            x = uniform.rvs(-50, 100)
            y = uniform.rvs(-50, 100)
            measurement_set.add(Clutter(np.array([[truth_x + x], [truth_y + y]]),
                                        timestamp=current.timestamp,
                                        measurement_model=measurement_model))

    scans.append(measurement_set)

# %%
# Visualise the tracks and the detections
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
from stonesoup.plotter import Plotterly

plotter = Plotterly()
plotter.plot_ground_truths(truths, [0, 2])
plotter.plot_measurements(scans, [0, 2])
plotter.fig

# %%
# 3. Prepare the tracking algorithm and run the detection clustering;
# -------------------------------------------------------------------
# It is time to prepare the tracking components, we use an :class:`~.ExtendedKalmanPredictor` and
# :class:`~.ExtendedKalmanUpdater` to perform the tracking. We consider a distance based data
# associator using :class:`~.GlobalNearestNeighbour`. We employ a
# :class:`~.MultiMeasurementInitiator` for initiating the tracks and a time based deleter using
# :class:`~.UpdateTimeStepsDeleter`.
#
# To process the cloud of detections generated at each timestep we use a K-means clustering method
# (where :math:`k=3`). This method identifies datapoints closer together (using the Euclidean
# distance measure) as elements belonging to the same cluster. From the identified clusters, we
# can obtain the centroid distribution that will be passed to the tracker as a unique detection.
#
# This simple example does not consider more refined clustering methods (e.g., inferring the
# number of clusters using the elbow method) or employing a different measurement model (which can
# include the shape of the target), but highlights a way to use the current implementation for EOT
# purposes.

# Load the tracking components
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater

# Initiator and deleter
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.initiator.simple import MultiMeasurementInitiator

# data associator
from stonesoup.dataassociator.neighbour import GlobalNearestNeighbour
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis

# Tracker

from stonesoup.tracker.simple import MultiTargetTracker
# Prepare the predictor and updater
predictor = ExtendedKalmanPredictor(transition_model)
updater = ExtendedKalmanUpdater(measurement_model)

# Instantiate the deleter
deleter = UpdateTimeStepsDeleter(3)

# Hypothesiser
hypothesiser = DistanceHypothesiser(
    predictor=predictor,
    updater=updater,
    measure=Mahalanobis(),
    missed_distance=10)

# Data associator
data_associator = GlobalNearestNeighbour(hypothesiser)

# Initiator
initiator = MultiMeasurementInitiator(
    prior_state=GaussianState(np.array([0, 0, 0, 0]),
                              np.diag([10, 0.5, 10, 0.5]) ** 2,
                              timestamp=start_time),
    measurement_model=None,
    deleter=deleter,
    data_associator=data_associator,
    updater=updater,
    min_points=7)

# %%
# Cluster the detections
# ^^^^^^^^^^^^^^^^^^^^^^
# Before running the tracker we employ a simple clustering method to identify neighbouring
# detections to the same target and identify a detection centroid, which will be used by the
# tracker.
# In this example, we assume known the number of targets and that number is not changing as the
# simulation evolves. More detailed approaches would solve these assumptions for a more general
# formulation.

# Create a list of detections and timestamps
centroid_detections = []
timestamps = []

for iscan in range(len(scans)):  # loop over the scans
    detections = []
    detection_set = set()
    # loop over the items of the scan, both detection and clutter
    for item in scans[iscan]:
        detections.append(np.array([item.state_vector[0], item.state_vector[1]]))

    # Find clusters in the data
    centroids, _ = kmeans(detections, 3)

    # iterate over the centroids to create a new set of detections
    for idet in range(len(centroids)):
        detection_set.add(Detection(state_vector=centroids[idet, :],
                                    timestamp=item.timestamp,
                                    measurement_model=item.measurement_model))

    # Now a new set of scans with the centroid detections
    centroid_detections.append(detection_set)
    timestamps.append(item.timestamp)

# prepare the tracker
tracker = MultiTargetTracker(
    initiator=initiator,
    deleter=deleter,
    data_associator=data_associator,
    updater=updater,
    detector=zip(timestamps, centroid_detections))


# %%
# 4. Run the tracker and visualise the results.
# ---------------------------------------------
# We can, finally, run the tracker and visualise the results.

tracks = set()
for time, current_tracks in tracker:
    tracks.update(current_tracks)

plotter.plot_measurements(centroid_detections, [0, 2], marker=dict(color='red'),
                          measurements_label='Cluster centroids')
plotter.plot_tracks(tracks, [0, 2])
plotter.fig

# %%
# Conclusion
# ----------
# In this example, we have presented a way to use Stone Soup components for extended object
# tracking. We have shown how to generate multiple detections from a single target by random
# sampling on the elliptic extent of the target. We have applied a clustering algorithm to
# identify the clouds of detections originating from the same target and use this information to
# perform the tracking.

# %%
# References
# ----------
# .. [#] Granström, Karl, and Marcus Baum. "A tutorial on multiple extended object tracking."
#        Authorea Preprints (2023).
# .. [#] Granström, Karl, Marcus Baum, and Stephan Reuter. "Extended object tracking: Introduction,
#        overview and applications." arXiv preprint arXiv:1604.00970 (2016).
