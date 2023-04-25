#!/usr/bin/env python
# coding: utf-8
"""
=================================================
:math:`k`-d trees and TPR-trees
=================================================
"""

# %%
# :math:`k`-d trees and TPR-trees are both data structures which can be used to index objects in multi-dimensional
# space.
#
# - A :math:`k`-d tree is an indexing tree structure that operates in :math:`k` dimensions. Each node represents a
#   :math:`k`-dimensional point, where :math:`k` represents any number of dimensions. A :math:`k`-d tree is useful for
#   indexing the positions of **static** objects.
# - A TPR-tree (a time parameterised range tree) is an indexing tree structure that can be used to store and query
#   information about **moving** objects. The tree stores functions of time that can be used to return the current or
#   predicted future positions of an object. The parameters of these functions are the position and velocity vector of
#   the object at a given time point.
#
#   The TPR-tree does not need to be updated at every time step as it contains the information to predict future
#   positions of an object. This saves computing power. To maintain accuracy, however, the TPR-tree should be updated
#   with new position and velocity vector information at the appropriate frequency. The frequency of updates will depend
#   on the desired accuracy and the movement exhibited by object(s) stored in the tree. E.g. frequent changes in the
#   velocity or direction of an object would require more frequent updates to the tree to maintain higher accuracy.
#
# A :math:`k`-d tree can also be used to index the positions of moving objects, but it will only show a snapshot of
# positions at a given time step. Unlike the TPR-tree, the :math:`k`-d tree must be reconstructed at each time step.


# %%
# Using :math:`k`-d trees or TPR trees for target tracking
# --------------------------------------------------------
# So far across the other tutorials, we have seen Nearest Neighbour (NN) and Probabilistic Data Association (PDA)
# algorithms applied linearly to target tracking to identify which detections should be associated with our target
# track(s). NN algorithms associate predictions with detections based on distance, and PDA algorithms associate them
# based on probability (see tutorials 5-8 for more details). Using a tree in combination with one of these algorithms
# can make the Data Association process much faster.
#
# During Data Association without a tree, distance or probability of association must be calculated between the
# prediction and *every* detection in order to select the best detection for association. These linear searches can be a
# suitable method to use for small sets of detections and small numbers of tracks but become inefficient as the number
# of detections and/or targets increases substantially.
#
# NN or PDA algorithms can be used in combination with trees to perform data association. Unlike linear searches,
# searches with :math:`k`-d trees or TPR trees do not have to compare every detection vector in the set to the
# prediction vector, and instead can quickly eliminate large sets of detections implicitly that cannot be better than
# the current best association. :math:`k`-d trees have an average O(:math:`log(n)`) search time with a worst case of
# O(:math:`n`) given a set of :math:`n` detections.
#
# We will look into detail at how :math:`k`-d trees are constructed and queried below.


# %%
# Constructing a :math:`k`-d tree
# -------------------------------
# :math:`k`-d trees are constructed by splitting a set of detections into two even groups, and then
# further splitting those two groups into two groups each, continuing this process recursively down until each leaf
# node contains the desired number of detections (it can be 1 or more). These splits alternate in each dimension of
# the detection vectors in turn (Fig. 1).
#
# For example, with a set of 2-dimensional detection vectors of :math:`\begin{bmatrix} x \\ y \\\end{bmatrix}` we start
# constructing the tree from the root node by identifying the median value of the :math:`x^{th}` dimension of all
# detections. All detections with :math:`x` coordinate :math:`\leq` the median are placed in the left subtree from the
# root node, and all detections with :math:`x` coordinate :math:`>` median are placed in the right subtree from the root
# node. The left and right subtrees are then split into two further subtrees in the same way, except this time using the
# values in the :math:`y^{th}` dimension in place of the :math:`x^{th}`. The third split is done in the :math:`x^{th}`
# dimension again and so on until you achieve a determined number of points in each leaf node. Using a split at the
# median is just one common way to construct it.
#
# Generation of a :math:`k`-d tree has O(:math:`nlog^2n`) time.
#
# .. image:: ../../_static/kd_tree_fig_1.png
#   :width: 1100
#   :height: 250
#   :alt: Image showing diagrammatical representation of kd-tree in 2 dimensions, described further in figure 1 caption.
#
# Fig. 1a offers a graphical representation of the :math:`k`-d tree shown in 1b. :math:`d` represents the depth of
# the tree. The :math:`k`-d tree is constructed from a series of detections, which in this case are 2-dimensional
# vectors, with splits alternating in each dimension. At :math:`d=0` the split occurs in the :math:`x`-axis, at
# :math:`d=1` the split occurs in the :math:`y`-axis etc. Colours of the splitting planes in 1a correspond with the
# splits at each depth shown in 1b.
#
# In comparison, entries in TPR tree leaf nodes are pairs of a moving object's position and an ID for the object. The
# internal nodes of the tree are bounding rectangles which bound the positions of all moving objects (or other bounding
# rectangles) in that subtree. The bounding rectangles' coordinates are functions of time, so they are capable of
# following the enclosed data points as they move over time.


# %%
# Searching a :math:`k`-d tree with Nearest Neighbour
# ---------------------------------------------------
# Once the :math:`k`-d tree has been constructed from the detection set, it can then be used for data association. In
# this example we will search for the detection that is the nearest neighbour of a given prediction.
#
# The method for searching the tree is similar to the method for building the tree: starting at the root node, the
# :math:`x^{th}` dimension of the prediction is compared with the :math:`x^{th}` dimension of the root node detection.
# If the value of the prediction's :math:`x` dimension is less than or equal to that of the root node detection, the
# search moves down the left subtree, and down the right subtree if the value is greater. This process is repeated
# recursively at each internal node, again alternating through the vector dimensions, until a leaf node is reached.
# Because we are conducting a Nearest Neighbour search, at each node the Euclidean distance between the prediction and
# the detection at that node is measured. If the distance is less than the current best distance, the detection at that
# node becomes the new current best nearest neighbour.
#
# To ensure that the true nearest neighbour has been identified, once we reach a leaf node the search must continue back
# up through the tree to ensure we haven't skipped a potential nearest neighbour that lies in an over-looked sub-space.
# As the search moves back up the tree, we should at each level consider if we should go down the sibling subtree.
#
# .. image:: ../../_static/kd_tree_fig_2.png
#   :width: 1200
#   :height: 250
#   :alt: Image showing diagrammatical representation of kd-tree in 2 dimensions, described further in figure 2 caption.
#
# Fig. 2a offers a graphical representation of the :math:`k`-d tree shown in 2b. 2a shows the prediction vector,
# :math:`[8,2]`, in blue. 2b shows the first part of the search taking place down the :math:`k`-d tree, with the text
# on the right indicating the direction of the search at each depth. Current best indicates the current best distance of
# any detection vector in the tree from the prediction. Once reaching the leaf node, the search continues back up the
# tree. In this case, there are no sibling subtrees with a distance from the prediction vector less than the current
# best so the detection at :math:`[9,1]` is identified as the nearest neighbour.


# %%
# To decide whether to explore the sibling subtree of a subtree that has already been searched, we consider the closest
# possible detection to our prediction that could theoretically lie in that unexplored subtree. Looking at Figure 3 for
# example, after traversing completely down the tree, our current best distance is 2.24 units for detection
# :math:`[7,5]`. The shortest distance between the prediction (blue) and the splitting plane on :math:`[7,5]` (where
# :math:`y=5`) is 1 unit, as shown by the red arrow in Figure 3.
#
# This means that an unexplored point that lies just beyond the splitting plane could have a distance from the
# prediction ofbetween 1 and 2.24 units, i.e. less than our current best of 2.24. For this reason, we determine that we
# must continue the search down the left subtree from detection :math:`[7,5]`. Indeed, there is a detection at
# :math:`[9,4]` in this subtree which has a distance from our prediction vector of 2 units, which is less than our
# current best distance so becomes the new current best. We then traverse back up the tree again. As there are no other
# subtrees which can possibly contain detections that beat the current best distance, the detection at :math:`[9,4]`
# is selected as the nearest neighbour and the search is complete.
#
# .. image:: ../../_static/kd_tree_fig_3.png
#   :width: 1200
#   :height: 250
#   :alt: Image showing diagrammatical representation of kd-tree in 2 dimensions, described further in figure 3 caption.
#
# Fig. 3 demonstrates a situation where it is useful to traverse back up the tree to complete the search for the
# nearest neighbour.


# %%
# As a general rule for traversing back up the tree, if the shortest distance between the prediction and the splitting
# plane of the parent node is less than the current best, the search will proceed down the sibling subtree. If the
# distance is greater than the current best, the search will continue up to the next node. The search continues this way
# recursively up the tree until we return to the root node and have considered whether to search every subtree.
#
# :math:`k`-d and TPR-trees can both be searched with either Nearest Neighbour or Probabilistic Data Association
# algorithms. Using the tree will increase the efficiency of searches where there are more targets and/or detections.


# %%
# Example of Global Nearest Neighbour search using :math:`k`-d tree and TPR tree
# -----------------------------------------------------------------
# In this example, we will be calculating the average run time of the Global Nearest Neighbour (GNN) data association
# algorithm with a :math:`k`-d tree and with a TPR tree and comparing the results with a linear GNN search.
# First, we will run the search with a :math:`k`-d tree.


# %%
# Simulate ground truth
# ^^^^^^^^^^^^^^^^^^^^^
# We will simulate a large number of targets moving in the :math:`x`, :math:`y` Cartesian plane. We will then add truth
# detections with a high clutter rate at each time step.

import numpy as np
import datetime

from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import GaussianState

initial_state_mean = StateVector([[0], [0], [0], [0]])
initial_state_covariance = CovarianceMatrix(np.diag([4, 0.5, 4, 0.5]))
timestep_size = datetime.timedelta(seconds=5)
number_of_steps = 25
birth_rate = 0.3
death_probability = 0.05
initial_state = GaussianState(initial_state_mean, initial_state_covariance)

from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, ConstantVelocity)

transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.05), ConstantVelocity(0.05)])

from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator

groundtruth_sim = MultiTargetGroundTruthSimulator(
    transition_model=transition_model,
    initial_state=initial_state,
    timestep=timestep_size,
    number_steps=number_of_steps,
    birth_rate=birth_rate,
    death_probability=death_probability,
    initial_number_targets=200
)

# %%
# Initialise the measurement models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from stonesoup.simulator.simple import SimpleDetectionSimulator
from stonesoup.models.measurement.linear import LinearGaussian

# initialise the measurement model
measurement_model_covariance = np.diag([0.25, 0.25])
measurement_model = LinearGaussian(4, [0, 2], measurement_model_covariance)

# probability of detection
probability_detection = 0.9

# clutter will be generated uniformly in this are around the target
clutter_area = np.array([[-1, 1], [-1, 1]]) * 500
clutter_rate = 50

detection_sim = SimpleDetectionSimulator(
    groundtruth=groundtruth_sim,
    measurement_model=measurement_model,
    detection_probability=probability_detection,
    meas_range=clutter_area,
    clutter_rate=clutter_rate
)

# %%
# Import tracker components
# ^^^^^^^^^^^^^^^^^^^^^^^^^

# tracker predictor and updater
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater

# initiator and deleter
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.initiator.simple import MultiMeasurementInitiator

# tracker
from stonesoup.tracker.simple import MultiTargetTracker

# timer for comparing run time of k-d tree algorithm with linear search
import time as timer

# %%
# 1. Run the search with a :math:`k`-d tree
# ^^^^^^^^^^^^^^^^^^^^^^
# Set up the :math:`k`-d tree to operate with the GNN algorithm to identify the most likely nearest neighbours to each
# track globally.

# create predictor and updater
predictor = KalmanPredictor(transition_model)
updater = KalmanUpdater(measurement_model)

# create hypothesiser
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis

hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=3)

# create KD-tree data associator
from stonesoup.dataassociator.tree import DetectionKDTreeGNN2D

KD_data_associator = DetectionKDTreeGNN2D(hypothesiser=hypothesiser,
                                          predictor=predictor,
                                          updater=updater,
                                          number_of_neighbours=3,
                                          max_distance_covariance_multiplier=3)

# %%
# Create tracker and run it
# ^^^^^^^^^^^^^^^^^^^^^^^^^

run_times_KDTree = []

# run loop to calculate average run time
for _ in range(0, 3):
    start_time = timer.perf_counter()

    # create tracker components
    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)

    # create deleter
    covariance_limit_for_delete = 100
    deleter = CovarianceBasedDeleter(covar_trace_thresh=covariance_limit_for_delete)

    # create initiator
    s_prior_state = GaussianState([[0], [0], [0], [0]], np.diag([0, 0.5, 0, 0.5]))
    min_detections = 3

    initiator = MultiMeasurementInitiator(
        prior_state=s_prior_state,
        measurement_model=measurement_model,
        deleter=deleter,
        data_associator=KD_data_associator,
        updater=updater,
        min_points=min_detections
    )

    # create tracker
    tracker = MultiTargetTracker(
        initiator=initiator,
        deleter=deleter,
        detector=detection_sim,
        data_associator=KD_data_associator,
        updater=updater
    )

    # run tracker
    groundtruth = set()
    detections = set()
    tracks = set()

    for time, ctracks in tracker:
        groundtruth.update(groundtruth_sim.groundtruth_paths)
        detections.update(detection_sim.detections)
        tracks.update(ctracks)

    end_time = timer.perf_counter()
    run_time = end_time - start_time

    run_times_KDTree.append(run_time)

# %%
# Plot the resulting tracks:
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

from stonesoup.plotter import Plotterly

plotter = Plotterly()
plotter.plot_ground_truths(groundtruth, mapping=[0, 2])
plotter.plot_measurements(detections, mapping=[0, 2])
plotter.plot_tracks(tracks, mapping=[0, 2])
plotter.fig

# %%
# 2. Run the search with a TPR tree
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Now we will construct a TPR tree to operate with the same GNN algorithm to identify the most likely nearest neighbours
# to each track globally to compare run times.

# create the TPR tree data associator
from stonesoup.dataassociator.tree import TPRTreeGNN2D

TPR_data_associator = TPRTreeGNN2D(hypothesiser=hypothesiser,
                                   measurement_model=measurement_model,
                                   horizon_time=datetime.timedelta(seconds=10))

run_times_TPRTree = []

for _ in range(0, 3):
    start_time = timer.perf_counter()

    TPR_data_associator = TPRTreeGNN2D(hypothesiser=hypothesiser,
                                       measurement_model=measurement_model,
                                       horizon_time=datetime.timedelta(seconds=10))

    # create predictor and updater
    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)

    # create deleter
    covariance_limit_for_delete = 100
    deleter = CovarianceBasedDeleter(covar_trace_thresh=covariance_limit_for_delete)

    # create initiator
    s_prior_state = GaussianState([[0], [0], [0], [0]], np.diag([0, 0.5, 0, 0.5]))
    min_detections = 3

    initiator = MultiMeasurementInitiator(
        prior_state=s_prior_state,
        measurement_model=measurement_model,
        deleter=deleter,
        data_associator=TPR_data_associator,
        updater=updater,
        min_points=min_detections
    )

    # create tracker
    tracker = MultiTargetTracker(
        initiator=initiator,
        deleter=deleter,
        detector=detection_sim,
        data_associator=TPR_data_associator,
        updater=updater
    )

    # run tracker
    groundtruth = set()
    detections = set()
    tracks = set()

    for time, ctracks in tracker:
        groundtruth.update(groundtruth_sim.groundtruth_paths)
        detections.update(detection_sim.detections)
        tracks.update(ctracks)

    end_time = timer.perf_counter()
    run_time = end_time - start_time

    run_times_TPRTree.append(run_time)

# %%
# Plot the resulting tracks:

from stonesoup.plotter import Plotterly

plotter = Plotterly()
plotter.plot_ground_truths(groundtruth, mapping=[0, 2])
plotter.plot_measurements(detections, mapping=[0, 2])
plotter.plot_tracks(tracks, mapping=[0, 2])
plotter.fig

# %%
# 3. Run the search with a linear search
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Now, to compare computing time, we will run a linear Global Nearest Neighbour search.

# set up GNN data associator
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
GNN_data_associator = GNNWith2DAssignment(hypothesiser=hypothesiser)

run_times_GNN = []

# run loop to calculate average run time
for _ in range(0, 3):
    start_time = timer.perf_counter()

    # create tracker components
    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)

    # create deleter
    covariance_limit_for_delete = 100
    deleter = CovarianceBasedDeleter(covar_trace_thresh=covariance_limit_for_delete)

    s_prior_state = GaussianState([[0], [0], [0], [0]], np.diag([0, 0.5, 0, 0.5]))
    min_detections = 3

    # create initiator
    initiator = MultiMeasurementInitiator(
        prior_state=s_prior_state,
        measurement_model=measurement_model,
        deleter=deleter,
        data_associator=GNN_data_associator,
        updater=updater,
        min_points=min_detections
    )

    # create tracker
    tracker = MultiTargetTracker(
        initiator=initiator,
        deleter=deleter,
        detector=detection_sim,
        data_associator=GNN_data_associator,
        updater=updater
    )

    # run tracker
    groundtruth = set()
    detections = set()
    tracks = set()

    for time, ctracks in tracker:
        groundtruth.update(groundtruth_sim.groundtruth_paths)
        detections.update(detection_sim.detections)
        tracks.update(ctracks)

    end_time = timer.perf_counter()
    run_time = end_time - start_time

    run_times_GNN.append(run_time)

# %%
# Plot the resulting tracks:
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

from stonesoup.plotter import Plotterly

plotter = Plotterly()
plotter.plot_ground_truths(groundtruth, mapping=[0, 2])
plotter.plot_measurements(detections, mapping=[0, 2])
plotter.plot_tracks(tracks, mapping=[0, 2])
plotter.fig

# %%
# Compare search run times
from statistics import median

print(f'Median run time of k-d tree search: {round(median(run_times_KDTree), 4)} seconds')
print(f'Median run time of TPR tree search: {round(median(run_times_TPRTree), 4)} seconds')
print(f'Median run time of linear search: {round(median(run_times_GNN), 4)} seconds')
print(f'\nThe GNN search runs {round(median(run_times_GNN)/median(run_times_KDTree), 2)} times faster on average with '
      f'kd-tree than with linear search')

# %%
# In the tracking situation chosen for this example, where we have 200 targets and relatively high probability of
# detection (90%), the :math:`k`-d tree outperforms the TPR tree, and both outperform the linear search in terms of
# computing time. The TPR tree should outperform the :math:`k`-d tree in tracking situations with a significantly higher
# number of targets and a lower probability of detection.
#
# You can change these parameters in the code above and see how the run times of the GNN algorithm are affected by the
# different data indexing structures demonstrated.
