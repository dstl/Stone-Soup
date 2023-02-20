#!/usr/bin/env python
# coding: utf-8
"""
=================================================
11 - :math:`k`-d trees and TPR-trees
=================================================
"""

# %%
# :math:`k`-d trees and TPR-trees are both data structures which can be used to index objects in multi-dimensional
# space.
# 
# - A :math:`k`-d tree is a binary search tree that operates in :math:`k` dimensions where each node represents a
#   :math:`k`-dimensional point, where :math:`k` represents any number of dimensions. A :math:`k`-d tree is useful for
#   indexing the positions of **static** objects.
# - A TPR-tree (a time parameterised range tree) is a binary search tree which stores information about the object's
#   position, direction, and velocity in substructures called Velocity Bounding Rectangles (VBRs). This information can
#   be used to compute both current and predicted positions of the object. A TPR-tree is therefore useful for indexing
#   the positions of **moving** objects.
# 
# NB: a :math:`k`-d tree can also be used to index the positions of moving objects, but it will only show a snapshot of
# positions at a given timestep. The :math:`k`-d tree must be reconstructed at each time step, but the TPR tree only
# needs to be reconstructed if there are any changes to an object's direction and velocity.
# 
# Both trees can increase the efficiency of searches for information relating to a given object when compared with
# linear searches.



# %%
# Using :math:`k`-d trees for target tracking
# -------------------------------------------
# So far across the other tutorials, we have seen linear Nearest Neighbour and Probabilistic Data Association search
# algorithms applied to target tracking to identify which detections should be associated with our target track(s).
# 
# During training of a basic Nearest Neighbour algorithm with a given training set of detections from a sensor, the
# algorithm simply stores the detection vectors. When the algorithm is called to identify the nearest neighbours of a
# vector of interest (i.e. a track prediction), which we will refer to here as the **query vector**, the algorithm
# computes the distance between each detection in the training set and the query vector. It identifies the detection
# with the lowest distance from the query vector as the Nearest Neighbour.
# 
# This can be a suitable method to use for small datasets, however, as the number of detections and/or targets
# increases, linear algorithms quickly become inefficient. A more efficient alternative to linear searches is to
# conduct searches using a :math:`k`-d tree, which can help to quickly eliminate large sets of detections from the
# search rather than comparing every detection to the query vector.


# %%
# Constructing a :math:`k`-d tree
# -------------------------------
# During training of an algorithm that uses a :math:`k`-d tree, the tree itself will be constructed. Trees are
# constructed by splitting a set of detections from a sensor evenly into two groups, and then further splitting those
# two groups into two groups each, and continuing this process recursively down until each leaf node contains a
# determined number of detections in each leaf node (it can be 1 or more). These splits alternate in each dimension of
# the detection vectors in turn, then circle back to the first dimension to continue the splitting.
# 
# For example, with a set of 2-dimensional detection vectors of :math:`\begin{bmatrix} x \\ y \\\end{bmatrix}` we start
# constructing the tree from the root node by identifying the median value of the :math:`x^{th}`
# dimension of all detections. All detections with :math:`x` coordinate :math:`\leq` median are placed in the left
# branch from the root node, and all detections with :math:`x` coordinate :math:`>` median are placed in the right
# branch from the root node. The left and right child branches are then split into two further branches in the same way,
# except this time using the values in the :math:`y^{th}` dimension in place of the :math:`x^{th}`.
# The third split is done in the :math:`x^{th}` dimension again and so on until you achieve a determined number
# of points in each leaf node. Using a split at the median is just one common way to construct it.
#
# Generation of a :math:`k`-d tree has O(:math:`nlog^2n`) time.
#
# .. image:: ../../_static/kd_tree_fig_1.png
#   :width: 1100
#   :height: 250
#   :alt: Image showing diagrammatical representation of kd-tree in 2 dimensions, described further in figure 1 caption.
#
# Fig. 1a offers a graphical representation of the :math:`k`-d tree shown in 1b. :math:`d` represents the depth of
# the tree. The :math:`k`-d tree is constructed from a series of detections, which are 2-dimensional vectors, with
# splits alternating in each dimension: at :math:`d=0` the split occurs in the :math:`x`-axis, at :math:`d=1` the
# split occurs in the :math:`y`-axis etc. Colours of the splitting lines in 1a correspond with the splits at each
# depth shown in 1b.


# %%
# Searching a :math:`k`-d tree with Nearest Neighbour
# ---------------------------------------------------
# Once the :math:`k`-d tree has been constructed from the detection set, it can then be used to search for the Nearest
# Neighbour of a given track prediction (which is the query vector).
# 
# The method for searching the tree is similar to the method for building the tree: starting at the root node, the
# :math:`x^{th}` dimension of the query vector is compared with :math:`x^{th}` dimension of the root
# node vector, if the value of the query vector is less than or equal to the root node detection vector, the search
# moves down the left subtree, and down the right subtree if the value is greater. This process is repeated recursively
# at each internal node, again alternating through the vector dimensions, until a leaf node is reached. At each node,
# the distance between the query vector and the node vector is measured, and if the distance is less than the current
# best, the node vector becomes the new current best nearest neighbour.
# 
# Once the search reaches a leaf node, the search continues back up through the tree to ensure the best nearest
# neighbour has been identified. From the leaf node, it will traverse back up to the parent node. If the distance from
# the query vector and the split plane (shown by the blue lines in Fig. 2a) of the parent node is less than the current
# best, the search will proceed down the sibling subtree as above. If the distance is greater than the current best, the
# search will proceed up to the next parent node. The search continues this way recursively up the tree until we return
# to the root node.
#
# .. image:: ../../_static/kd_tree_fig_2.png
#   :width: 1200
#   :height: 250
#   :alt: Image showing diagrammatical representation of kd-tree in 2 dimensions, described further in figure 2 caption.
#
# Fig. 2a offers a graphical representation of the :math:`k`-d tree shown in 2b. 2a shows the query vector,
# :math:`[8,2]`, in blue. 2b shows the first part of the search taking place down the :math:`k`-d tree, with the text
# on the right indicating the direction of the search at each depth. Current best indicates the current best distance of
# any detection vector in the tree from the query vector. Once reaching the leaf node, the search continues back up the
# tree. In this case, there are no sibling subtrees with a distance from the query vector less than the current best.
# The detection at :math:`[9,1]` is selected as the nearest neighbour.


# %%
# A Nearest Neighbour algorithm using a :math:`k`-d tree can be faster than linear Nearest Neighbour searches because it
# does not compare every detection vector in the set to the query vector, but implicitly eliminates points that cannot
# be closer in distance than the current best. It has an average O(:math:`log(n)`) search time with a worst case of
# O(:math:`n`) given a set of :math:`n` detections.
#
# .. image:: ../../_static/kd_tree_fig_3.png
#   :width: 1200
#   :height: 250
#   :alt: Image showing diagrammatical representation of kd-tree in 2 dimensions, described further in figure 3 caption.
#
# Fig. 3 demonstrates a situation where it is useful to traverse back up the tree to complete the search for the
# nearest neighbour. Here, our query vector is :math:`[9,6]`. If we stop the search at the first leaf node we reach,
# which is the detection at :math:`[8,8]` with a current best of 3, we miss the true nearest neighbour to our query
# vector, which is :math:`[9,4]` with a current best of 2. The red arrow indicates the minimum distance from the split
# plane of the parent node of the detection at :math:`[8,8]`. The minimum distance is 1. This is less than our current
# best (3), which indicates that the search should continue down the sister sub-tree, thus leading us to the detection
# at :math:`[9,4]`.


# %%
# :math:`k`-d and TPR-trees can both be searched with either Nearest Neighbour or Probabilistic Data Association
# algorithms. Using the tree will increase the efficiency of searches where there are more targets and/or detections.


# %%
# Example of Global Nearest Neighbour search using :math:`k`-d tree
# -----------------------------------------------------------------
# In this example, we will be calculating the average run time of the Global Nearest Neighbour (GNN) data association
# algorithm with :math:`k`-d tree vs with linear GNN search to compare efficiency of the search. First we will run the
# search with the :math:`k`-d tree.


# %%
# Simulate ground truth
# ^^^^^^^^^^^^^^^^^^^^^
# We will simulate a large number of targets moving in the :math:`x`, :math:`y` Cartesian plane. We then add truth
# detections with a high clutter rate at each time-step.

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
# Build :math:`k`-d tree
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
    start_time = timer.time()

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

    end_time = timer.time()
    run_time = end_time - start_time

    run_times_KDTree.append(run_time)

# %%
# Plot the resulting tracks - :math:`k`-d-tree
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from stonesoup.plotter import Plotterly

plotter = Plotterly()
plotter.plot_ground_truths(groundtruth, mapping=[0, 2])
plotter.plot_measurements(detections, mapping=[0, 2])
plotter.plot_tracks(tracks, mapping=[0, 2])
plotter.fig

# %%
# Compare with linear search
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# Now, to compare computing time, we will run a linear Global Nearest Neighbour search.

# set up GNN data associator
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
GNN_data_associator = GNNWith2DAssignment(hypothesiser=hypothesiser)

run_times_GNN = []

# run loop to calculate average run time
for _ in range(0, 3):
    start_time = timer.time()

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

    end_time = timer.time()
    run_time = end_time - start_time

    run_times_GNN.append(run_time)

# %%
# Plot the resulting tracks - GNN
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from stonesoup.plotter import Plotterly

plotter = Plotterly()
plotter.plot_ground_truths(groundtruth, mapping=[0, 2])
plotter.plot_measurements(detections, mapping=[0, 2])
plotter.plot_tracks(tracks, mapping=[0, 2])
plotter.fig

# %%
# Now we compare average run times of a Global Nearest Neighbour search with a :math:`k`-d tree vs linear search.

from statistics import median

print(f'Median run time of k-d tree search: {round(median(run_times_KDTree), 4)} seconds')
print(f'Median run time of linear search: {round(median(run_times_GNN), 4)} seconds')
print(f'\nThe GNN search runs {round(median(run_times_GNN) / median(run_times_KDTree), 2)} '
      f'faster on average with kd-tree than with linear search')


