#!/usr/bin/env python
"""
Tracking cell data from intracellular infection assays
======================================================
"""

# %%
# Introduction
# ------------
# Tracking cellular movement within biological imagery is a complex task. In
# this demo we are using Stone Soup to track cells in confocal transmitted light
# imagery. Utilising tracking to be able to analyse single cell data over time
# from frame based population data increases the utility of the data set. It
# allows for the characteristics of individual cells to be monitored throughout
# the time course, providing an additional dimension to the data set for
# analyses. This demo shows how Stone Soup can be utilised for tracking of the
# cells. A number of packages in Python are available to generate morphological
# metrics from the segmented datasets.
#
# In this notebook we will load a csv file containing segmented cell data and metadata, to use as
# detections in the Stone Soup tracker. We will build the tracker and use the Optuna framework to 
# optimise the tracker over a number of parameters. We will display the track output in a video.

# %%
# Software Dependencies
# ---------------------
# A number of installations are required for this demonstration.
#
# Cellpose
# ~~~~~~~~
# Cellpose is a deep learning-based algorithm for segmentation of cells and nuclei. There is also a
# human-in-the-loop option for training the model (v2.0). Cellpose is used to produce the csv of
# segmented cell data used by the tracker. Installation instructions can be found
# `here <https://github.com/MouseLand/cellpose/blob/main/README.md/#Installation>`__.
#
# Optuna
# ~~~~~~
# Optuna is a hyperparameter optimisation software. It can be used to explore the parameter space
# and find the optimal combination of parameters based on pre-defined metrics. Optuna is framework
# agnostic, so is easy to implement on any algorithm framework. Further details, including 
# installation instructions, can be found
# `here <https://optuna.readthedocs.io/en/stable/index.html>`__.

# %%
# Data
# ----
# In this demo cell time lapse imagery was captured using confocal microscope time lapse imaging,
# using a Zeiss LSM 710 microscope. This produced proprietary .czi files for analysis. A number of
# processing steps were performed before tracking algorithms were implemented. These steps are as
# follows:
#
# 1. Unpack the .czi file into individual frames and image channels (Brightfield and fluorescence) using the aicspylibczi library. 
# 2. Segment the images using Cellpose. To optimise this a range of different parameters were applied to a random set of the frames and these were visualised to select the best Cellpose parameters.
# 3. Use best Cellpose parameters to segment all images within a time lapse experiment.
# 4. From this the segmentations were used to derive population statistics using the sci-imaging library. 
# 5. The centroid_X and centroid_Y values of the bounding box for each cells were used in Stone Soup as the location of each individual cell.
#
# It is important to note that the extraction of time lapse frames will differ depending on the
# confocal imaging approach used. However, any microscope and segmentation approach could be
# utilised as long as within the end result you have the centroid_X and centroid_Y value to then
# input into the Stone Soup tracking algorithms.  

# set the csv filename to track
csvfile = 'filtered_cell_stats_Assay 3.csv'
# set the relevant position to track in
position = 0
# set the time range (in units of frames)
tMin = 15    #inclusive
tMax = 45   #inclusive

csv_subset = f'{csvfile[0:len(csvfile)-4]}_p{position}_t{tMin}-{tMax}_pre-track.csv'

# %%
# The resulting data can be visualised in the video below, where the white lines show the outlines
# of the segmented cells.
#
# .. raw:: html
#
#     <video autoplay loop controls>
#       <source src="../_static/cell_demo_0.mp4" type="video/mp4">
#     </video>
#

# %%
# Tracking Algorithm
# ------------------
import numpy as np
from stonesoup.reader.generic import CSVDetectionReader
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, RandomWalk, ConstantVelocity)
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.types.state import GaussianState
from stonesoup.tracker.simple import MultiTargetTracker
from stonesoup.types.update import Update

# %%
# We will set up the tracking algorithm to use cell position (x and y coordinates), median
# intensity (a measure of the extent to which the infection has spread in the cell) and cell area
# as the state vector fields. Initially, we estimate the associated parameters (change in the state
# vector fields and error in those change values) based on prior knowledge of reasonable cell
# behaviour. The initiator's minimum number of points and the deleter's number of points are also
# estimated.
dele = 5
init = 4
change_pos = 5 
change_area = 100
change_intensity = 2
error_pos = 4
error_area = 300
error_intensity = 5

# %%
# Using these estimated parameters, we run the algorithm.
#
# - **Detector** - Since the segmented cell data is in a simple CSV format :class:`~.CSVDetectionReader` is used to make the detector. The column names of the state vector fields are included as a list, and the column name of the time field is also specified.
# - **Transition Model** - A :class:`~.CombinedLinearGaussianTransitionModel` is used. The position, split into x and y coordinates, and area are assumed to change according to a :class:'~.RandomWalk`. The intensity, or spread of infection within the cells, is assumed to change with :class:`~.ConstantVelocity`.
# - **Measurement Model** - A :class:`~.LinearGaussian` measurement model is used. The state vector has five dimensions - one each for the x position, y position and area and two for the intensity.
# - **Predictor and Updater** - The tracker is made using a Kalman filter, with the :class:`~.KalmanPredictor` and :class:`~.KalmanUpdater` classes being used for the predictor and updater respectively.
# - **Hypothesiser** - The :class:`~.DistanceHypothesiser` is used to generate the hypothesis pairs of detections and predicted measurements. This uses the :class:`~.Mahalanobis` distance as the measure of the quality of these pairs. 
# - **Data Associator** - The Global Nearest Neighbour algorithm is used as the data associator to pick the best hypothesis pair.
# - **Deleter** - An :class:`~.UpdateTimeStepsDeleter` is used to delete tracks that have not been seen in the last 5 frames.
# - **Initiator** - A :class:`~.MultiMeasurementInitiator` is used to add tracks that have been seen in the last 4 frames. Within this there is a deleter set to 3, so the potential tracks are deleted if they are not seen in the last 3 frames.
detector = CSVDetectionReader(f'{csv_subset}',
                              state_vector_fields=("centroid_X", "centroid_Y", "median_intensity", "area"),
                              time_field="Timepoint",
                              timestamp=True)

transition_model = CombinedLinearGaussianTransitionModel((RandomWalk(change_pos), RandomWalk(change_pos),
                                                          ConstantVelocity(change_intensity),
                                                          RandomWalk(change_area)))

measurement_model = LinearGaussian(ndim_state=5,
                                   mapping=[0, 1, 2, 4],
                                   noise_covar=np.diag([error_pos**2, error_pos**2,
                                                        error_intensity**2, error_area**2]))

predictor = KalmanPredictor(transition_model)

updater = KalmanUpdater(measurement_model)

measure = Mahalanobis(mapping=[0, 1, 2, 3])

hypothesiser = DistanceHypothesiser(predictor,
                                    updater,
                                    measure,
                                    missed_distance=2)

data_associator = GNNWith2DAssignment(hypothesiser)

deleter = UpdateTimeStepsDeleter(dele,
                                 delete_last_pred=True)

initiator = MultiMeasurementInitiator(GaussianState(np.array([[0], [0], [0], [.5], [0]]),
                                                    np.diag([15**2, 15**2, 15**2, 20**2, 100**2])),
                                      min_points=init,
                                      deleter=UpdateTimeStepsDeleter(3, delete_last_pred=True),
                                      measurement_model=measurement_model,
                                      data_associator=data_associator,
                                      updater=updater)

tracker = MultiTargetTracker(initiator=initiator,
                             deleter=deleter,
                             detector=detector,
                             data_associator=data_associator,
                             updater=updater,)

# %%
# Run the tracker

# initialize variables
tracks = set()
detections = set()

# go through each frame
for step, (time, current_tracks) in enumerate(tracker, 1):
    # update track list
    tracks.update(current_tracks)
    
    # update list of detected cells
    detections.update(tracker.detector.detections)
    
    # detections that are part of a track
    tracked_detections = {track.hypothesis.measurement for track in current_tracks if isinstance(track.state,Update)}
    
    # total detections
    current_detections = tracker.detector.detections

# %%
# Below is a video showing the resulting tracks, where the yellow outlines show the tracked cells and
# the white outlines show the cells that have been segmented but are not in tracks.
#
# .. raw:: html
#
#     <video autoplay loop controls>
#       <source src="../_static/cell_demo_1.mp4" type="video/mp4">
#     </video>
#

# %%
# Optimizer
# ---------
# Next, we implement the Optuna optimizer on the Stone Soup algorithm. To do this, we first have to
# define an objective function. This is done through the following steps:
#
# 1) Set the parameters to vary the values of. In this example all of the the parameters being changed are integers so the suggest_int function is used. The numbers passed in as arguments of the function are the minimum and maximum values that the parameter can take (inclusive).
# 2) Run the algorithm. This is done as above, running through each frame to generate the tracks.
# 3) Define metric(s) for optimisation. These are meant to be representative of the quality of the tracks, such that maximising or minimising these metrics will improve the tracking algorithm. The objective function returns the metric(s) defined for optimisation. In this case, these are the number of long tracks, defined as the number of tracks spanning at least 90% of the frames, and the total number of tracks. These are standardised to account for the difference in size of the two values, so that the multi-objective optimisation isn't weighted towards the metric that is the larger value (total number of tracks). Single-objective optimisation is also possible in Optuna, and indeed is the more widely used optimisation method. However, as there is no group truth for this problem, multi-objective optimisation was preferred in this case.
#
# While only integer values are changed in this demonstration, Optuna
# can also be used to vary a range of other types of variables. See `here <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html>`__
# for more information.
#
# Also, it is important to note that the default optimisation approach that Optuna uses is Bayesian
# optimisation, which works by randomly selecting the parameters for the first n trials (where n is
# typically around 50) and then subsequently uses these values to determine which area of the
# parameter space is best to explore. In this demo, the number of trials is kept low (20) to reduce
# run time, but for best results a higher number of trials should be used (e.g. >100).
import optuna

def objective(trial):
    
    # 1) set parameters to change:
    dele = trial.suggest_int('dele', 4, 11)
    init = trial.suggest_int('init', 4, 11)
    change_pos = trial.suggest_int('change_pos', 4, 15) #how much position is expected to change between frames
    change_area = trial.suggest_int('change_area', 75, 175) #how much the area is expected to change
    change_intensity = trial.suggest_int('change_intensity', 2, 10) #how much the intensity is expected to change
    error_pos = trial.suggest_int('error_pos', 3, 8) #possible error in change_pos
    error_area = trial.suggest_int('error_area', 200, 400) #possible error in change_area
    error_intensity = trial.suggest_int('error_intensity', 4, 10) #possible error in change_intensity
    
    # 2) run algorithm:
    detector = CSVDetectionReader(f'{csv_subset}',
                                  state_vector_fields=("centroid_X", "centroid_Y", "median_intensity", "area"),
                                  time_field="Timepoint",
                                  timestamp=True)

    transition_model = CombinedLinearGaussianTransitionModel((RandomWalk(change_pos), RandomWalk(change_pos),
                                                              ConstantVelocity(change_intensity),
                                                              RandomWalk(change_area)))

    measurement_model = LinearGaussian(ndim_state=5,
                                       mapping=[0, 1, 2, 4],
                                       noise_covar=np.diag([error_pos**2, error_pos**2,
                                                            error_intensity**2, error_area**2]))

    predictor = KalmanPredictor(transition_model)

    updater = KalmanUpdater(measurement_model)

    measure = Mahalanobis(mapping=[0, 1, 2, 3])

    hypothesiser = DistanceHypothesiser(predictor,
                                        updater,
                                        measure,
                                        missed_distance=2)

    data_associator = GNNWith2DAssignment(hypothesiser)

    deleter = UpdateTimeStepsDeleter(dele,
                                     delete_last_pred=True)

    initiator = MultiMeasurementInitiator(GaussianState(np.array([[0], [0], [0], [.5], [0]]),
                                                        np.diag([15**2, 15**2, 15**2, 20**2, 100**2])),
                                          min_points=init,
                                          deleter=UpdateTimeStepsDeleter(3, delete_last_pred=True),
                                          measurement_model=measurement_model,
                                          data_associator=data_associator,
                                          updater=updater)

    tracker = MultiTargetTracker(initiator=initiator,
                                 deleter=deleter,
                                 detector=detector,
                                 data_associator=data_associator,
                                 updater=updater)

    # initialize variables
    tracks = set()

    # go through each frame
    for step, (time, current_tracks) in enumerate(tracker, 1):
        # update track list
        tracks.update(current_tracks)

    # 3) define metrics for optimisation
    track_lengths = np.array([len(track) for track in tracks])
    
    long_tracks = np.array(track_lengths > (0.9*(tMax - tMin))).sum() / 60
    total_tracks = np.array(track_lengths > 0).sum() / 2500
    
    return long_tracks, total_tracks

# %%
# Once the objective function has been defined, a study can be created. The directions argument
# specifies whether the metrics being outputted by the objective function should be maximized or
# minimized. Also, in optimizing the study n_trials must be specified, which is the number of
# trials to be run in the study. Each trial is a full run of the tracking algorithm using
# parameters selected within the ranges given. 
study = optuna.create_study(directions=['maximize', 'minimize'])
study.optimize(objective, n_trials=20)

# %%
# The results of the study can then be visualised though the use of the graphing functions built in
# to Optuna. The Pareto-front plot function can be used when you are optimising over two metrics.
# This plots each trial in the study as a function of the two metrics we were optimising over:
# number of long tracks and total number of tracks. Since we are trying to minimise the total
# number of tracks (y axis) and maximise the number of long tracks (x axis), the trials in the
# bottom right of the graph can be assumed to be the trials with the best parameter set-ups.
plot = optuna.visualization.plot_pareto_front(study, target_names=['Number of Long Tracks (>90% Frames)',
                                                                   'Total Number of Tracks'])
plot.update_layout(showlegend=False)
plot

# %%
# Another plotting function within the Optuna package is the contour plot. This can be plotted for
# any of the optimization parameters (as set by the 'target_name' argument), and shows the
# relationships between pairs of hyperparameters we are optimizing over. We can either plot a grid
# containing all of the pairs of hyperparameters being optimized, or select a subset of
# hyperparameters by including the 'params' argument.
#
# In the contour plot below the interactions between the initiator number, deleter number and
# positional error are plotted with respect to the long tracks optimization metric (number of
# tracks spanning at least 90% of frames). The light areas show the hyperparameter combinations
# that result in a greater number of long tracks, so we can see that higher values for the
# initiator number, deleter number and positional error term are all associated with more long
# tracks.
#
# The second contour plot below shows the same hyperparameters but in this case they are plotted
# with respect to the total tracks optimization metric.
from optuna.visualization import plot_contour

plot_contour(study, target=lambda t: t.values[0], target_name='long tracks', params=['init', 'dele', 'error_pos'])

# %%
plot_contour(study, target=lambda t: t.values[1], target_name='total tracks', params=['init', 'dele', 'error_pos'])

# %%
# We can also plot the hyperparameter importances, as shown below. This gives the importance of
# each of the hyperparameters that we set to vary in our study to the metrics we optimize over. The
# importance is given as a fraction of the change in the metric that the given hyperparameter is
# responsible for (i.e., a hyperparameter importance of 1 indicates that any change to the metric
# is a result of changes to that hyperparameter only).
#
# The hyperparameter importance plot for the long tracks metric indicates that the errors in the
# state vector fields have the biggest impact on determining the number of long tracks produced by
# an algorithm, with the positional error being the most important. The hyperparameter importance
# plot for the total tracks metric shows that the minimum number of frames required for a track to
# initiate has by far the biggest impact on the total number of tracks.
from optuna.visualization import plot_param_importances

plot_param_importances(study, target=lambda t: t.values[0], target_name='long tracks')

# %%
plot_param_importances(study, target=lambda t: t.values[1], target_name='total_tracks')

# %%
# Below is a video showing the tracks using the set-up that was determined to be optimum (the
# furthest right point seen on the Pareto-front plot). The yellow outlines denote the cells included
# in tracks, while the white outlines show the cells that have been segmented but not included in
# tracks.
#
# .. raw:: html
#
#     <video autoplay loop controls>
#       <source src="../_static/cell_demo_2.mp4" type="video/mp4">
#     </video>
#
