#!/usr/bin/env python

"""
=================
CLEAR MOT example
=================
"""

# %%
# Introduction
# ------------
# This example demonstrates the CLEAR MOT metrics available in Stone Soup and how they
# are used with the :class:`~.MultiManager` to assess tracking performance. The
# CLEAR MOT metrics require a specific association scheme between the truths and tracks
# by matching a single truth to a track based on the proximity and previous assignment.
#
# To generate CLEAR MOT metrics, we need:
#  - An instance of  :class:`~.ClearMotAssociator` - this is used to associate the truths and
#    tracks, so that, a single truth is associated with a single track by a pre-specified distance
#    threshold.
#  - An instance of :class:`~.ClearMotMetrics` - these are used to compute the MOTA and MOTP
#    metrics based on the associations between both truths and tracks.
#  - The :class:`~.MultiManager` metric manager - this is used to hold the metric generator(s)
#    as well as all the ground truth and track sets we want to generate our
#    metrics from. We will generate our metrics using the :meth:`generate_metrics` method
#    of the :class:`~.MultiManager` class.

# %%
# Generate ground truths and tracks
# ---------------------------------
# We start by simulating 2 targets moving in different directions across the 2D Cartesian plane.
# They start at (0, 0) and (0, 20) and cross roughly half-way through their transit. This
# section is solely for demonstration purposes, feel free to replace with your own tracking logic
# to create the sets for truth and tracks. Both sets are used to generate the metrics in the next
# section.

# %%
# Generate ground truth tracks
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Two targets moving in different directions across the 2D Cartesian plane.

from datetime import datetime, timedelta

import numpy as np

start_time = datetime.now().replace(microsecond=0)

from ordered_set import OrderedSet

from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

np.random.seed(1991)

truths = OrderedSet()

transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                          ConstantVelocity(0.005)])

timesteps = [start_time]
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=timesteps[0])])
for k in range(1, 21):
    timesteps.append(start_time+timedelta(seconds=k))
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=timesteps[k]))
truths.add(truth)

truth = GroundTruthPath([GroundTruthState([0, 1, 20, -1], timestamp=timesteps[0])])
for k in range(1, 21):
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=timesteps[k]))
_ = truths.add(truth)

# %%
# Create an interactive plot instance and add the truth to it.

from stonesoup.plotter import AnimatedPlotterly

plotter = AnimatedPlotterly(timesteps, tail_length=0.3)
plotter.plot_ground_truths(truths, [0, 2])
plotter.fig

# %%
# Generate detections with clutter
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The detections (and clutter) from the truth tracks are later used as input for the
# multiple target tracking.

from scipy.stats import uniform

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.detection import Clutter, TrueDetection

measurement_model = LinearGaussian(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.array([[0.75, 0],
                          [0, 0.75]])
    )
all_measurements = []

for k in range(20):
    measurement_set = set()

    for truth in truths:
        # Generate actual detection from the state with a 10% chance that no detection is received.
        if np.random.rand() <= 0.9:
            measurement = measurement_model.function(truth[k], noise=True)
            measurement_set.add(TrueDetection(state_vector=measurement,
                                              groundtruth_path=truth,
                                              timestamp=truth[k].timestamp,
                                              measurement_model=measurement_model))

        # Generate clutter at this time-step
        truth_x = truth[k].state_vector[0]
        truth_y = truth[k].state_vector[2]
        for _ in range(np.random.randint(10)):
            x = uniform.rvs(truth_x - 10, 20)
            y = uniform.rvs(truth_y - 10, 20)
            measurement_set.add(Clutter(np.array([[x], [y]]), timestamp=truth[k].timestamp,
                                        measurement_model=measurement_model))
    all_measurements.append(measurement_set)

# %%
# Run multiple target tracking
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Create the Kalman predictor and updater
from stonesoup.predictor.kalman import KalmanPredictor

predictor = KalmanPredictor(transition_model)

from stonesoup.updater.kalman import KalmanUpdater

updater = KalmanUpdater(measurement_model)

# %%
# We will quantify predicted-measurement to measurement distance
# using the Mahalanobis distance.

from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis

hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=3)


from stonesoup.dataassociator.neighbour import GlobalNearestNeighbour

data_associator = GlobalNearestNeighbour(hypothesiser)

# %%
# We create 2 priors reflecting the targets' initial states.
from stonesoup.types.state import GaussianState

prior1 = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)
prior2 = GaussianState([[0], [1], [20], [-1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)

# %%
# Loop through the predict, hypothesise, associate and update steps.
from stonesoup.types.track import Track

tracks = {Track([prior1]), Track([prior2])}

for n, measurements in enumerate(all_measurements):
    # Calculate all hypothesis pairs and associate the elements in the best subset to the tracks.
    hypotheses = data_associator.associate(tracks,
                                           measurements,
                                           start_time + timedelta(seconds=n))
    for track in tracks:
        hypothesis = hypotheses[track]
        if hypothesis.measurement:
            post = updater.update(hypothesis)
            track.append(post)
        else:  # When data associator says no detections are good enough, we'll keep the prediction
            track.append(hypothesis.prediction)

# %%
# Add tracks to the interactive plot

plotter.plot_tracks(tracks, [0, 2], uncertainty=False)
plotter.fig 

# %%
# Compute CLEAR MOT metrics
# -------------------------
# Having both the `truths` and  `tracks` sets, we now can compute the metrics.

# %%
# Create metric generator and metric manager
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from stonesoup.dataassociator.clearmot import ClearMotAssociator
from stonesoup.measures import Euclidean
from stonesoup.metricgenerator.clearmotmetrics import ClearMotMetrics
from stonesoup.metricgenerator.manager import MultiManager

clear_mot_metrics = ClearMotMetrics(generator_name='CLEARMOT_gen',
                                    tracks_key='tracks', truths_key='truths',
                                    distance_measure=Euclidean((0, 2)))

association_distance = 3.0  # meters
clear_mot_associator = ClearMotAssociator(measure=Euclidean((0, 2)),
                                          association_threshold=association_distance)
metric_manager = MultiManager([clear_mot_metrics], associator=clear_mot_associator)

# %%
# Add tracks data to metric manager

metric_manager.add_data({'truths': truths,
                         'tracks': tracks}, overwrite=False)

# %%
# Compute metrics
# ^^^^^^^^^^^^^^^
# We are now ready to generate the metrics from our MultiManager.

metrics = metric_manager.generate_metrics()

print("MOTP:", "{:.2f}m".format(metrics["CLEARMOT_gen"]["MOTP"].value))
print("MOTA:", "{:.2f}".format(metrics["CLEARMOT_gen"]["MOTA"].value))

# %%
# Discussion
# ^^^^^^^^^^
# First, we plot both tracks and truths

plotter.fig

# %%
# When associated, the average distance between tracks and truths is around 1m, which is reflected
# by the MOTP metric.
#
# The MOTA score is around 0.57, which means that more than a half of the truth samples are
# successfully tracked (i.e. below the association distance of 3 metres).
# By observing the proximity of truths and tracks, we can see that the green (south-east heading)
# is followed by the violet track. At least half (i.e. 50%) of the total number of truth samples
# is successfully tracked.
#
# While observing the red (north-east heading) truth, we see that the orange track deviates from
# it at around the first third of the complete observation period. I.e. a third of the
# red track is successfully tracked, which adds a positive amount to the percentage of tracked
# truths.
#
# In total, both tracks cover 57% of the truth samples. Unmatched track samples are regarded
# as False Positives.
