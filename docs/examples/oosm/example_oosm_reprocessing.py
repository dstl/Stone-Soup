#!/usr/bin/env python
# coding: utf-8

"""
=====================================================
Handling OOSM by reprocessing the history of a track
=====================================================
"""

# %%
# In previous examples we have presented a number of approaches to deal with out-of-sequence
# measurements (OOSM): storing and re-ordering measurements in a fixed lag buffer, creating
# pseudo-measurements using inverse time dynamics, running a tracker behind real time, and
# re-weighting particle trajectories.
#
# In this example we consider a different approach, where measurements are processed as soon as
# they arrive, and when an out of sequence measurement arrives, the track's history is used to
# insert the measurement at the right point in time. This is achieved by:
#
# 1. finding the most recent state in the track's history *before* the OOSM timestamp
#    :math:`\tau`, and predicting from it to :math:`\tau`;
# 2. updating this prediction with the OOSM;
# 3. reprocessing all the subsequent states, re-predicting and re-updating with their
#    measurements, to bring the track back up to the current time.
#
# This is similar to Algorithm A in [#]_, but rather than delaying all processing with a fixed lag
# buffer of measurements, the latest estimate is always available, and the past states and their
# associated measurements are held by the track. As the reprocessing reuses the original
# measurements, with linear Kalman filters this gives the same result as if the measurements
# had arrived in order.
#
# In Stone Soup, this is implemented with two wrapper components, which can wrap existing
# predictors and updaters:
#
# - :class:`~.OOSPredictorWrapper` predicts from the appropriate state in the track's
#   history, rather than the latest state;
# - :class:`~.OOSUpdaterWrapper` carries out the update, and reprocesses the subsequent states.
#
# As they are wrappers, they can be used with other standard components, such as hypothesisers
# and data associators. In this example we consider a multi-target scenario with clutter, where
# two sensors observe two targets. The scans from the second sensor arrive with a delay, and so
# will be out of sequence with respect to the scans from the first sensor. We compare:
#
# - a tracker using the OOS wrappers, processing scans in the order they arrive;
# - a tracker which ignores any measurements arriving out of sequence;
# - a reference tracker which receives all the scans in the correct order, as if there were no
#   delay.
#
# This example follows this structure:
#
# 1. Create the ground truths, and scans of detections from the two sensors;
# 2. Instantiate the tracking components;
# 3. Run the trackers;
# 4. Visualise the results and compare the tracks using metrics.
#

# %%
# General imports
# ^^^^^^^^^^^^^^^
from datetime import datetime, timedelta

import numpy as np

# Simulation parameters
start_time = datetime.now().replace(microsecond=0)
np.random.seed(1111)  # fix a random seed
num_scans = 60  # number of scans from each sensor
scan_interval = timedelta(seconds=5)  # time between scans of each sensor
sensor2_offset = timedelta(seconds=2.5)  # sensor 2 scans offset from sensor 1 scans
sensor2_latency = timedelta(seconds=12)  # delay in sensor 2 scans arriving
prob_detect = 0.9  # probability of detection
lambdaV = 2  # mean number of clutter detections per scan
v_bounds = np.array([[-300, 300], [-300, 300]])  # x-y bounds of clutter
clutter_spatial_density = lambdaV/np.prod(v_bounds[:, 1] - v_bounds[:, 0])

# %%
# Stone Soup imports
# ^^^^^^^^^^^^^^^^^^
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.state import GaussianState, State, StateVector

# %%
# 1. Create the ground truths, and scans of detections from the two sensors;
# --------------------------------------------------------------------------
# We simulate two targets moving with a nearly constant velocity. As the sensors scan at
# different times, the ground truth is generated every 2.5 seconds, so that there is a truth
# state at each of the scan times.

transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),
                                                          ConstantVelocity(0.05)])

truth_interval = sensor2_offset
timestamps = [start_time + k*truth_interval for k in range(2*num_scans)]

truths = []
for initial_state in ([0, 0.5, -100, 0.3], [0, 0.5, 100, -0.3]):
    truth = GroundTruthPath([GroundTruthState(initial_state, timestamp=start_time)])
    for timestamp in timestamps[1:]:
        truth.append(GroundTruthState(
            transition_model.function(truth[-1], noise=True, time_interval=truth_interval),
            timestamp=timestamp))
    truths.append(truth)

# %%
# Both sensors measure bearing and range, using a :class:`~.CartesianToBearingRange` measurement
# model. The first sensor is located at the origin, and the second sensor is located away from
# the origin, to the south of the targets, so as to observe them from a different angle.

from stonesoup.models.measurement.nonlinear import CartesianToBearingRange

sensor1_mm = CartesianToBearingRange(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.diag([np.radians(1)**2, 5**2]))

sensor2_mm = CartesianToBearingRange(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.diag([np.radians(1)**2, 5**2]),
    translation_offset=StateVector([150, -300]))

# %%
# We now generate the scans of detections from each sensor, including clutter. Each scan is
# recorded with its arrival time at the tracker, as well as the scan time. The scans from the
# first sensor arrive immediately, whereas scans from the second sensor arrive with a delay of
# 12 seconds. As the sensors scan every 5 seconds, each scan from the second sensor arrives after
# two later scans from the first sensor, and so will be out of sequence.

from stonesoup.types.detection import TrueDetection, Clutter

scans = []  # list of (arrival time, scan time, detections)
for k, timestamp in enumerate(timestamps):
    if k % 2 == 0:  # sensor 1 scan
        measurement_model = sensor1_mm
        arrival_time = timestamp
    else:  # sensor 2 scan
        measurement_model = sensor2_mm
        arrival_time = timestamp + sensor2_latency

    detections = set()
    for truth in truths:
        if np.random.rand() <= prob_detect:
            measurement = measurement_model.function(truth[k], noise=True)
            detections.add(TrueDetection(state_vector=measurement,
                                         groundtruth_path=truth,
                                         timestamp=timestamp,
                                         measurement_model=measurement_model))

    # Generate clutter uniformly in the x-y bounds
    for _ in range(np.random.poisson(lambdaV)):
        x = np.random.uniform(*v_bounds[0, :])
        y = np.random.uniform(*v_bounds[1, :])
        clutter_state = State(StateVector([x, 0, y, 0]))
        detections.add(Clutter(measurement_model.function(clutter_state, noise=False),
                               timestamp=timestamp,
                               measurement_model=measurement_model))

    scans.append((arrival_time, timestamp, detections))

# Order the scans by the time they arrive at the tracker
arrival_ordered_scans = sorted(scans, key=lambda scan: scan[0])

# %%
# 2. Instantiate the tracking components;
# ---------------------------------------
# We use an :class:`~.ExtendedKalmanPredictor` and :class:`~.ExtendedKalmanUpdater`. As each
# detection carries its own measurement model, we don't need to provide one to the updater.

from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater

predictor = ExtendedKalmanPredictor(transition_model)
updater = ExtendedKalmanUpdater(measurement_model=None)

# %%
# We now wrap these in the OOS components. The :class:`~.OOSUpdaterWrapper` also requires a
# predictor, which is used to re-predict the subsequent states when reprocessing. Both wrappers
# have further options, which aren't used in this example:
#
# - :attr:`~.OOSPredictorWrapper.backward_predictor` can be used to predict backwards, in case a
#   measurement arrives from before the start of the track;
# - :attr:`~.OOSPredictorWrapper.smoother` can be used to smooth the track's history, so that the
#   prediction to the OOSM time also uses information from later measurements, which can help
#   with data association. The update itself is still carried out using the unsmoothed state;
# - :attr:`~.OOSUpdaterWrapper.hypothesiser` can be used to regenerate hypotheses for the
#   subsequent measurements when reprocessing, e.g. to recalculate association probabilities for
#   probabilistic data association;
# - :attr:`~.OOSUpdaterWrapper.min_history` can be used to limit how far back the history is
#   kept, allowing old states to be released from memory. By default, all history is kept.

from stonesoup.predictor.oos import OOSPredictorWrapper
from stonesoup.updater.oos import OOSUpdaterWrapper

oos_predictor = OOSPredictorWrapper(predictor)
oos_updater = OOSUpdaterWrapper(updater, predictor=oos_predictor)

# %%
# For data association, we use :class:`~.GNNWith2DAssignment` with a
# :class:`~.DistanceHypothesiser`. One set of these is created with the OOS components, and
# another with the standard components, for the other trackers.

from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis

oos_data_associator = GNNWith2DAssignment(DistanceHypothesiser(
    oos_predictor, oos_updater, measure=Mahalanobis(), missed_distance=3))

data_associator = GNNWith2DAssignment(DistanceHypothesiser(
    predictor, updater, measure=Mahalanobis(), missed_distance=3))

# %%
# For simplicity, we start the tracks from known priors, near the targets' initial positions.

priors = [
    GaussianState([0, 0.5, -100, 0.3], np.diag([25, 1, 25, 1]), timestamp=start_time),
    GaussianState([0, 0.5, 100, -0.3], np.diag([25, 1, 25, 1]), timestamp=start_time),
]

# %%
# 3. Run the trackers;
# --------------------
# We define a simple function to run a tracker over a sequence of scans. For each scan, the
# tracks are associated with the detections, and tracks are updated with any associated
# detection, or otherwise the prediction is added to the track.
#
# Note that for the OOS components, missed detections are also passed to the
# :class:`~.OOSUpdaterWrapper`. This is because the prediction may be for a time in the past, in
# which case the updater will insert the prediction into the track's history, and reprocess the
# subsequent states to return an estimate at the latest time.

from stonesoup.types.track import Track


def run_tracker(scans, data_associator, updater):
    tracks = {Track([prior]) for prior in priors}
    for _, scan_time, detections in scans:
        associations = data_associator.associate(tracks, detections, scan_time)
        for track, hypothesis in associations.items():
            if hypothesis or isinstance(updater, OOSUpdaterWrapper):
                track.append(updater.update(hypothesis))
            else:
                track.append(hypothesis.prediction)
    return tracks


# %%
# Firstly, we run the tracker with the OOS components, processing the scans in the order they
# arrive. When a scan from the second sensor arrives, the tracks have already been updated with
# the later scans from the first sensor. The :class:`~.OOSPredictorWrapper` predicts from the
# state before the scan time, and the :class:`~.OOSUpdaterWrapper` updates and then reprocesses
# the later updates, returning a new estimate at the latest time.

oos_tracks = run_tracker(arrival_ordered_scans, oos_data_associator, oos_updater)

# %%
# Each track contains the estimates as they were at the time they were produced, so the states
# in the track before an OOSM's time don't include the information from the OOSM. The full
# reprocessed history can be retrieved from the latest state using
# :func:`~.get_past_states`, which follows the chain of states back from the latest estimate.

from stonesoup.predictor.oos import get_past_states

oos_tracks = {Track(list(get_past_states(track.state))[::-1]) for track in oos_tracks}

# %%
# Secondly, we run the tracker with the standard components, processing the scans in the order
# they arrive, but ignoring any scans that arrive out of sequence. As all scans from the second
# sensor arrive out of sequence, this tracker effectively only uses the first sensor.

in_sequence_scans = []
latest_scan_time = start_time
for scan in arrival_ordered_scans:
    if scan[1] >= latest_scan_time:
        in_sequence_scans.append(scan)
        latest_scan_time = scan[1]

ignore_oosm_tracks = run_tracker(in_sequence_scans, data_associator, updater)

# %%
# Finally, as a reference, we run the tracker with the standard components, processing all the
# scans in the order of their scan time, as if there were no delay.

reference_tracks = run_tracker(scans, data_associator, updater)

# %%
# As the OOS components reprocess the history of the track using the same measurements, the OOS
# tracks will be the same as the reference tracks, provided the same data association decisions
# are made. However, these decisions can differ: when a scan from the first sensor arrives, it is
# associated before the earlier, delayed, scan from the second sensor has arrived, and these
# associations are kept when reprocessing (unless a :attr:`~.OOSUpdaterWrapper.hypothesiser` is
# provided). We can check how many of the states in the OOS tracks match the reference tracks.


def matches(state, track):
    return any(state.timestamp == reference_state.timestamp
               and np.allclose(state.state_vector, reference_state.state_vector)
               and np.allclose(state.covar, reference_state.covar)
               for reference_state in track)


def sort_by_initial_y(tracks):
    return sorted(tracks, key=lambda track: track[0].state_vector[2])


for oos_track, reference_track in zip(
        sort_by_initial_y(oos_tracks), sort_by_initial_y(reference_tracks)):
    num_matches = sum(matches(state, reference_track) for state in oos_track)
    print(f"{num_matches} of {len(oos_track)} OOS track states match the reference track")

# %%
# 4. Visualise the results and compare the tracks using metrics.
# --------------------------------------------------------------
# We can now plot the tracks, along with the ground truths and detections from both sensors.

from stonesoup.plotter import AnimatedPlotterly

plotter = AnimatedPlotterly(timesteps=timestamps)
plotter.plot_ground_truths(truths, [0, 2])
plotter.plot_measurements([scan[2] for scan in scans if scan[2]], [0, 2])
plotter.plot_tracks(oos_tracks, [0, 2], uncertainty=True, label='OOS Tracks',
                    line=dict(color='orange'))
plotter.plot_tracks(ignore_oosm_tracks, [0, 2], label='Ignore OOSM Tracks',
                    line=dict(color='red'))
plotter.plot_tracks(reference_tracks, [0, 2], label='Reference Tracks',
                    line=dict(color='green', dash='dot'))
plotter.fig

# %%
# Evaluate the track accuracy
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# To compare the trackers, we use the OSPA metric between each set of tracks and the ground
# truths. As the tracker ignoring OOSM has no estimates at the second sensor's scan times, the
# metric is calculated at the first sensor's scan times only, using the latest estimate of each
# track at each of these times.

from stonesoup.dataassociator.tracktotrack import TrackToTruth
from stonesoup.metricgenerator.manager import MultiManager
from stonesoup.metricgenerator.ospametric import OSPAMetric

sensor1_timestamps = timestamps[::2]


def at_sensor1_timestamps(paths):
    return {type(path)([path[timestamp] for timestamp in sensor1_timestamps])
            for path in paths}


generators = [
    OSPAMetric(c=40, p=1, generator_name=f'OSPA {name}',
               tracks_key=name, truths_key='truths')
    for name in ('OOS', 'Ignore OOSM', 'Reference')]
metric_manager = MultiManager(generators, TrackToTruth(association_threshold=30))
metric_manager.add_data({'truths': at_sensor1_timestamps(truths),
                         'OOS': at_sensor1_timestamps(oos_tracks),
                         'Ignore OOSM': at_sensor1_timestamps(ignore_oosm_tracks),
                         'Reference': at_sensor1_timestamps(reference_tracks)})
metrics = metric_manager.generate_metrics()

from stonesoup.plotter import MetricPlotter

graph = MetricPlotter()
graph.plot_metrics(metrics, generator_names=[generator.generator_name for generator in generators],
                   color=['orange', 'red', 'green'])
graph.axes[0].set(ylabel='OSPA metrics', title='OSPA distances over time')
graph.fig

# %%
# As the OOS tracks match the reference tracks, the OSPA distance for the OOS tracks is hidden
# under that of the reference tracks. The tracker ignoring the OOSM is generally less accurate,
# as it doesn't use the information from the second sensor.

# %%
# Conclusion
# ----------
# In this example we have shown how to use the :class:`~.OOSPredictorWrapper` and
# :class:`~.OOSUpdaterWrapper` components to handle out of sequence measurements, by
# reprocessing the history of each track. These components wrap the standard predictors and
# updaters, and so can be used alongside other Stone Soup components, such as hypothesisers and
# data associators.
#
# In this scenario, the tracks produced match those of a tracker which received the measurements
# in the correct order, while still producing estimates as soon as each measurement arrives.
# Ignoring the out of sequence measurements instead discards the information from the second
# sensor, resulting in less accurate tracks.

# %%
# References
# ----------
# .. [#] Y. Bar-Shalom, M. Mallick, H. Chen, R. Washburn, 2002,
#        One-step solution for the general out-of-sequence measurement
#        problem in tracking, Proceedings of the 2002 IEEE Aerospace
#        Conference.
