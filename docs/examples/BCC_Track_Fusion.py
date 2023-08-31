"""
Track Fusion Example
===========================
This example demonstrates how to use :class:`.TrackFusedTracker`. Two radar
sensors with identical specification track two crossing targets. The individual
tracks from these sensors is used as input for the `TrackFusedTracker`. Lastly a
detection fused tracker is created and simulated for comparison.
"""


# %%
# Build the Scenario
# ------------------
# Waypoints are used to define the targets’ flight path. These waypoints are
# interpolated to give a GroundTruthPath with states at every second. Two static
# :class:`~.RadarElevationBearingRange` sensors are created and located ~100m
# away from the targets’ starting location. The targets fly towards the sensors
# during the simulation.


# %%
# First some general setup

# Some general imports
import datetime
from typing import Collection
import numpy as np

# Standard Stone Soup imports
from stonesoup.dataassociator.neighbour import GlobalNearestNeighbour
from stonesoup.dataassociator.tracktotrack import OneToOneTrackAssociator
from stonesoup.deleter.time import UpdateTimeDeleter
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.initiator.simple import SimpleMeasurementInitiator
from stonesoup.measures import Euclidean
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.plotter import Plotterly
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.sensor.radar.radar import RadarElevationBearingRange
from stonesoup.tracker.simple import MultiTargetTracker
from stonesoup.types.groundtruth import GroundTruthPath
from stonesoup.types.state import GaussianState, State, StateVector
from stonesoup.updater.kalman import ExtendedKalmanUpdater

# Imports required for track fusion
from stonesoup.feeder.multi import SyncMultipleTrackFeedersToOneFeeder
from stonesoup.feeder.track import SyncMultiTrackFeeder
from stonesoup.feeder.track_continuity_buffer import TrackerWithContinuityBuffer, \
    BetaTrackContinuityBuffer
from stonesoup.mixturereducer.gaussianmixture import BasicConvexCombination
from stonesoup.non_state_measures import MeanMeasure, RecentStateSequenceMeasure
from stonesoup.tracker.track_fusion import TrackFusedTracker

# Set up some initial parameters
np.random.seed(2023)
start_time = datetime.datetime(2023, 1, 1, 0, 0, 0)

# Mapping indices
_X = 0
_Y = 2
_Z = 4


# %%
# Create the Target Trajectories
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In this section the movement of the target aircraft is created. This specific geometry was
# chosen to give an easy challenge to the track association algorithms.

# Create waypoints of both targets
target_1_waypoints = [
    State([[10], [0], [100], [-1], [1], [0]], start_time + datetime.timedelta(seconds=0)),
    State([[10], [1], [90], [-1], [1], [0]], start_time + datetime.timedelta(seconds=10)),
    State([[50], [0], [50], [-1], [1], [0]], start_time + datetime.timedelta(seconds=50)),
    State([[50], [0], [40], [-1], [1], [0]], start_time + datetime.timedelta(seconds=60))
]
target_2_waypoints = [
    State([[50], [0], [100], [-1], [2], [0]], start_time + datetime.timedelta(seconds=0)),
    State([[50], [-1], [90], [-1], [2], [0]], start_time + datetime.timedelta(seconds=10)),
    State([[10], [0], [50], [-1], [2], [0]], start_time + datetime.timedelta(seconds=50)),
    State([[10], [0], [40], [-1], [2], [0]], start_time + datetime.timedelta(seconds=60))
]


# %%
# In between waypoints is interpolated to make a consistent sequence of states (locations). The
# `interpolate_states` function performs a linear interpolation between states to create new
# intermediate states.
def interpolate_states(existing_states: Collection[State], interpolate_time: datetime.datetime):
    float_times = [state.timestamp.timestamp() for state in existing_states]
    output = np.zeros(len(existing_states[0].state_vector))
    for i in range(len(output)):
        a_states = [np.double(state.state_vector[i]) for state in existing_states]
        output[i] = np.interp(interpolate_time.timestamp(), float_times, a_states)

    return State(StateVector(output), timestamp=interpolate_time)


all_times = [start_time + datetime.timedelta(seconds=x) for x in range(60)]
target_1_states = []
target_2_states = []
for t in all_times:
    target_1_states.append(interpolate_states(target_1_waypoints, t))
    target_2_states.append(interpolate_states(target_2_waypoints, t))

target_1_truth = GroundTruthPath(target_1_states)
target_2_truth = GroundTruthPath(target_2_states)


# %%
# Create Sensors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Two static :class:`~.RadarElevationBearingRange` sensors are created. Both
# have the same accuracy, they have elevation accuracy of ~14 degrees, a
# bearing angle accuracy of ~25 degrees and a range accuracy* of ~30m. The
# sensors are located 20m apart.
#
# The accuracy values is the standard deviation of the measures
sensor_1_position = np.array([[20], [10], [0]])
sensor_2_position = np.array([[40], [10], [0]])

sensor1 = RadarElevationBearingRange(
    position=sensor_1_position,
    ndim_state=6,
    position_mapping=(0, 2, 4),
    noise_covar=np.diag([np.radians(0.05), np.radians(0.2), 0.1]),
)
sensor2 = RadarElevationBearingRange(
    position=sensor_2_position,
    ndim_state=6,
    position_mapping=(0, 2, 4),
    noise_covar=np.diag([np.radians(0.05), np.radians(0.2), 0.1]),
)


# %%
# Plot the Scenario
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

plotterXZ = Plotterly(xaxis=dict(title=dict(text="<i>x</i> (East)")),
                      yaxis=dict(title=dict(text="<i>z</i> (Altitude)")))
plotterXZ.plot_ground_truths({target_1_truth, target_2_truth}, [_X, _Z],
                             truths_label="Target Flight Path")
plotterXZ.fig

# %%
# There isn’t much to see in this graph as both targets are at the same altitude for the whole
# simulation.


plotterXY = Plotterly(xaxis=dict(title=dict(text="<i>x</i> (East)")),
                      yaxis=dict(title=dict(text="<i>y</i> (North)"), scaleanchor="x",
                                 scaleratio=1))
plotterXY.plot_ground_truths({target_1_truth, target_2_truth}, [_X, _Y],
                             truths_label="Target Flight Path")
plotterXY.plot_sensors([sensor1, sensor2], sensor_label="Sensor Location")
plotterXY.fig


# %%
# Generate Detections
# -------------------
# We’re going to simulate detecting both targets from both sensors.

s1_detection_inputs = []
s2_detection_inputs = []
for t, state1, state2 in zip(all_times, target_1_truth, target_2_truth):
    states = state1, state2
    s1_detection_inputs.append((t, sensor1.measure(set(states), noise=True)))
    s2_detection_inputs.append((t, sensor2.measure(set(states), noise=True)))

s1_measurements = [detection
                   for _, set_of_detections in s1_detection_inputs
                   for detection in set_of_detections]

s2_measurements = [detection
                   for _, set_of_detections in s2_detection_inputs
                   for detection in set_of_detections]


# %%
# Detections in XY
plotterXY.plot_measurements(s1_measurements, [_X, _Y], measurements_label="Sensor One",
                            marker=dict(color='#1F77B4'))
plotterXY.plot_measurements(s2_measurements, [_X, _Y], measurements_label="Sensor Two",
                            marker=dict(color='#FF7F0E'))
plotterXY.fig


# %%
# Detections in XZ
plotterXZ.plot_measurements(s1_measurements, [_X, _Z], measurements_label="Sensor One",
                            marker=dict(color='#1F77B4'))
plotterXZ.plot_measurements(s2_measurements, [_X, _Z], measurements_label="Sensor Two",
                            marker=dict(color='#FF7F0E'))
plotterXZ.fig


# %%
# Tracking
# -------------------------------------------------
# In this section four trackers will be created: radar tracker, infrared tracker, track fusion
# tracker and detection fusion tracker. All the trackers will be based on a standard tracker. The
# trackers will be ran and will process the sensor data. The output of the trackers will be
# explored in subsequent sections


# %%
# Create Standard Tracker
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The standard tracker uses an Extended Kalman Filter (EKF). Global Nearest Neighbour (GNN) is
# used to associate detections to tracks. Tracks are initiated from any detections that aren’t
# associated to a track. Tracks will be deleted after 10 seconds without a detection being
# associated to them. The `create_tracker_kwargs` function generates the key word arguments for
# each tracker
def create_tracker_kwargs(transition_model, detector=None, measurement_model=None):
    initial_state = GaussianState(state_vector=[0] * 6, covar=np.diag([1000] * 6))
    initiator = SimpleMeasurementInitiator(initial_state, measurement_model=measurement_model,
                                           skip_non_reversible=True)
    deleter = UpdateTimeDeleter(time_since_update=datetime.timedelta(seconds=10))
    predictor = ExtendedKalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=measurement_model)
    data_associator = GlobalNearestNeighbour(DistanceHypothesiser(
        predictor, updater, Euclidean(), missed_distance=10))

    return {"initiator": initiator,
            "deleter": deleter,
            "detector": detector,
            "data_associator": data_associator,
            "updater": updater}


# %%
# Create Sensor One Tracker
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Sensor One Tracker (S1 Tracker) uses the standard tracker inputs with a three
# dimensional constant velocity
# model.
transition_noise = 0.02
transition_model_xyz = CombinedLinearGaussianTransitionModel((
    ConstantVelocity(transition_noise),
    ConstantVelocity(transition_noise),
    ConstantVelocity(transition_noise)))

s1_tracker = MultiTargetTracker(**create_tracker_kwargs(
    transition_model=transition_model_xyz,
    detector=s1_detection_inputs)
)


# %%
# Create Sensor Two Tracker
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Sensor Two Tracker (S2 Tracekr) is identical to the Sensor One Tracker. It
# uses the standard tracker inputs with a three dimensional constant velocity
# model.
s2_tracker = MultiTargetTracker(**create_tracker_kwargs(
    transition_model=transition_model_xyz,
    detector=s2_detection_inputs)
)

# %%
# Create Track Fusion Tracker
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The output of the S1 Tracker and S2 tracker are fed into a :class:`~.TrackFusedTracker`.
# A track association algorithm will associate closely spaced tracks together.
# After being associated together the tracks are combined using
# :class:`~.BasicConvexCombination`. The tracker will output
# (:class:`~.datetime`, Set[:class:`.Track`]) like a normal tracker.

# %%
# A :class:`~.SyncMultiTrackFeeder` is used to send the output of tracker S1 and
# tracker S2 to multiple locations. This allows the track output to be recorded
# normally and the track fused tracker to receive the output
s1_track_multi_feeder = SyncMultiTrackFeeder(s1_tracker)
s2_track_multi_feeder = SyncMultiTrackFeeder(s2_tracker)

# %%
# A :class:`.SyncMultipleTrackFeedersToOneFeeder` is used to feed the output of
# the S1 and S2 trackers into the track fused tracker.
multi_track_feeder = SyncMultipleTrackFeedersToOneFeeder(readers=[
    s1_track_multi_feeder.create_track_feeder(),
    s2_track_multi_feeder.create_track_feeder()]
)

# %%
# An :class:`.OneToOneTrackAssociator` is used to associate tracks together.
# The associator will look for pairs of tracks with the minimum average
# Euclidean distance between them over the last 5 time steps. These tracks will
# be combined together using :class:`.BasicConvexCombination`.
track_associator = OneToOneTrackAssociator(
        measure=MeanMeasure(
            RecentStateSequenceMeasure(n_states_to_compare=5,
                                       measure=Euclidean(mapping=[0, 2, 4]))))

combined_tracker = TrackFusedTracker(
    multiple_track_feeder=multi_track_feeder,
    state_combiner=BasicConvexCombination(),
    track_associator=track_associator,
)

# %%
# The unfiltered output of a :class:`.TrackFusedTracker` isn’t very useful and
# won’t normally be used. However in this example is used to demonstrate its
# impracticality in the :ref:`auto_examples/bcc_track_fusion:raw track fused tracking`
# section. A :class:`.TrackerWithContinuityBuffer` is used with a
# :class:`.BetaTrackContinuityBuffer` to filter the output to provide a more
# valuable output.
combined_tracks_feeder = SyncMultiTrackFeeder(combined_tracker)

raw_combined_tracks_feeder = combined_tracks_feeder.create_track_feeder()

processed_combined_tracks_feeder = TrackerWithContinuityBuffer(
    combined_tracks_feeder.create_track_feeder(),
    BetaTrackContinuityBuffer()
)


# %%
# Create Detection Fusion Tracker
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The Detection Fusion Tracker uses an identical tracker to the other trackers. The only difference
# is that the detection fusion tracker receives both detections.


# Combine all measurement input
det_fuse_inputs = sorted([*s1_detection_inputs, *s2_detection_inputs])

# Create the tracker
det_fuse_tracker = MultiTargetTracker(**create_tracker_kwargs(
    transition_model=transition_model_xyz,
    detector=det_fuse_inputs)
)


# %%
# Run Trackers
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The S1, S2 and Combined Tracker are all run synchronously.
def run_trackers_concurrently(*track_feeders):
    track_feeders_outputs = [set() for _ in track_feeders]

    for time_tracks in zip(*track_feeders):
        times = set()
        for tracker_output, (time, tracks) in zip(track_feeders_outputs, time_tracks):
            times.add(time)
            tracker_output |= tracks
        if len(times) > 1:
            raise Exception("Time from trackers does not match")

    return track_feeders_outputs


all_s1_tracks, all_s2_tracks, all_raw_combo_tracks, all_processed_combo_tracks = \
    run_trackers_concurrently(
        s1_track_multi_feeder.create_track_feeder(),
        s2_track_multi_feeder.create_track_feeder(),
        raw_combined_tracks_feeder,
        processed_combined_tracks_feeder
    )


# %%
# Run the detection fusion tracker. The detection fusion tracker can't be run
# with the other trackers in the `run_trackers_concurrently` function. This is
# because the detection fusion tracker takes double the number of processing
# steps as it has to process sensor one and sensor two detections separately.
all_tracks_fuse = set()
for time, tracks in det_fuse_tracker:
    all_tracks_fuse |= tracks


# %%
# Tracking Output
# -----------------

# %%
# Quick function to show only certain parts of the graph
def hide_plot_traces(fig, items_to_hide: set):
    for fig_data in fig.data:
        if fig_data.legendgroup in items_to_hide:
            fig_data.visible = "legendonly"
        else:
            fig_data.visible = None


# The XY plot will shown later in the example
plotterXY.plot_tracks(all_s1_tracks, [_X, _Y], track_label="Sensor One Tracker")
plotterXY.plot_tracks(all_s2_tracks, [_X, _Y], track_label="Sensor Two Tracker")
plotterXY.plot_tracks(all_raw_combo_tracks, [_X, _Y], track_label="Raw Track Fused Tracker")
plotterXY.plot_tracks(all_processed_combo_tracks, [_X, _Y],
                      track_label="Processed Track Fused Tracker")

# Plot Tracks in XZ
plotterXZ.plot_tracks(all_s1_tracks, [_X, _Z], track_label="Sensor One Tracker")
plotterXZ.plot_tracks(all_s2_tracks, [_X, _Z], track_label="Sensor Two Tracker")
plotterXZ.plot_tracks(all_raw_combo_tracks, [_X, _Z], track_label="Raw Track Fused Tracker")
plotterXZ.plot_tracks(all_processed_combo_tracks, [_X, _Z],
                      track_label="Processed Track Fused Tracker")
plotterXZ.fig
# %%
# Due to the close spaced targets in altitude, there isn't much to see in the XZ plot
# Plot Tracks in XY


# %%
# Standard Tracking
# ^^^^^^^^^^^^^^^^^

# %%
# **Sensor 1**
hide_plot_traces(plotterXY.fig, {"Sensor Two<br>(Detections)", "Sensor Two Tracker",
                                 "Raw Track Fused Tracker", "Processed Track Fused Tracker",
                                 "Detection Fusion Tracker"})
plotterXY.fig

# %%
# **Sensor 2**
hide_plot_traces(plotterXY.fig, {"Sensor One<br>(Detections)", "Sensor One Tracker",
                                 "Raw Track Fused Tracker", "Processed Track Fused Tracker",
                                 "Detection Fusion Tracker"})
plotterXY.fig


# %%
# Raw Track Fused Tracking
# ^^^^^^^^^^^^^^^^^^^^^^^^
# :class:`~.TrackFusedTracker` produces a new track object on each timestep. This is due to that
# each track association and combination is independent from the previous. The result is that each
# state is a new track object
hide_plot_traces(plotterXY.fig, {"Sensor One<br>(Detections)", "Sensor One Tracker",
                                 "Sensor Two<br>(Detections)", "Sensor Two Tracker",
                                 "Processed Track Fused Tracker", "Detection Fusion Tracker"})
plotterXY.fig


# %%
# Track Fused Tracking
# ^^^^^^^^^^^^^^^^^^^^
# For subsequent time-steps the :class:`~.BetaTrackContinuityBuffer` joins the
# tracks that have the same :attr:`.Track.id`. When :class:`~.TrackFusedTracker`
# combines two tracks, the output track id is created from the contributing
# tracks using :func:`~.get_fused_id` (which is reproducible). Other
# :class:`.TrackContinuityBuffer` classes could be used.

hide_plot_traces(plotterXY.fig, {"Sensor One<br>(Detections)", "Sensor Two<br>(Detections)",
                                 "Raw Track Fused Tracker", "Detection Fusion Tracker",
                                 "Sensor Location"})
plotterXY.fig

# %%
# The processed track fused track should be in between the two individual sensor tracks

# %%
# Detection Fusion Tracking
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Plot Sensor One & Two Tracks in XY plane
plotterXY.plot_tracks(all_tracks_fuse, [_X, _Y], track_label="Detection Fusion Tracker")
hide_plot_traces(plotterXY.fig, {"Sensor One Tracker", "Sensor Two Tracker",
                                 "Raw Track Fused Tracker", "Processed Track Fused Tracker"})
plotterXY.fig

# %%
# Reproducibility
# ^^^^^^^^^^^^^^^
# Usually the individual sensors, track fused and detection fused trackers
# perform well. However they can be confused during the target crossing
# resulting in poor tracking (e.g. track breakages).  The standard tracker
# isn’t optimised at all and optimising the trackers could improve the tracking
# performance and reliability.
