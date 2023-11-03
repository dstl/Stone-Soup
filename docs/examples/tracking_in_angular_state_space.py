"""
Tracking in Angular State Space Example
========================================

Todo - Add Description
"""

# %%
# Build the scenario
# -------------------------------------------------

# %%
# First all the modules needed for this script are imported. Some initial parameters are
# created. The numpy random seed is set*, this is to ensure consistent output from the script.

import datetime
from collections import defaultdict
from typing import Sequence, Union, Set

import numpy as np

from stonesoup.dataassociator.neighbour import GlobalNearestNeighbour
from stonesoup.deleter.time import UpdateTimeDeleter
from stonesoup.feeder.modify import AngleTrackingDetectionFeeder, \
    StaticRotationalFrameAngleDetectionFeeder
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.initiator.simple import SimpleMeasurementInitiator
from stonesoup.measures import Mahalanobis
from stonesoup.models.transition.linear import ConstantVelocity, \
    CombinedLinearGaussianTransitionModel
from stonesoup.platform.base import MovingPlatform
from stonesoup.plotter import Plotterly as Plotter
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.sensor.passive import PassiveElevationBearing
from stonesoup.tracker.angle import AngleMultipleTargetTracker
from stonesoup.tracker.simple import MultiTargetTracker
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.state import GaussianState
from stonesoup.types.state import State
from stonesoup.types.track import Track
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.types.detection import Detection, TrueDetection
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearing, \
    CartesianToElevationBearingRange


_X = 0
_Y = 2
_Z = 4
start_time = datetime.datetime(2023, 1, 1, 0, 0, 0)
np.random.seed(42)


# %%
# Useful Function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This function converts elevation-bearing detections to elevation-bearing-range detections.
# Todo - Not sure if this function should be belong in the main library.
def convert_eb_detections_to_ebr(detection: Detection, new_range: Union[None, float]) -> Detection:
    # noinspection PyTypeChecker
    original_measurement_model: CartesianToElevationBearing = detection.measurement_model
    new_measurement_model = CartesianToElevationBearingRange(
        ndim_state=original_measurement_model.ndim_state,
        mapping=original_measurement_model.mapping,
        noise_covar=np.pad(original_measurement_model.noise_covar, ((0, 1), (0, 1))),
        rotation_offset=original_measurement_model.rotation_offset,
        translation_offset=original_measurement_model.translation_offset)

    if new_range is None:
        if isinstance(detection, TrueDetection):
            new_measurement_vector = new_measurement_model.function(
                detection.groundtruth_path[detection.timestamp], noise=False)
            new_measurement_vector[0: 2] = detection.state_vector[0: 2]
        else:
            raise TypeError("Cannot retrieve ground truth path from detection")
    else:
        new_measurement_vector = StateVector(np.append(detection.state_vector, new_range))

    # noinspection PyTypeChecker
    return detection.from_state(detection,
                                state_vector=new_measurement_vector,
                                measurement_model=new_measurement_model)


# %%
# Create the Target Trajectories
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In this section the movement of the target aircraft is created. This specific geometry was
# chosen to challenge the association algorithms. Target 1 does a weave moving left to right.
# Target two moves down with a single turn at 40 seconds.

# Create waypoints of both targets
target_1_waypoints = [State([[t], [1], [40 + 10 * np.sin(t / 10)], [np.cos(t / 10)], [10], [0]],
                            start_time + datetime.timedelta(seconds=t)) for t in range(140)]

target_2_waypoints = [
    State([[40], [1], [130], [-1], [8], [0]], start_time + datetime.timedelta(seconds=0)),
    State([[80], [0], [90], [-1], [8], [0]], start_time + datetime.timedelta(seconds=40)),
    State([[80], [0], [-10], [-1], [8], [0]], start_time + datetime.timedelta(seconds=140))
]


# %%
# In between waypoints is interpolated to make a consistent sequence of states (locations). The
# `interpolate_states` function performs a linear interpolation between states to create new
# intermediate states.
def interpolate_states(existing_states: Sequence[State], interpolate_time: datetime.datetime):
    # An interpolate states function is due to enter stone-soup in PR #872.
    # Todo - If PR #872 is merged, replace this function with an import
    float_times = [state.timestamp.timestamp() for state in existing_states]
    output = np.zeros(len(existing_states[0].state_vector))
    for i in range(len(output)):
        a_states = [np.double(state.state_vector[i]) for state in existing_states]
        output[i] = np.interp(interpolate_time.timestamp(), float_times, a_states)

    return State(StateVector(output), timestamp=interpolate_time)


all_times = [start_time + datetime.timedelta(seconds=x) for x in range(140)]
target_1_states = []
target_2_states = []
target_3_states = []
for t in all_times:
    target_1_states.append(interpolate_states(target_1_waypoints, t))
    target_2_states.append(interpolate_states(target_2_waypoints, t))
    target_3_states.append(State([[120 + np.random.rand()], [0],
                                  [120 + np.random.rand()], [0], [12], [0]], timestamp=t))


# %%
# Create Sensors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Todo - Add Description

sensor_position = np.array([[50], [10], [0]])
static_sensor_orientation = StateVector([[0], [0], [np.pi / 2]])

sensor_platform_transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(1e-2), ConstantVelocity(1e-2), ConstantVelocity(1e-5)])

sensor1 = PassiveElevationBearing(
    ndim_state=6,
    mapping=(0, 2, 4),
    noise_covar=np.diag([np.deg2rad(1.5)**2,  # Elevation standard deviation
                         np.radians(1.5)**2   # Azimuth/bearing standard deviation
                         ]),
)
sensor_platform = MovingPlatform(
    states=[State([0, 1, 0, 0, 1, 0], start_time)],
    position_mapping=(0, 2, 4),
    velocity_mapping=(1, 3, 5),
    transition_model=sensor_platform_transition_model,
    sensors=[sensor1]
)


sensor2 = PassiveElevationBearing(
    position=sensor_position,
    ndim_state=6,
    mapping=(0, 2, 4),
    noise_covar=np.diag([np.deg2rad(1.5)**2,  # Elevation standard deviation
                         np.radians(1.5)**2   # Azimuth/bearing standard deviation
                         ]),
    orientation=static_sensor_orientation
)


# %%
# Measure the target
# -------------------------------------------------
# Todo - Add Description

target_1_truth = GroundTruthPath(id="Target 1")
target_2_truth = GroundTruthPath(id="Target 2")
target_3_truth = GroundTruthPath(id="Target 3")
all_ground_truth = {target_1_truth, target_2_truth, target_3_truth}

sensor_1_eb_truth = []
sensor_2_eb_truth = []

moving_sensor_inputs = []
static_sensor_inputs = []
for target_1_state, target_2_state, target_3_state in \
        zip(target_1_states, target_2_states, target_3_states):

    assert target_1_state.timestamp == target_2_state.timestamp == target_3_state.timestamp, \
        "Times should synchronised."
    time = target_1_state.timestamp

    target_1_truth.append(target_1_state)
    target_2_truth.append(target_2_state)
    target_3_truth.append(target_3_state)

    sensor_platform.move(time)

    moving_sensor_detections_this_time_step = sensor1.measure(all_ground_truth, noise=True)
    static_sensor_detections_this_time_step = sensor2.measure(all_ground_truth, noise=True)

    # This is used for angular truth data.
    sensor_1_eb_truth.extend(sensor1.measure(all_ground_truth, noise=False))
    sensor_2_eb_truth.extend(sensor2.measure(all_ground_truth, noise=False))

    moving_sensor_inputs.append((time, moving_sensor_detections_this_time_step))
    static_sensor_inputs.append((time, static_sensor_detections_this_time_step))


# %%
# Create Elevation-Bearing Truth for Each Target
# -------------------------------------------------
# Todo - Add Description

sensor_1_truth_dict = defaultdict(list)
for det in sensor_1_eb_truth:
    sensor_1_truth_dict[det.groundtruth_path].append(det)


sensor_2_truth_dict = defaultdict(list)
for det in sensor_2_eb_truth:
    sensor_2_truth_dict[det.groundtruth_path].append(det)


pre_rotate_dets = sensor_1_truth_dict[target_3_truth]

plotter_az_time2 = Plotter(xaxis=dict(title=dict(text="Time (seconds)")),
                           yaxis=dict(title=dict(text="Azimuth/Bearing Angle (radians)")))

plotter_az_time2.plot_measurements(pre_rotate_dets, mapping=[1],
                                   measurements_label="Angle-Only Measurements",
                                   convert_measurements=False)
# %%
# This graphs plots the **noiseless** detections from sensor on the moving platform (these
# detections were generated with ``noise=False``).

plotter_az_time2.fig


# %%
# Despite there being no noise generated in the measurement process there appears to be a lot of
# noise in the measurements. This is caused by the orientation of the moving platform changing on
# each time-step. The measurement values are relative to the platform's orientation. This
# introduces additional noise when tracking.

# %%
# To remove the additional noise caused by the platform's orientation. The detections measurements
# are rotated by :class:`.StaticRotationalFrameAngleDetectionFeeder`. The class also changes the
# rotation offset in the measurement model.
#
# In this example all the detections are rotated to same orientation as the same orientation as the
# static sensor (North/flat).


detection_rotater = StaticRotationalFrameAngleDetectionFeeder(
    reader=None, static_rotation_offset=static_sensor_orientation)


post_rotate_dets = [detection_rotater.alter_detection(det)
                    for det in pre_rotate_dets]

plotter_az_time3 = Plotter(xaxis=dict(title=dict(text="Time (seconds)")),
                           yaxis=dict(title=dict(text="Azimuth/Bearing Angle (radians)")))

plotter_az_time3.plot_measurements(post_rotate_dets, mapping=[1],
                                   measurements_label="Angle-Only Measurements",
                                   convert_measurements=False)
plotter_az_time3.fig

# %%
# This looks much better. The rotation isn't needed for the static sensor as its orientation
# doesn't change.
# Todo - Add More Description


sensor_1_eb_ground_truths = defaultdict(GroundTruthPath)
for det in sensor_1_eb_truth:
    sensor_1_eb_ground_truths[det.groundtruth_path].append(
        det.from_state(state=detection_rotater.alter_detection(det),
                       target_type=GroundTruthState)
    )

sensor_2_eb_ground_truths = defaultdict(GroundTruthPath)
for det in sensor_2_eb_truth:
    sensor_2_eb_ground_truths[det.groundtruth_path].append(
        det.from_state(state=detection_rotater.alter_detection(det),
                       target_type=GroundTruthState)
    )

for cart_ground_truth, eb_ground_truth in sensor_1_eb_ground_truths.items():
    eb_ground_truth.id = cart_ground_truth.id
for cart_ground_truth, eb_ground_truth in sensor_2_eb_ground_truths.items():
    eb_ground_truth.id = cart_ground_truth.id

# %%
# Convert all to detections to north


moving_sensor_inputs = [detection_rotater.alter_output(moving_sensor_input)
                        for moving_sensor_input in moving_sensor_inputs]


moving_sensor_measurements = [detection for _, set_of_detections in moving_sensor_inputs
                              for detection in set_of_detections]


static_sensor_measurements = [detection for _, set_of_detections in static_sensor_inputs
                              for detection in set_of_detections]


# %%
# Plot the Scenario
# --------------------------


plotter = Plotter(xaxis=dict(title=dict(text="<i>x</i> (East)")),
                  yaxis=dict(title=dict(text="<i>y</i> (North)"), scaleanchor="x", scaleratio=1))
for gt in all_ground_truth:
    plotter.plot_ground_truths({gt}, [_X, _Y], truths_label=gt.id)
# plotter.plot_ground_truths(all_ground_truth, [_X, _Y], truths_label="Target Flight Path")
plotter.plot_sensors([sensor2], sensor_label="Static Sensor Location")
plotter.plot_ground_truths({sensor_platform}, [_X, _Y], truths_label="Sensor Platform Flight Path")
plotter.fig

# %%
# This graphs show the two different flight paths for the targets. Target one has a sinusoidal
# flight path moving west to east. Target Two moves south-east and then turns 45 degrees to fly
# south. These flight paths look benign but can cause an issue with an angle-only sensor as you’ll
# see later in this example.

# %%
# Both sensors have 100% probability of detection. Both sensors have detected both targets
# at every time step.

plottable_static_sensor_detections = [convert_eb_detections_to_ebr(ao_det, None)
                                      for ao_det in static_sensor_measurements]

plottable_moving_sensor_detections = [convert_eb_detections_to_ebr(ao_det, None)
                                      for ao_det in moving_sensor_measurements]

plotter.plot_measurements(plottable_moving_sensor_detections, [_X, _Y],
                          measurements_label="Sensor 1 (Moving) Measurements")
plotter.plot_measurements(plottable_static_sensor_detections, [_X, _Y], marker=dict(color='orange'),
                          measurements_label="Sensor 2 (Static) Measurements")

# %%
# To aid visualisation the angle-only detections have been plotted with an artificial true range.
# This range isn’t used in tracking and is only used for plotting.

plotter.fig

# %%
# todo - talk about detections

plotterXZ = Plotter(xaxis=dict(title=dict(text="<i>x</i> (East)")),
                    yaxis=dict(title=dict(text="<i>z</i> (Altitude)")))
# plotterXZ.plot_ground_truths(all_ground_truth, [_X, _Z], truths_label="Target Flight Path")
for gt in all_ground_truth:
    plotterXZ.plot_ground_truths({gt}, [_X, _Z], truths_label=gt.id)
plotterXZ.fig


# %%
# The three targets are all at different altitudes.


plotterXZ.plot_measurements(plottable_moving_sensor_detections, [_X, _Z],
                            measurements_label="Sensor 1 (Moving) Measurements")
plotterXZ.plot_measurements(plottable_static_sensor_detections, [_X, _Z], marker=dict(color='orange'),
                            measurements_label="Sensor 2 (Static) Measurements")
plotterXZ.fig

# %%
# The graph above shows the altitude of the detections versus their east position.
# Todo - Talk about large variation in detection elevation


# %%
# Sensor 1 - Moving Sensor
# --------------------------


# %%
# Elevation vs Bearing Graph
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The detections will be plotted on an azimuth/bearing and elevation graph.
#
# Ground truth is
# generated in angle space by measuring the targets with a perfect sensor (a standard sensor with
# the inaccuracy (noise) set to zero).


plotterAzEl_1 = Plotter(xaxis=dict(title=dict(text="Azimuth/Bearing angle (radians)")),
                        yaxis=dict(title=dict(text="Elevation angle (radians)")))
plotterAzEl_1.fig.update_xaxes(autorange="reversed")

for gt in sensor_1_eb_ground_truths.values():
    plotterAzEl_1.plot_ground_truths({gt}, mapping=[1, 0], truths_label=gt.id)


plotterAzEl_1.plot_measurements(moving_sensor_measurements, mapping=[1, 0], measurements_label="Measurements",
                                convert_measurements=False)
plotterAzEl_1.fig

# %%
# Note: The x-axis has been reversed in all Azimuth-Bearing plots. This is because Stone-Soup
# measures the azimuth angle anti-clockwise from the x-axis (East). The axis has been reversed to
# match the movement of target 1 from left (west) to right (east) in the Cartesian graphs.


# %%
# Azimuth vs Time Graph
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
plotter_az_time_1 = Plotter(xaxis=dict(title=dict(text="Time (seconds)")),
                            yaxis=dict(title=dict(text="Azimuth/Bearing Angle (radians)")))

plotter_az_time_1.plot_measurements(moving_sensor_measurements, mapping=[1],
                                    measurements_label="Measurements",
                                    convert_measurements=False)

for gt in sensor_1_eb_ground_truths.values():
    plotter_az_time_1.plot_ground_truths({gt}, mapping=[1], truths_label=gt.id)

plotter_az_time_1.fig


# %%
# Sensor 2 - Static Sensor
# --------------------------


# %%
# Elevation vs Bearing Graph
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The detections will be plotted on an azimuth/bearing and elevation graph.
#
# Ground truth is
# generated in angle space by measuring the targets with a perfect sensor, a standard sensor with
# the inaccuracy (noise) set to zero.


plotterAzEl_2 = Plotter(xaxis=dict(title=dict(text="Azimuth/Bearing angle (radians)")),
                        yaxis=dict(title=dict(text="Elevation angle (radians)")))
plotterAzEl_2.fig.update_xaxes(autorange="reversed")

for gt in sensor_2_eb_ground_truths.values():
    plotterAzEl_2.plot_ground_truths({gt}, mapping=[1, 0], truths_label=gt.id)


plotterAzEl_2.plot_measurements(static_sensor_measurements, mapping=[1, 0], measurements_label="Measurements",
                                convert_measurements=False)
plotterAzEl_2.fig

# %%
# Note: The x-axis has been reversed in all Azimuth-Bearing plots. This is because Stone-Soup
# measures the azimuth angle anti-clockwise from the x-axis (East). The axis has been reversed to
# match the movement of target 1 from left (west) to right (east) in the Cartesian graphs.


# %%
# Azimuth vs Time Graph
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
plotter_az_time_2 = Plotter(xaxis=dict(title=dict(text="Time (seconds)")),
                            yaxis=dict(title=dict(text="Azimuth/Bearing Angle (radians)")))

plotter_az_time_2.plot_measurements(static_sensor_measurements, mapping=[1],
                                    measurements_label="Measurements",
                                    convert_measurements=False)

for gt in sensor_2_eb_ground_truths.values():
    plotter_az_time_2.plot_ground_truths({gt}, mapping=[1], truths_label=gt.id)

plotter_az_time_2.fig


# %%
# Tracking
# -------------------------------------------------
# In this section two trackers will be created:
#  * Moving Sensor Tracker
#  * Static Sensor Tracker,
# Both trackers will be the same. The
# trackers will be run and the sensor data will be processed. The output of the trackers will be
# explored in subsequent sections.


# %%
# Create Standard Tracker
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The standard tracker uses an Extended Kalman Filter (EKF). Global Nearest Neighbour (GNN) is
# used to associate detections to tracks. Tracks are initiated from any detections that aren’t
# associated to a track. Tracks will be deleted after 10 seconds without a detection being
# associated to them. The create_tracker_kwargs function generates the key word arguments for each
# tracker
def create_tracker_kwargs(transition_model, detector=None, ndim_state=None):
    ndim_state = ndim_state or (len(transition_model.model_list) * 2)  # Constant Velocity
    initial_state = GaussianState(state_vector=[0] * ndim_state,
                                  covar=np.diag([1000] * ndim_state))
    initiator = SimpleMeasurementInitiator(initial_state, measurement_model=None,
                                           skip_non_reversible=True)
    deleter = UpdateTimeDeleter(time_since_update=datetime.timedelta(seconds=10))
    predictor = ExtendedKalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=None)
    data_associator = GlobalNearestNeighbour(DistanceHypothesiser(
        predictor, updater, Mahalanobis(), missed_distance=10))

    return {"initiator": initiator,
            "deleter": deleter,
            "detector": detector,
            "data_associator": data_associator,
            "updater": updater}


# %%
# Create Tracker
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Both trackers uses the standard tracker inputs with a two-dimensional constant velocity
# model. The noise in the motion model is particularly small as the units are :math:`rad^2 s^{-3}`.
angular_motion_model_noise = 1e-6

transition_model_ao = CombinedLinearGaussianTransitionModel((
    ConstantVelocity(angular_motion_model_noise),
    ConstantVelocity(angular_motion_model_noise)))

tracker_moving_sensor = AngleMultipleTargetTracker(
    **create_tracker_kwargs(transition_model=transition_model_ao,
                            detector=None))

tracker_moving_sensor.detector = AngleTrackingDetectionFeeder(moving_sensor_inputs)


tracker_static_sensor = AngleMultipleTargetTracker(
    **create_tracker_kwargs(transition_model=transition_model_ao,
                            detector=None))

tracker_static_sensor.detector = AngleTrackingDetectionFeeder(static_sensor_inputs)

# %%
# Todo - Talk about the :class:`AngleMultipleTargetTracker`
#
#  * Almost identical to `MultipleTargetTracker`
#  * Only difference is the track

# %%
# Note - The :class:`AngleTrackingDetectionFeeder` uses the
# :class:`.StaticRotationalFrameAngleDetectionFeeder` to rotate the detections before they enter
# the tracker.


# %%
# Run Tracker
# ^^^^^^^^^^^^^
# Use the standard for-loop to run the trackers
def run_tracker(tracker):
    all_tracks = set()
    for _, tracks in tracker:
        all_tracks |= tracks
    return all_tracks


moving_sensor_tracks = list(run_tracker(tracker_moving_sensor))
static_sensor_tracks = list(run_tracker(tracker_static_sensor))


# %%
# Sensor 1 - Moving Sensor
# --------------------------


# %%
# Elevation vs Bearing Graph
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
for idx, track in enumerate(moving_sensor_tracks):
    plotterAzEl_1.plot_tracks({track}, mapping=[2, 0], track_label=f"Track {idx}")
plotterAzEl_1.fig

# %%
# Todo


# %%
# Azimuth vs Time Graph
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

for idx, track in enumerate(moving_sensor_tracks):
    plotter_az_time_1.plot_tracks({track}, mapping=[2], track_label=f"Track {idx}")

plotter_az_time_1.fig

# %%
# Sensor 2 - Static Sensor
# --------------------------


# %%
# Elevation vs Bearing Graph
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
for idx, track in enumerate(static_sensor_tracks):
    plotterAzEl_2.plot_tracks({track}, mapping=[2, 0], track_label=f"Track {idx}")
plotterAzEl_2.fig

# %%
# Todo


# %%
# Azimuth vs Time Graph
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

for idx, track in enumerate(static_sensor_tracks):
    plotter_az_time_2.plot_tracks({track}, mapping=[2], track_label=f"Track {idx}")

plotter_az_time_2.fig

# %%
# Tracking in Cartesian using Bearing Only
# ===========================================
# Why not track in cartesian?
# A small additional. Not much additional work


# %%
# Quick function to show only certain parts of the graph
def hide_plot_traces(fig, items_to_hide: set):
    for fig_data in fig.data:
        if fig_data.legendgroup in items_to_hide:
            fig_data.visible = "legendonly"
        else:
            fig_data.visible = None


# %%
# IR Tracks in 3D
# -----------------------
# Todo - Add AO with false range

# %%
# Single Sensor Tracking
# -----------------------
# Todo- info

# %%
# Create 3D Tracker
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The 3D tracker uses the standard tracker inputs with a three-dimensional constant velocity
# model.
transition_noise = 0.02
transition_model_xyz = CombinedLinearGaussianTransitionModel((
    ConstantVelocity(transition_noise),
    ConstantVelocity(transition_noise),
    ConstantVelocity(transition_noise)))


# %%
# There isn't currently a method in stone-soup to create Cartesian tracks from angle-only
# measurements. Therefore this functions **cheats** and uses the ground truth to create the
# initial state in the track. A large covariance is given to the initial :class:`~GaussianState`
# so that the initial cheat state has less influence on the subsequent states in the track.
def create_initial_tracks() -> Set[Track]:
    return {Track([State.from_state(state=ground_truth.states[0],
                                    target_type=GaussianState,
                                    covar=np.diag([100000] * 6))
                   ])
            for ground_truth in all_ground_truth}


# %%
# Sensor 1 (Moving)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tracker_moving_sensor_cart = MultiTargetTracker(**create_tracker_kwargs(
    transition_model=transition_model_xyz,
    detector=moving_sensor_inputs))

# %%
# Cheat - Create Initial Tracks
initial_tracks = create_initial_tracks()
tracker_moving_sensor_cart._tracks = initial_tracks


moving_sensor_tracks_cart = list(run_tracker(tracker_moving_sensor_cart))

plotter.plot_tracks(moving_sensor_tracks_cart, [_X, _Y], uncertainty=False, track_label="Sensor 1 Tracker")
plotter.fig

# %%
# The tracks are rubbish
hide_plot_traces(plotter.fig, {"Sensor 1 Tracker"})

# %%
# Sensor 2 (Static)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tracker_static_sensor_cart = MultiTargetTracker(**create_tracker_kwargs(
    transition_model=transition_model_xyz,
    detector=static_sensor_inputs))

# %%
# Cheat - Create Initial Tracks
initial_tracks = create_initial_tracks()
tracker_static_sensor_cart._tracks = initial_tracks


static_sensor_tracks_cart = list(run_tracker(tracker_static_sensor_cart))

plotter.plot_tracks(static_sensor_tracks_cart, [_X, _Y], uncertainty=False, track_label="Sensor 2 Tracker")
plotter.fig

# %%
# The tracks are even worst!
hide_plot_traces(plotter.fig, {"Sensor 1 Tracker", "Sensor 2 Tracker"})


# %%
# Create Detection Fusion Tracker
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The Detection Fusion Tracker uses an identical tracker to other cartesian trackers. The only
# difference is that the detection fusion tracker receives detections from both sensors.


# Combine all measurement input
all_inputs = sorted([*moving_sensor_inputs, *static_sensor_inputs])

# Create the tracker
det_fuse_tracker = MultiTargetTracker(**create_tracker_kwargs(
    transition_model=transition_model_xyz,
    detector=all_inputs)
)

# %%
# Cheat - Create Initial Tracks
initial_tracks = create_initial_tracks()
det_fuse_tracker._tracks = initial_tracks

det_tracks = list(run_tracker(det_fuse_tracker))

plotter.plot_tracks(det_tracks, [_X, _Y], uncertainty=False,
                    track_label="Detection Fusion Tracker")
plotter.fig

# %%
# These tracks are better.
# Todo - Talk about having multiple angle-only perspectives on a track make the track much more
# stable.
