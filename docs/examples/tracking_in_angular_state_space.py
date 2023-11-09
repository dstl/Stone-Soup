"""
Tracking in Angular State Space Example
========================================

This example demonstrates the use of an :class:`~.AngleMultipleTargetTracker` class to create
tracks from angle only detections made by :class:`~PassiveElevationBearing` sensors. Two sensors
are used, one static and one moving, to compare the tracks produced on three targets.
As an extension the angle only measurements are used with a :class:`~MultiTargetTracker` to produce
tracks in cartesian space.
"""

# %%
# Build the scenario
# -------------------------------------------------

# %%
# First, all the modules needed for this script are imported, some initial parameters are
# created, and the numpy random seed is set (this is to ensure consistent output from the script).

import datetime
from collections import defaultdict
from typing import Sequence, Union, Set

import numpy as np

from stonesoup.dataassociator.neighbour import GlobalNearestNeighbour
from stonesoup.deleter.time import UpdateTimeDeleter
from stonesoup.feeder.modify import AngleTrackingDetectionFeeder, \
    StaticRotationalFrameAngleDetectionFeeder
from stonesoup.functions.interpolate import interpolate_state_mutable_sequence
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
# This function converts elevation-bearing detections to elevation-bearing-range detections for the
# purpose of visualising the detections made by the sensors.
#
# TODO - Not sure if this function should be in the main library.
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
# Target 2 moves down with a single turn at 40 seconds.

# Create waypoints of both targets
target_1_waypoints = [State([[t], [1], [40 + 10 * np.sin(t / 10)], [np.cos(t / 10)], [10], [0]],
                            start_time + datetime.timedelta(seconds=t)) for t in range(140)]

target_2_waypoints = [
    State([[40], [1], [130], [-1], [8], [0]], start_time + datetime.timedelta(seconds=0)),
    State([[80], [0], [90], [-1], [8], [0]], start_time + datetime.timedelta(seconds=40)),
    State([[80], [0], [-10], [-1], [8], [0]], start_time + datetime.timedelta(seconds=140))
]


# %%
# The timesteps in between waypoints are interpolated to make a consistent sequence of states
# (locations). The `interpolate_states_mutable_sequence` function performs a linear interpolation
# between states to create new intermediate states. A third target is also created which remains
# static.


all_times = [start_time + datetime.timedelta(seconds=x) for x in range(140)]
target_1_states = interpolate_state_mutable_sequence(target_1_waypoints, all_times)
target_2_states = interpolate_state_mutable_sequence(target_1_waypoints, all_times)
target_3_states = []
for t in all_times:
    target_3_states.append(State([[120 + np.random.rand()], [0],
                                  [120 + np.random.rand()], [0], [12], [0]], timestamp=t))


# %%
# Create Sensors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Here we create two angle only sensors using the :class:`~PassiveElevationBearing` class. The
# first sensor is attached to a moving platform while the second sensor is static at the position
# (50, 10, 0).

sensor_position = np.array([[50], [10], [0]])
static_sensor_orientation = StateVector([[0], [0], [np.pi / 2]])

sensor_platform_transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(1e-2), ConstantVelocity(1e-2), ConstantVelocity(1e-5)])

moving_sensor = PassiveElevationBearing(
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
    sensors=[moving_sensor]
)


static_sensor = PassiveElevationBearing(
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
# Now the truth paths can be measured to obtain the angle only detections. Each sensor measures
# each ground truth twice, once with noise to provide the actual detections and once without noise
# to create elevation-bearing ground truth paths.

target_1_truth = GroundTruthPath(id="Target 1")
target_2_truth = GroundTruthPath(id="Target 2")
target_3_truth = GroundTruthPath(id="Target 3")
all_ground_truth = {target_1_truth, target_2_truth, target_3_truth}

moving_sensor_eb_truth = []
static_sensor_eb_truth = []

moving_sensor_inputs = []
static_sensor_inputs = []
for target_1_state, target_2_state, target_3_state in \
        zip(target_1_states, target_2_states, target_3_states):

    assert target_1_state.timestamp == target_2_state.timestamp == target_3_state.timestamp, \
        "Times should be synchronised."
    time = target_1_state.timestamp

    target_1_truth.append(target_1_state)
    target_2_truth.append(target_2_state)
    target_3_truth.append(target_3_state)

    sensor_platform.move(time)

    moving_sensor_detections_this_time_step = moving_sensor.measure(all_ground_truth, noise=True)
    static_sensor_detections_this_time_step = static_sensor.measure(all_ground_truth, noise=True)

    # This is used for angular truth data.
    moving_sensor_eb_truth.extend(moving_sensor.measure(all_ground_truth, noise=False))
    static_sensor_eb_truth.extend(static_sensor.measure(all_ground_truth, noise=False))

    moving_sensor_inputs.append((time, moving_sensor_detections_this_time_step))
    static_sensor_inputs.append((time, static_sensor_detections_this_time_step))

xy_plotter = Plotter(xaxis=dict(title=dict(text="<i>x</i> (Easting)")),
                     yaxis=dict(title=dict(text="<i>y</i> (Northing)"),
                                scaleanchor="x", scaleratio=1))
for gt in all_ground_truth:
    xy_plotter.plot_ground_truths({gt}, [_X, _Y], truths_label=gt.id)
xy_plotter.plot_sensors([static_sensor], sensor_label="Static Sensor Location")
xy_plotter.plot_ground_truths({sensor_platform}, [_X, _Y],
                              truths_label="Sensor Platform Flight Path")
xy_plotter.fig

# %%
# This graph shows the three different flight paths for the targets. Target 1 has a sinusoidal
# flight path moving west to east. Target 2 moves south-east and then turns 45 degrees to fly
# south. Target 3 is static. These flight paths look benign but can cause an issue with an
# angle-only sensor as you’ll see later in this example.

# %%
# Create Elevation-Bearing Truth for Each Target
# -------------------------------------------------
# We now assign each angle only truth detection to its corresponding ground truth.

moving_sensor_truth_dict = defaultdict(list)
for det in moving_sensor_eb_truth:
    moving_sensor_truth_dict[det.groundtruth_path].append(det)


static_sensor_truth_dict = defaultdict(list)
for det in static_sensor_eb_truth:
    static_sensor_truth_dict[det.groundtruth_path].append(det)


pre_rotate_dets = moving_sensor_truth_dict[target_3_truth]

bt_plotter = Plotter(xaxis=dict(title=dict(text="Time (seconds)")),
                     yaxis=dict(title=dict(text="Bearing Angle (radians)")))

bt_plotter.plot_measurements(pre_rotate_dets, mapping=[1],
                             measurements_label="Angle-Only Measurements",
                             convert_measurements=False)
# %%
# This graph plots the **noiseless** detections from the sensor on the moving platform (these
# detections were generated with ``noise=False``).

bt_plotter.fig


# %%
# Despite there being no noise generated in the measurement process, there appears to be a lot of
# noise in the measurements. This is caused by the orientation of the moving platform changing on
# each time-step. The measurement values are relative to the platform's orientation. This
# introduces additional noise when tracking.

# %%
# To remove the additional noise caused by the platform's orientation. The detection measurements
# are rotated by a :class:`.StaticRotationalFrameAngleDetectionFeeder`. This class also changes the
# rotation offset in the measurement model.
#
# In this example all the detections are rotated to have the same orientation as the static sensor
# (north/flat).


detection_rotater = StaticRotationalFrameAngleDetectionFeeder(
    reader=None, static_rotation_offset=static_sensor_orientation)


post_rotate_dets = [detection_rotater.alter_detection(det)
                    for det in pre_rotate_dets]

bt_plotter_fixed = Plotter(xaxis=dict(title=dict(text="Time (seconds)")),
                           yaxis=dict(title=dict(text="Bearing Angle (radians)")))

bt_plotter_fixed.plot_measurements(post_rotate_dets, mapping=[1],
                                   measurements_label="Angle-Only Measurements",
                                   convert_measurements=False)
bt_plotter_fixed.fig

# %%
# This looks much better. The rotation isn't needed for the static sensor as its orientation
# doesn't change.
#
# The detections now show much more consistency. This rotation is not required for the static
# sensor as it has a constant orientation.


moving_sensor_eb_ground_truths = defaultdict(GroundTruthPath)
for det in moving_sensor_eb_truth:
    moving_sensor_eb_ground_truths[det.groundtruth_path].append(
        det.from_state(state=detection_rotater.alter_detection(det),
                       target_type=GroundTruthState)
    )

# TODO query? just said doesnt need to be done for static sensor
static_sensor_eb_ground_truths = defaultdict(GroundTruthPath)
for det in static_sensor_eb_truth:
    static_sensor_eb_ground_truths[det.groundtruth_path].append(
        det.from_state(state=detection_rotater.alter_detection(det),
                       target_type=GroundTruthState)
    )

for cart_ground_truth, eb_ground_truth in moving_sensor_eb_ground_truths.items():
    eb_ground_truth.id = cart_ground_truth.id
for cart_ground_truth, eb_ground_truth in static_sensor_eb_ground_truths.items():
    eb_ground_truth.id = cart_ground_truth.id

# %%
# Convert all detections to be from north.


moving_sensor_inputs = [detection_rotater.alter_output(moving_sensor_input)
                        for moving_sensor_input in moving_sensor_inputs]


moving_sensor_measurements = [detection for _, set_of_detections in moving_sensor_inputs
                              for detection in set_of_detections]


static_sensor_measurements = [detection for _, set_of_detections in static_sensor_inputs
                              for detection in set_of_detections]


# %%
# Plot the Scenario
# --------------------------

# %%
# Both sensors have 100% probability of detection and so have detected all targets at every time
# step.

plottable_static_sensor_detections = [convert_eb_detections_to_ebr(ao_det, None)
                                      for ao_det in static_sensor_measurements]

plottable_moving_sensor_detections = [convert_eb_detections_to_ebr(ao_det, None)
                                      for ao_det in moving_sensor_measurements]

xy_plotter.plot_measurements(plottable_moving_sensor_detections, [_X, _Y],
                             measurements_label="Moving Sensor Measurements")
xy_plotter.plot_measurements(plottable_static_sensor_detections, [_X, _Y],
                             marker=dict(color='orange'),
                             measurements_label="Static Sensor Measurements")

# %%
# To aid visualisation the angle-only detections have been plotted with an artificial true range.
# This range isn’t used in tracking and is only used for plotting.

xy_plotter.fig

# %%
# This graph shows the detections from both sensors against the paths of all three tracks in x-y
# cartesian space. While the detections match the paths of targets 1 and 2, the error in the
# measurements can be most clearly seen by the variation on the detections made for the static
# target 3.

xz_plotter = Plotter(xaxis=dict(title=dict(text="<i>x</i> (Easting)")),
                     yaxis=dict(title=dict(text="<i>z</i> (Altitude)")))
for gt in all_ground_truth:
    xz_plotter.plot_ground_truths({gt}, [_X, _Z], truths_label=gt.id)
xz_plotter.fig


# %%
# The three targets are all at different altitudes.


xz_plotter.plot_measurements(plottable_moving_sensor_detections, [_X, _Z],
                             measurements_label="Sensor 1 (Moving) Measurements")
xz_plotter.plot_measurements(plottable_static_sensor_detections, [_X, _Z],
                             marker=dict(color='orange'),
                             measurements_label="Sensor 2 (Static) Measurements")
xz_plotter.fig

# %%
# The graph above shows the altitude of the detections versus their easting. From this perspective,
# the error in detections for all three targets becomes more apparent. There are large variations
# in altitude measurements for each target despite all three targets holding constant elevations.

# %%
# Moving Sensor - Detections
# --------------------------


# %%
# Elevation vs Bearing Graph
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The detections will be plotted on an elevation-bearing graph.
#
# The ground truths have been generated in angle space through the use of a perfect sensor (a
# standard sensor with the inaccuracy (noise) set to zero).


eb_plotter_moving = Plotter(xaxis=dict(title=dict(text="Bearing angle (radians)")),
                            yaxis=dict(title=dict(text="Elevation angle (radians)")))
eb_plotter_moving.fig.update_xaxes(autorange="reversed")

for gt in moving_sensor_eb_ground_truths.values():
    eb_plotter_moving.plot_ground_truths({gt}, mapping=[1, 0], truths_label=gt.id)


eb_plotter_moving.plot_measurements(moving_sensor_measurements, mapping=[1, 0],
                                    measurements_label="Measurements",
                                    convert_measurements=False)
eb_plotter_moving.fig

# %%
# From this graph we can see that when the tracks are separated, the detections follow the paths of
# the targets clearly. However, when the tracks start to overlap it becomes much less clear which
# detection was made for which track.

# %%
# Note: The x-axis has been reversed in all bearing plots. This is because Stone-Soup
# measures the azimuth angle anti-clockwise from the x-axis (East). The axis has been reversed to
# match the movement of target 1 from left (west) to right (east) in the Cartesian graphs.

# %%
# Bearing vs Time Graph
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
bt_plotter_moving = Plotter(xaxis=dict(title=dict(text="Time (seconds)")),
                            yaxis=dict(title=dict(text="Bearing Angle (radians)")))

bt_plotter_moving.plot_measurements(moving_sensor_measurements, mapping=[1],
                                    measurements_label="Measurements",
                                    convert_measurements=False)

for gt in moving_sensor_eb_ground_truths.values():
    bt_plotter_moving.plot_ground_truths({gt}, mapping=[1], truths_label=gt.id)

bt_plotter_moving.fig

# %%
# The bearing-time graph shows a similar trend. The general shape of each path is shown in
# the detections, but when the tracks become close to each other an association of detections to
# targets becomes very hard to do. There are multiple times in this simulation where the bearings
# overlap causing the detections to become unclear.

# %%
# Elevation vs Time Graph
# ^^^^^^^^^^^^^^^^^^^^^^^
et_plotter_moving = Plotter(xaxis=dict(title=dict(text="Time (seconds)")),
                            yaxis=dict(title=dict(text="Elevation Angle (radians)")))
et_plotter_moving.plot_measurements(moving_sensor_measurements, mapping=[0],
                                    measurements_label="Measurements",
                                    convert_measurements=False)

for gt in moving_sensor_eb_ground_truths.values():
    et_plotter_moving.plot_ground_truths({gt}, mapping=[0], truths_label=gt.id)

et_plotter_moving.fig

# %%
# This elevation-time graph shows that, for the majority of the simulation, the elevation
# measurements of targets 2 and 3 overlap causing the detections to be cluttered. The elevation
# measurements of target 1 begin to become tangled with the other targets during the second half.


# %%
# Static Sensor - Detections
# --------------------------


# %%
# Elevation vs Bearing Graph
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The detections will be plotted on an elevation-bearing graph.


eb_plotter_static = Plotter(xaxis=dict(title=dict(text="Bearing angle (radians)")),
                            yaxis=dict(title=dict(text="Elevation angle (radians)")))
eb_plotter_static.fig.update_xaxes(autorange="reversed")

for gt in static_sensor_eb_ground_truths.values():
    eb_plotter_static.plot_ground_truths({gt}, mapping=[1, 0], truths_label=gt.id)


eb_plotter_static.plot_measurements(static_sensor_measurements, mapping=[1, 0],
                                    measurements_label="Measurements",
                                    convert_measurements=False)
eb_plotter_static.fig

# %%
# When compared to the moving sensor, the static sensor has a much clearer association between
# detections and paths. This is due to the fact that the paths of targets 1 and 2 cross each other
# further away from target 3. However, at these intersections there are still large groupings of
# ambiguous detections.


# %%
# Bearing vs Time Graph
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
bt_plotter_static = Plotter(xaxis=dict(title=dict(text="Time (seconds)")),
                            yaxis=dict(title=dict(text="Bearing Angle (radians)")))

bt_plotter_static.plot_measurements(static_sensor_measurements, mapping=[1],
                                    measurements_label="Measurements",
                                    convert_measurements=False)

for gt in static_sensor_eb_ground_truths.values():
    bt_plotter_static.plot_ground_truths({gt}, mapping=[1], truths_label=gt.id)

bt_plotter_static.fig

# %%
# This graph shows that the bearing coordinates of all three targets intersect at around the same
# time. This causes a large amount of the detections to be hard to trace back to the original
# target.

# %%
# Elevation vs Time Graph
# ^^^^^^^^^^^^^^^^^^^^^^^
et_plotter_static = Plotter(xaxis=dict(title=dict(text="Time (seconds)")),
                            yaxis=dict(title=dict(text="Elevation Angle (radians)")))
et_plotter_static.plot_measurements(static_sensor_measurements, mapping=[0],
                                    measurements_label="Measurements",
                                    convert_measurements=False)

for gt in static_sensor_eb_ground_truths.values():
    et_plotter_static.plot_ground_truths({gt}, mapping=[0], truths_label=gt.id)

et_plotter_static.fig
# %%
# Similar to the elevation-time graph for the moving sensor, the detections of targets 2 and 3
# have significant overlap. Target 1 also experiences a large amount of overlap in detections
# towards the end of the simulation.


# %%
# Tracking
# -------------------------------------------------
# In this section two identical trackers will be created, one for each of the sensors. The trackers
# will be run and the sensor data will be processed. The output of the trackers will be explored in
# subsequent sections.


# %%
# Create Standard Tracker
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The standard tracker uses an Extended Kalman Filter (EKF). Global Nearest Neighbour (GNN) is
# used to associate detections to tracks. Tracks are initiated from any detections that aren’t
# associated to a track. Tracks will be deleted after 10 seconds without a detection being
# associated to them. The `create_tracker_kwargs` function generates the key word arguments for
# each tracker.
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
# Both trackers use the standard tracker inputs with a two-dimensional constant velocity model. The
# noise in the motion model is particularly small as the units are :math:`rad^2 s^{-3}`.
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
# The :class:`~.AngleMultipleTargetTracker` sits on top of the :class:`~.MultipleTargetTracker`
# class and produces track outputs in an elevation-bearing state space from angle only detections.

# %%
# Note: The :class:`AngleTrackingDetectionFeeder` uses the
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
# Moving Sensor - Tracks
# --------------------------


# %%
# Elevation vs Bearing Graph
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
for idx, track in enumerate(moving_sensor_tracks):
    eb_plotter_moving.plot_tracks({track}, mapping=[2, 0], track_label=f"Track {idx}")
eb_plotter_moving.fig

# %%
# From the elevation-bearing graph we can see that the general shape of the tracks follows that of
# the targets truth paths. Although, around target 1 there are large disparities between the
# position of the target and the track.
#
# TODO add description
#
# TODO can't talk about overlap / lack of overlap in tracks as different builds have variance

# %%
# Bearing vs Time Graph
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

for idx, track in enumerate(moving_sensor_tracks):
    bt_plotter_moving.plot_tracks({track}, mapping=[2], track_label=f"Track {idx}")

bt_plotter_moving.fig

# %%
# The bearing-time graph shows that very little variance between the bearing of the tracks and
# those of the truth paths.
#
# TODO Add description

# %%
# Elevation vs Time Graph
# ^^^^^^^^^^^^^^^^^^^^^^^

for idx, track in enumerate(moving_sensor_tracks):
    et_plotter_moving.plot_tracks({track}, mapping=[0], track_label=f"Track {idx}")

et_plotter_moving.fig

# %%
# The elevation-time graph shows much larger variances from the truth path than was apparent in the
# bearing-time graph.
#
# TODO add description

# %%
# Static Sensor - Tracks
# --------------------------


# %%
# Elevation vs Bearing Graph
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
for idx, track in enumerate(static_sensor_tracks):
    eb_plotter_static.plot_tracks({track}, mapping=[2, 0], track_label=f"Track {idx}")
eb_plotter_static.fig

# %%
# We can see that, like the moving tracker, the general paths of each target have been tracked
# accurately. Like before, there are visible disparities between the tracks and truth for target 1.
#
# TODO add description


# %%
# Bearing vs Time Graph
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

for idx, track in enumerate(static_sensor_tracks):
    bt_plotter_static.plot_tracks({track}, mapping=[2], track_label=f"Track {idx}")

bt_plotter_static.fig

# %%
# The bearing-time graph shows a minuscule disparity between the truth paths of the targets and the
# tracks produced.
#
# TODO Add description

# %%
# Elevation vs Time Graph
# ^^^^^^^^^^^^^^^^^^^^^^^

for idx, track in enumerate(static_sensor_tracks):
    et_plotter_static.plot_tracks({track}, mapping=[0], track_label=f"Track {idx}")

et_plotter_static.fig

# %%
# However, as was the case for the moving sensor, there exists a much larger disparity in the
# elevation of the tracks when compared to the target truths.
#
# TODO add description

# %%
# Tracking in Cartesian using Bearing Only
# ----------------------------------------
# As an extension we can take the angle only detections and try to produce tracks in a cartesian
# space.


# %%
# This function makes it so that only certain parts of the graph are shown.
def hide_plot_traces(fig, items_to_hide: set):
    for fig_data in fig.data:
        if fig_data.legendgroup in items_to_hide:
            fig_data.visible = "legendonly"
        else:
            fig_data.visible = None


# %%
# Single Sensor Tracking
# ^^^^^^^^^^^^^^^^^^^^^^
# The first thing to do is to produce cartesian tracks using detections from only one sensor. These
# tracks will be created using the :class:`~.MultiTargetTracker` class.

# %%
# Create 3D Tracker
# """""""""""""""""
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
# Moving Sensor
# """""""""""""
tracker_moving_sensor_cart = MultiTargetTracker(**create_tracker_kwargs(
    transition_model=transition_model_xyz,
    detector=moving_sensor_inputs))

initial_tracks = create_initial_tracks()
tracker_moving_sensor_cart._tracks = initial_tracks


moving_sensor_tracks_cart = list(run_tracker(tracker_moving_sensor_cart))

xy_plotter.plot_tracks(moving_sensor_tracks_cart, [_X, _Y], uncertainty=False,
                       track_label="Moving Sensor Tracker")
xy_plotter.fig

# %%
# As we can see, these tracks do not bear any resemblance to the actual shape of the paths.

hide_plot_traces(xy_plotter.fig, {"Moving Sensor Tracker"})

# %%
# Static Sensor
# """""""""""""
tracker_static_sensor_cart = MultiTargetTracker(**create_tracker_kwargs(
    transition_model=transition_model_xyz,
    detector=static_sensor_inputs))

initial_tracks = create_initial_tracks()
tracker_static_sensor_cart._tracks = initial_tracks


static_sensor_tracks_cart = list(run_tracker(tracker_static_sensor_cart))

xy_plotter.plot_tracks(static_sensor_tracks_cart, [_X, _Y],
                       uncertainty=False, track_label="Sensor 2 Tracker")
xy_plotter.fig

# %%
# The tracks produced by the static sensor vary from the truth even more.

hide_plot_traces(xy_plotter.fig, {"Sensor 1 Tracker", "Sensor 2 Tracker"})

# %%
# Multi Sensor Tracking
# ^^^^^^^^^^^^^^^^^^^^^
# We will now fuse the detections from both sensors to produce cartesian tracks.

# %%
# Create Detection Fusion Tracker
# """""""""""""""""""""""""""""""
# The Detection Fusion Tracker uses an identical tracker to the other cartesian trackers. The only
# difference is that the detection fusion tracker receives detections from both sensors.


# Combine all measurement input
all_inputs = sorted([*moving_sensor_inputs, *static_sensor_inputs])

# Create the tracker
det_fuse_tracker = MultiTargetTracker(**create_tracker_kwargs(
    transition_model=transition_model_xyz,
    detector=all_inputs)
)

initial_tracks = create_initial_tracks()
det_fuse_tracker._tracks = initial_tracks

det_fuse_tracks = list(run_tracker(det_fuse_tracker))

xy_plotter.plot_tracks(det_fuse_tracks, [_X, _Y],
                       uncertainty=False, track_label="Detection Fusion Tracker")
xy_plotter.fig

# %%
# These tracks are better. This is due to having angle-only detections from multiple positions.
# This makes it so that the positions of each target are possible to calculate and so the tracks
# become much more stable. Despite this, there is still a large disparity between the shapes of the
# tracks and the shapes of the target truths.
