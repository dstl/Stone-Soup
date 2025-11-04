#!/usr/bin/env python
# coding: utf-8

"""
Estimating Bias Between Sensors
===============================
"""
# %%
# This example demonstrates how to simulate and estimate a drifting bias in the position of a sensor platform.
# Specifically, the platform at index 0 (and its sensor) will have a time-varying bias applied to its position.
# We use Stone-Soup's bias wrappers, feeders and updater to estimate this changing bias from sensor measurements.

# Some initial imports and set up
import datetime
import numpy as np

np.random.seed(2001)
start_time = datetime.datetime.now().replace(microsecond=0)

# %%
# Define Platforms and Sensors
# ----------------------------
#
# We create three moving platforms, each with a radar sensor. The first platform will have a drifting bias applied.
from stonesoup.models.transition.linear import \
    RandomWalk, ConstantVelocity, OrnsteinUhlenbeck, CombinedLinearGaussianTransitionModel
from stonesoup.platform import MovingPlatform
from stonesoup.sensor.radar.radar import RadarBearingRange
from stonesoup.types.state import State, GaussianState
from stonesoup.types.track import Track
from stonesoup.types.array import StateVector
from stonesoup.plotter import Plotterly
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
platforms = [
    MovingPlatform(
        states=State(
            StateVector([-10., 1., -10., 1.]),
            timestamp=start_time
        ),
        position_mapping=[0, 2],
        transition_model=CombinedLinearGaussianTransitionModel([OrnsteinUhlenbeck(0.1, 5e-1)]*2),
    ),
    MovingPlatform(
        states=State(
            StateVector([20., -1., 20., -1.]),
            timestamp=start_time
        ),
        position_mapping=[0, 2],
        transition_model=CombinedLinearGaussianTransitionModel([OrnsteinUhlenbeck(0.1, 5e-1)]*2),
    ),
    MovingPlatform(
        states=State(
            StateVector([-20., -1., -30., -1.]),
            timestamp=start_time
        ),
        position_mapping=[0, 2],
        transition_model=CombinedLinearGaussianTransitionModel([OrnsteinUhlenbeck(0.1, 5e-1)]*2),
    )
]

sensors = [
    RadarBearingRange(
        ndim_state=4, position_mapping=[0, 2], noise_covar=np.diag([np.radians(0.05), 0.5])),
    RadarBearingRange(
        ndim_state=4, position_mapping=[0, 2], noise_covar=np.diag([np.radians(0.05), 0.5])),
    RadarBearingRange(
        ndim_state=4, position_mapping=[0, 2], noise_covar=np.diag([np.radians(0.05), 0.5])),
]
# %%
# Attach Sensors to Platforms
for platform, sensor in zip(platforms, sensors):
    platform.add_sensor(sensor)

# %%
# Add Targets
# -----------
#
# We add several moving targets to the scenario, each with its own motion model.
targets = {
    MovingPlatform(
        states=State(
            StateVector([i * 5, 0.1*np.sign(i), i * 5, -0.1*np.sign(i+1)]),
            timestamp=start_time
        ),
        position_mapping=[0, 2],
        transition_model=CombinedLinearGaussianTransitionModel([ConstantVelocity(0.01)]*2),
    )
    for i in range(-3, 4)
}

# %%
# Simulate Platform Motion and Sensor Measurements
# ------------------------------------------------
#
# We simulate the motion of each platform and generate sensor measurements for each target.
# The first platform's sensor measurements will be affected by a drifting bias.
#
# We create a time-varying bias using a random walk model, and apply this bias to the measurements
# of platform 0.
true_bias_prior = State([[5.], [5.]], start_time)
bias_transition_model = CombinedLinearGaussianTransitionModel([RandomWalk(1e-2)]*2)
true_bias = GroundTruthPath([true_bias_prior])

# %%
# Simulate platforms and measurements including bias for platform 0
#
ground_truths = [GroundTruthPath() for _ in platforms]

timestamps = [start_time + datetime.timedelta(seconds=n) for n in range(1, 51)]
measurements = [[] for _ in sensors]

for time in timestamps:
    # Update the true bias using the transition model
    true_bias.append(State.from_state(
        true_bias.state,
        state_vector=bias_transition_model.function(
            true_bias, noise=True, time_interval=time - true_bias.timestamp),
        timestamp=time))
    for target in targets:
        target.move(timestamp=time)
    for platform_index, platform in enumerate(platforms):
        platform.move(noise=True, timestamp=time)

        # Add ground truth state for each platform
        ground_truth_state = GroundTruthState(platform.state_vector, timestamp=time)
        ground_truths[platform_index].append(ground_truth_state)

        # Generate measurement for each platform
        measurements[platform_index].append((time, (meas := platform.sensors[0].measure(targets))))
        # Apply drifting bias to platform 0's sensor measurements
        if platform_index == 0:
            for model in {m.measurement_model for m in meas}:
                model.translation_offset = model.translation_offset + true_bias.state_vector


# %%
# Visualise Ground Truths and Measurements
# ----------------------------------------
# We plot the ground truth positions of platforms and targets, and the sensor measurements
# (with bias for platform 0 in green).
plotter = Plotterly()
plotter.plot_ground_truths(ground_truths, mapping=[0, 2], line_dash="solid", label="Platforms")
plotter.plot_ground_truths(targets, mapping=[0, 2])
for n, sensor_measurements in enumerate(measurements):
    kwargs = {}
    if n == 0:
        kwargs['marker'] = {'color': 'green'}
    plotter.plot_measurements(
        {m for ms in sensor_measurements for m in ms[1]},
        mapping=[0, 2],
        label=f'Sensor {n}',
        **kwargs)
plotter.fig

# %%
# Initialise Bias Estimation
# --------------------------
# We set up the bias feeder and predictor to apply the drifting bias from platform 0's sensor
# measurements.
#
# These are  all added to a MultiDataFeeder to combine them into single detection feed.
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.feeder.bias import TranslationBiasFeeder
from stonesoup.feeder.multi import MultiDataFeeder

bias_state = GaussianState([[0.], [0.]], np.diag([5**2, 5**2]), start_time)
bias_track = Track([bias_state])

bias_predictor = KalmanPredictor(CombinedLinearGaussianTransitionModel([RandomWalk(1e-1)]*2))
bias_feeder = TranslationBiasFeeder(measurements[0], bias_track)

# %%
# These are  all added to a MultiDataFeeder to combine them into single detection feed.
feeder = MultiDataFeeder([*measurements[1:], bias_feeder])

# %%
# Run Tracking and Bias Estimation
# --------------------------------
# We use an Extended Kalman Predictor and Unscented Kalman Updater for tracking, and associate
# measurements using a global nearest neighbour associator.
from stonesoup.predictor.kalman import KalmanPredictor, ExtendedKalmanPredictor
predictor = ExtendedKalmanPredictor(
    CombinedLinearGaussianTransitionModel([ConstantVelocity(0.01)]*2))

from stonesoup.updater.kalman import UnscentedKalmanUpdater
updater = UnscentedKalmanUpdater(None)

from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=5)

from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
data_associator = GNNWith2DAssignment(hypothesiser)

# %%
# A bias aware hypothesiser and data associator are created to factor the bias uncertainty into
# association threshold. These use bias updater wrapper, which is also used to update target
# and bias estimates.
from stonesoup.updater.bias import GaussianBiasUpdater
from stonesoup.models.measurement.bias import TranslationBiasModelWrapper

bias_updater = GaussianBiasUpdater(
    bias_track, bias_predictor, TranslationBiasModelWrapper, updater)
bias_hypothesiser = DistanceHypothesiser(
    predictor, bias_updater, measure=Mahalanobis(), missed_distance=5)
bias_data_associator = GNNWith2DAssignment(bias_hypothesiser)

# %%
# Tracks will be initialised by taking first observation from unbiased sensor
from stonesoup.initiator.simple import SinglePointMeasurementInitiator
initiator = SinglePointMeasurementInitiator(
    GaussianState([0., 0., 0., 0.], np.diag([0., 1., 0., 1.]))
)
tracks = initiator.initiate(sensors[1].measure({t[0] for t in targets}, noise=False), start_time)

# %%
# For each time step, we associate measurements to tracks, update the bias estimate,
# and update the tracks accordingly.
for time, detections in feeder:
    if any(hasattr(measurement.measurement_model, 'applied_bias') for measurement in detections):
        hypotheses = bias_data_associator.associate(tracks, detections, time)
        # Update bias estimate using associated measurements
        updates = bias_updater.update([h for h in hypotheses.values() if h])
        for track, update in zip((t for t, h in hypotheses.items() if h), updates):
            track.append(update)
        for track, hyp in {t: h for t, h in hypotheses.items() if not h}.items():
            track.append(hyp.prediction)

        # Adjust measurement models by removing relative bias for plotting later
        rel_bias_vector = bias_track[-2].state_vector - bias_track[-1].state_vector
        for model in {d.measurement_model for d in detections}:
            model.translation_offset -= rel_bias_vector
            model.applied_bias += rel_bias_vector  # No longer used, but for completeness
    else:
        # Standard track update if no bias applied i.e. unbiased sensors
        hypotheses = data_associator.associate(tracks, detections, time)
        for track in tracks:
            hypothesis = hypotheses[track]
            if hypothesis.measurement:
                post = updater.update(hypothesis)
                track.append(post)
            else:
                track.append(hypothesis.prediction)

# %%
# Visualise Tracking Results
# --------------------------
#
# We plot the estimated tracks alongside the ground truths and measurements, showing the effect
# of bias estimation.
#
# By comparing the green biased detection to the previous plot (with ground truth layer also to
# make comparison clearer), it can be seen that the bias has been corrected.
plotter = Plotterly()
plotter.plot_ground_truths(ground_truths, mapping=[0, 2], line_dash="solid", label="Platforms")
plotter.plot_ground_truths(targets, mapping=[0, 2])
for n, sensor_measurements in enumerate(measurements):
    kwargs = {}
    if n == 0:
        kwargs['marker'] = {'color': 'green'}
    plotter.plot_measurements(
        {m for ms in sensor_measurements for m in ms[1]},
        mapping=[0, 2],
        label=f'Sensor {n}',
        **kwargs)
plotter.plot_tracks(tracks, [0, 2])
plotter.fig

# %%
# Visualise Bias Estimation
# -------------------------
#
# Finally, we plot the true bias and the estimated bias over time, for both x and y components,
# including 1 standard deviation error area.

# sphinx_gallery_thumbnail_number = 3

plotter = Plotterly(dimension=1, axis_labels=['Bias', 'Time'])
plotter.plot_ground_truths(true_bias, mapping=[0], label="True 𝑥 bias")
plotter.plot_tracks(bias_track, mapping=[0], uncertainty=True, label="𝑥 bias estimate")
plotter.plot_ground_truths(true_bias, mapping=[1], label="True 𝑦 bias")
plotter.plot_tracks(bias_track, mapping=[1], uncertainty=True, label="𝑦 bias estimate")
plotter.fig
