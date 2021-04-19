#!/usr/bin/env python
# coding: utf-8

"""
==========================================================
2 - Multiple Sensor Management
==========================================================
"""

# %%
#
# This notebook follows on from Sensor Management Tutorial 1 and further explores how existing
# Stone Soup features can be used to build simple sensor management algorithms for tracking and
# state estimation. This second tutorial demonstrates the limitations of the brute force optimisation
# method introduced in Tutorial 1 by increasing the number of sensors used in the scenario.
#
# Introducing multiple sensors
# ----------------------------
# The example in this tutorial considers two simple sensor management methods and applies them to the
# same set of ground truths in order to observe the difference in tracks. The scenario simulates 12
# targets moving on nearly constant velocity trajectories and an adjustable number of sensors. Each
# sensor can only observe one target each at each time step but a target may be observed by more than one sensor.
#
# The first method, "RandomManager" chooses a target for each sensor to observe randomly with equal probability.
#
# The second method, "UncertaintyManager" aims to reduce the total uncertainty of the track estimates at each
# time step. To achieve this the sensor manager considers all possible combinations of target observations for
# the given number of sensors. The sensor manager chooses the configuration for which the sum of estimated
# uncertainties (as represented by the Frobenius norm of the covariance matrix) can be reduced the most by observing
# the chosen targets.
#
# The introduction of multiple sensors means an increase in the possible combinations of target observations
# that the UncertaintyManager must consider. This brute force optimisation method of looking at every possible
# combination of observations becomes very slow as more sensors are introduced. This demonstrates the
# limitations of using this method with a large number of sensors.
#
# As in the first tutorial the OSPA [#]_, SIAP [#]_ and uncertainty metrics are used to assess the performance of the
# sensor managers.

# %%
# Sensor Management example
# -------------------------
#
# Setup
# ^^^^^
#
# First a simulation must be set up using components from Stone Soup. For this the following imports are required.

import numpy as np
from datetime import datetime, timedelta

start_time = datetime.now()

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from stonesoup.base import Property, Base
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.detection import Detection

# %%
# Generate ground truth
# ^^^^^^^^^^^^^^^^^^^^^
#
# Generate transition model and ground truths as in Tutorial 1.
#
# The number of targets in this simulation is defined by `n_truths` - here there are 12 targets. The time the
# simulation is observed for is defined by `time_max`.
#
# We can fix our random number generator in order to probe a particular example repeatedly. This can be undone by
# commenting out the first line in the next cell.

np.random.seed(1991)

# Generate transition model
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                          ConstantVelocity(0.005)])

yps = range(0, 120, 10)  # y value for prior state
truths = []
ntruths = 12  # number of ground truths in simulation
time_max = 100  # timestamps the simulation is observed over

# Generate ground truths
for j in range(0, ntruths):
    truth = GroundTruthPath([GroundTruthState([0, 1, yps[j], 1], timestamp=start_time)],
                            id=f"id{j}")

    for k in range(1, time_max):
        truth.append(
            GroundTruthState(transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
                             timestamp=start_time + timedelta(seconds=k)))
    truths.append(truth)

# %%
# Plot the ground truths. This is done using the :class:`~.Plotter` class from Stone Soup.

from stonesoup.plotter import Plotter

# Stonesoup plotter requires sets not lists
truths_set = set(truths)

plotter = Plotter()
plotter.ax.axis('auto')
plotter.plot_ground_truths(truths_set, [0, 2])

# %%
# Create sensors
# ^^^^^^^^^^^^^^
# Create a set of sensors. This notebook explores the use of the :class:`~.RadarBearingRange` sensor with the
# number of sensors initially set as 3. Each sensor is positioned along the line :math:`x=10`, at distance
# intervals of 10.
#
# Increasing the number of sensors above 3 significantly increases the run time of the sensor manager due to the
# increase in combinations to be considered by the :class:`UncertaintyManager`. This is discussed further later.

total_no_sensors = 3

from stonesoup.sensor.radar.radar import RadarBearingRange

sensor_set = set()

for n in range(0, total_no_sensors):
    sensor = RadarBearingRange(position_mapping=(0, 2),
                               noise_covar=np.array([[np.radians(0.5)**2, 0],
                                                    [0, 0.75**2]]),
                               ndim_state=4,
                               position=np.array([[10], [n*10]])
                               )
    sensor_set.add(sensor)

# %%
# Create the Kalman predictor and updater
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Construct a predictor and updater using the :class:`~.KalmanPredictor` and :class:`~.ExtendedKalmanUpdater`
# components from Stone Soup. The measurement model for the updater is `None` as it is an attribute of the sensor.

from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

from stonesoup.updater.kalman import ExtendedKalmanUpdater
updater = ExtendedKalmanUpdater(measurement_model=None)
# measurement model is added to detections by the sensor

# %%
# Run the Kalman filters
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Create priors which estimate the targets' initial states.

from stonesoup.types.state import GaussianState

priors = []
for j in range(0, ntruths):
    priors.append(GaussianState([[0], [1.5], [yps[j]+5], [1.5]], np.diag([1.5, 0.25, 1.5, 0.25]+np.random.normal(0,5e-4,4)),
                                timestamp=start_time))

# %%
# Initialise the tracks by creating an empty list and appending the priors generated. This needs to be done
# separately for both sensor manager methods as they will generate different sets of tracks.

from stonesoup.types.track import Track

# Initialise tracks from the RandomManager
tracksA = []
for j, prior in enumerate(priors):
    tracksA.append(Track([prior], id=f"id{j}"))

# Initialise tracks from the UncertaintyManager
tracksB = []
for j, prior in enumerate(priors):
    tracksB.append(Track([prior], id=f"id{j}"))

# %%
# Create sensor management classes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Next we create our sensor management classes. As is Tutorial 1 two sensor manager classes are built -
# :class:`RandomManager` and :class:`UncertaintyManager`.
#
# RandomManager class
# """""""""""""""""""
#
# The first method :class:`RandomManager` chooses a target to observe randomly. To do this the
# :meth:`choose_actions` function uses :meth:`random.choice()` to choose a track from `tracks_list` for each
# sensor to observe. It returns the chosen configuration of sensors and tracks to be observed as a mapping.


class RandomManager(Base):

    sensor_set: set = Property(doc="Set of sensors in use")

    def choose_actions(self, tracks_list, metric_time):
        config = dict()
        trackIDs = [track.id for track in tracks_list]

        # For each sensor, randomly select a track to be observed
        for sensor in self.sensor_set:
            trackID = np.random.choice(trackIDs)
            for track in tracks_list:
                if track.id == trackID:
                    config[sensor] = track

        # Return dictionary of sensors and tracks to be observed
        return config

# %%
# UncertaintyManager class
# """"""""""""""""""""""""
#
# The second method :class:`UncertaintyManager` chooses the target observation configuration which results
# in the largest difference between the uncertainty covariances of the target predictions and posteriors
# assuming a predicted measurement corresponding to that prediction. This means the sensor manager chooses
# to observe the targets such that the uncertainty will be reduced the most by making the chosen observations.
#
# For a given configuration of sensors and tracks to observe the :meth:`calculate_reward` function calculates the
# difference between the covariance matrix norms of the prediction and the posterior assuming a predicted
# measurement corresponding to that prediction. The sum of these differences is returned as a metric for
# that observation configuration.
#
# The :meth:`choose_actions` function selects the configuration with the maximum value for the metric generated
# by :meth:`calculate_reward`. This identifies the choice of track observations which results in the greatest
# reduction in uncertainty and returns the configuration of sensors and tracks to be observed as a mapping.


import itertools as it
from functools import partial
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange


class UncertaintyManager(Base):

    sensor_set: set = Property(doc="Set of sensors in use")
    predictor: KalmanPredictor = Property()
    updater: ExtendedKalmanUpdater = Property()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Sensors must have a fixed order within the manager
        self.sensor_list = list(sensor_set)

        # Generate a dictionary measurement models for each sensor (for use before any measurements have been made)
        self.measurement_models = dict()
        for sensor in self.sensor_list:
            measurement_model = CartesianToBearingRange(
                ndim_state=4,
                mapping=(0, 2),
                noise_covar=sensor.noise_covar,
                translation_offset=sensor.position)
            self.measurement_models[sensor] = measurement_model

    def choose_actions(self, tracks_list, metric_time):
        # Generate a tuple of dictionaries where the dictionaries are potential sensor/track observation configurations
        possible_configs = ({sensor: track
                             for sensor, track in zip(self.sensor_list, config)}
                            for config in it.product(tracks_list,
                                                     repeat=len(self.sensor_list)))

        # Select the configuration which gives the maximum reward at the given time
        config = max(possible_configs, key=partial(self.calculate_reward,
                                                   metric_time=metric_time))

        # Return selected configuration as a dictionary of sensors and tracks to be observed
        return config

    def calculate_reward(self, config, metric_time):
        config_metric = 0

        # Create dictionary of predictions for the tracks in the configuration
        predictions_updates = {track: self.predictor.predict(track[-1],
                                                             timestamp=metric_time)
                               for track in config.values()}

        # For each sensor in the configuration
        for sensor, track in config.items():
            # Provide the updater with the correct measurement model for the sensor
            self.updater.measurement_model = self.measurement_models[sensor]

            # If the track is selected by a sensor for the first time 'previous' is the prediction
            # If the track has already been selected by a sensor 'previous' is the most recent update
            previous = predictions_updates[track]
            previous_cov_norm = np.linalg.norm(previous.covar)

            # Calculate predicted measurement
            predicted_measurement = self.updater.predict_measurement(previous)

            # Generate detection from predicted measurement
            detection = Detection(predicted_measurement.state_vector,
                                  timestamp=metric_time)

            # Generate hypothesis based on prediction/previous update and detection
            hypothesis = SingleHypothesis(previous, detection)

            # Do the update based on this hypothesis and store covariance matrix
            update = self.updater.update(hypothesis)
            update_cov_norm = np.linalg.norm(update.covar)

            # Replace prediction in dictionary with update
            predictions_updates[track] = update

            # Calculate metric for the track observation and add to the metric for the configuration
            metric = previous_cov_norm - update_cov_norm
            config_metric = config_metric + metric

        # Return value of configuration metric
        return config_metric

# %%
# Create an instance of the sensor manager
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Create an instance of each sensor manager class. Both sensor managers take in the `sensor_set`.
# The :class:`UncertaintyManager` also requires a predictor and an updater.


randommanager = RandomManager(sensor_set)

uncertaintymanager = UncertaintyManager(sensor_set, predictor, updater)

# %%
# Run the sensor managers
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# Both sensor management methods require a list of tracks and a timestamp at each time step when calling
# the function :meth:`choose_actions`. This returns a mapping of sensors and tracks to be observed by each
# sensor, decided by the sensor managers.
#
# For both sensor management methods, at each time step a prediction is made for each of the targets except the
# chosen targets, which are updated. These states are appended to the tracks list.
#
# The ground truths, tracks and uncertainty ellipses are then plotted.
#
# RandomManager
# """""""""""""
#
# Here the chosen target for observation is selected randomly using the method :meth:`choose_actions()` from the class
# :class:`RandomManager`. This returns a mapping of sensors to tracks where tracks are selected randomly.

from collections import defaultdict

# Generate list of timesteps from ground truth timestamps
timesteps = []
for state in truths[0]:
    timesteps.append(state.timestamp)

for timestep in timesteps[1:]:

    # Generate chosen configuration
    chosen_action = randommanager.choose_actions(tracksA, timestep)

    # Create empty dictionary for lists of measurements
    measurementsA = defaultdict(list)

    for sensor, track in chosen_action.items():

        # The selected ground truth will be:
        for truth in truths:
            if truth.id == track.id:
                selected_truth = truth[timestep]
                break
        else:
            raise ValueError()

        # Observe this ground truth
        measurement = sensor.measure([selected_truth], noise=True)
        measurementsA[track].append(measurement.pop())

    # Do the prediction (for all targets) and the update for those observed
    for track in tracksA:
        # Do the prediction
        new_state = predictor.predict(track[-1],
                                      timestamp=timestep)

        for measurement in measurementsA[track]:  # Update the prediction
            # Association - just a single hypothesis at present
            hypothesis = SingleHypothesis(new_state,
                                          measurement)  # Group a prediction and measurement

            # Update and add to track
            new_state = updater.update(hypothesis)

        track.append(new_state)

# %%
# Plot ground truths, tracks and uncertainty ellipses for each target. The positions of the sensors are indicated
# by black x markers.

from matplotlib.lines import Line2D

plotterA = Plotter()
plotterA.ax.axis('auto')

# Plot sensor positions as black x markers
for sensor in sensor_set:
    plotterA.ax.scatter(sensor.position[0], sensor.position[1], marker='x', c='black')
plotterA.labels_list.append('Sensor')
plotterA.handles_list.append(Line2D([], [], linestyle='', marker='x', c='black'))

plotterA.plot_ground_truths(truths_set, [0, 2])
plotterA.plot_tracks(set(tracksA), [0, 2], uncertainty=True)

# %%
# In comparison to Tutorial 1 the performance of the :class:`RandomManager` has improved greatly. This is
# because a greater number of sensors means each target is more likely to be observed. This means the uncertainty
# of the track does not increase as much because the targets are observed more often.

# %%
# UncertaintyManager
# """"""""""""""""""
#
# Here the chosen target for observation is selected based on the difference between the covariance matrices of the
# prediction and the update of predicted measurement.
#
# First a list is created of all possible sensor/track configurations. Each possible configuration is a mapping
# of sensors to tracks.
#
# At each time step, for each target in each configuration:
#
# * A prediction is made and the covariance matrix norms stored
# * A predicted measurement is made
# * A synthetic detection is generated from this predicted measurement
# * A hypothesis generated based on the detection and prediction
# * This hypothesis is used to do an update and the covariance matrix norms of the update are stored.
#
# The metric `config_metric` is calculated as the sum of the differences between these covariance matrix norms
# for the tracks in the possible configuration.
#
# The sensor manager identifies the configuration which results in the largest value of this metric and therefore
# largest reduction in uncertainty. It returns the optimum sensor/track configuration as a dictionary.
#
# The prediction for each track is appended to the tracks list at each time step, except for the observed
# tracks for which an update is appended using the selected sensor(s).

for timestep in timesteps[1:]:

    # Generate chosen configuration
    chosen_action = uncertaintymanager.choose_actions(tracksB, timestep)

    # Create empty dictionary for lists of measurements
    measurementsB = defaultdict(list)

    for sensor, track in chosen_action.items():

        # The selected ground truth will be:
        for truth in truths:
            if truth.id == track.id:
                selected_truth = truth[timestep]
                break
        else:
            raise ValueError()

        # Observe this ground truth
        measurement = sensor.measure([selected_truth], noise=True)
        measurementsB[track].append(measurement.pop())

    # Do the prediction (for all targets) and the update for those observed
    for track in tracksB:
        # Do the prediction
        new_state = predictor.predict(track[-1],
                                      timestamp=timestep)

        for measurement in measurementsB[track]:  # Update the prediction
            # Association - just a single hypothesis at present
            hypothesis = SingleHypothesis(new_state,
                                          measurement)  # Group a prediction and measurement

            # Update and add to track
            new_state = updater.update(hypothesis)

        track.append(new_state)

# %%
# Plot ground truths, tracks and uncertainty ellipses for each target.

plotterB = Plotter()
plotterB.ax.axis('auto')

# Plot sensor positions as black x markers
for sensor in sensor_set:
    plotterB.ax.scatter(sensor.position[0], sensor.position[1], marker='x', c='black')
# Add to legend generated
plotterB.labels_list.append('Sensor')
plotterB.handles_list.append(Line2D([], [], linestyle='', marker='x', c='black'))

plotterB.plot_ground_truths(truths_set, [0, 2])
plotterB.plot_tracks(set(tracksB), [0, 2], uncertainty=True)

# %%
# The smaller uncertainty ellipses in this plot suggest that the :class:`UncertaintyManager` provides a much
# better track than the :class:`RandomManager`. As with the :class:`RandomManager`, performance has improved from
# Tutorial 1 due to the additional sensors.

# %%
# Combinatorics
# ^^^^^^^^^^^^^
#
# The following graph demonstrates how the number of possible configurations increases with the number
# of sensors and number of targets. The number of configurations which are considered by the sensor manager
# for :math:`M` targets and :math:`N` sensors is :math:`M^N`.
#
# In this example there are 12 targets so the number of possible configurations should be :math:`12^N`
# where :math:`N` is the number of sensors. This exponential increase means that as larger number of
# sensors slows down the run time of the sensor manager significantly because there are so many more iterations
# to consider.
#
# Changing the number of sensors to :math:`N\geq 4` leads to a much longer run time.
# This highlights a practical limitation of using this brute force optimisation method for multiple
# sensors.

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

nsensors = np.arange(1, 100.0)
ntargets = np.arange(1, 100.0)
nsensors, ntargets = np.meshgrid(nsensors, ntargets)
ncombinations = ntargets**nsensors

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(nsensors, ntargets, np.log10(ncombinations), cmap='coolwarm')
ax.set_xlabel("No. sensors")
ax.set_ylabel("No. targets")
ax.set_zlabel("log of no. combinations")

# %%
# Metrics
# -------
#
# Metrics can be used to compare how well different sensor management techniques are working.
# As in Tutorial 1 the metrics used here are the OSPA, SIAP and uncertainty metrics.

from stonesoup.metricgenerator.ospametric import OSPAMetric
ospa_generator = OSPAMetric(c=40, p=1)

from stonesoup.metricgenerator.tracktotruthmetrics import SIAPMetrics
siap_generator = SIAPMetrics(position_mapping=[0, 2], velocity_mapping=[1, 3])

from stonesoup.dataassociator.tracktotrack import TrackIDbased
associator=TrackIDbased()

from stonesoup.metricgenerator.uncertaintymetric import SumofCovarianceNormsMetric
uncertainty_generator = SumofCovarianceNormsMetric()

# %%
# Generate a metrics manager for each sensor management method.

from stonesoup.metricgenerator.manager import SimpleManager

metric_managerA = SimpleManager([ospa_generator, siap_generator, uncertainty_generator],
                                associator=associator)

metric_managerB = SimpleManager([ospa_generator, siap_generator, uncertainty_generator],
                                associator=associator)

# %%
# For each time step, data is added to the metric manager on truths and tracks.
# The metrics themselves can then be generated from the metric manager.

metric_managerA.add_data(truths, tracksA)
metric_managerB.add_data(truths, tracksB)

metricsA = metric_managerA.generate_metrics()
metricsB = metric_managerB.generate_metrics()

# %%
# OSPA metric
# ^^^^^^^^^^^
#
# First we look at the OSPA metric. This is plotted over time for each sensor manager method.

ospa_metricA = {metric for metric in metricsA if metric.title == "OSPA distances"}.pop()
ospa_metricB = {metric for metric in metricsB if metric.title == "OSPA distances"}.pop()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot([i.timestamp for i in ospa_metricA.value],
        [i.value for i in ospa_metricA.value],
        label='RandomManager')
ax.plot([i.timestamp for i in ospa_metricB.value],
        [i.value for i in ospa_metricB.value],
        label='UncertaintyManager')
ax.set_ylabel("OSPA distance")
ax.set_xlabel("Time")
ax.legend()

# %%
# OSPA distance starts large due to the position offset in the priors and then improves for both scenarios as
# observations are made. The :class:`UncertaintyManager` generally results in a smaller OSPA distance than the random
# observations of the :class:`RandomManager`.
#
# A larger number of sensors results in improved performance for the :class:`RandomManager`, in comparison to
# Tutorial 1 where there is only one sensor. This is due to the increased likelihood of each target being
# observed randomly.
#
# SIAP metrics
# ^^^^^^^^^^^^
#
# Next we look at SIAP metrics. We are only interested in the positional accuracy (PA) and
# velocity accuracy (VA). These metrics can be plotted to show how they change over time.

fig, axes = plt.subplots(2)

for metric in metricsA:
    if metric.title.startswith('time-based SIAP PA'):
        pa_metricA = metric
    elif metric.title.startswith('time-based SIAP VA'):
        va_metricA = metric
    else:
        pass

for metric in metricsB:
    if metric.title.startswith('time-based SIAP PA'):
        pa_metricB = metric
    elif metric.title.startswith('time-based SIAP VA'):
        va_metricB = metric
    else:
        pass

times = metric_managerB.list_timestamps()

axes[0].set(title='Positional Accuracy', xlabel='Time', ylabel='PA')
axes[0].plot(times, [metric.value for metric in pa_metricA.value],
             label='RandomManager')
axes[0].plot(times, [metric.value for metric in pa_metricB.value],
             label='UncertaintyManager')
axes[0].legend()

axes[1].set(title='Velocity Accuracy', xlabel='Time', ylabel='VA')
axes[1].plot(times, [metric.value for metric in va_metricA.value],
             label='RandomManager')
axes[1].plot(times, [metric.value for metric in va_metricB.value],
             label='UncertaintyManager')
axes[1].legend()

# %%
# Similar to the OSPA distances, positional accuracy starts as quite poor for both scenarios due to the offset in the
# priors, and then improves over time as observations are made. Again the :class:`UncertaintyManager`
# generally results in a better positional accuracy than the random observations of the :class:`RandomManager`.
#
# Velocity accuracy also starts quite poor due to an error in the priors. It improves over time as more observations
# are made, then remains relatively similar for each sensor manager. This is because the velocity remains nearly
# constant throughout the simulation. This is not likely to be the case in a real-world scenario.
#
# As with the OSPA metric the larger number of sensors generally results in improved performance
# for the :class:`RandomManager` in comparison to Tutorial 1. This is due to the increased likelihood of
# each target being observed randomly.
#
# Uncertainty metric
# ^^^^^^^^^^^^^^^^^^
#
# Finally we look at the uncertainty metric which computes the sum of covariance matrix norms of each state at each
# time step. This is plotted over time for each sensor manager method.

uncertainty_metricA = {metric for metric in metricsA if metric.title == "Sum of Covariance Norms Metric"}.pop()
uncertainty_metricB = {metric for metric in metricsB if metric.title == "Sum of Covariance Norms Metric"}.pop()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot([i.timestamp for i in uncertainty_metricA.value],
        [i.value for i in uncertainty_metricA.value],
        label='RandomManager')
ax.plot([i.timestamp for i in uncertainty_metricB.value],
        [i.value for i in uncertainty_metricB.value],
        label='UncertaintyManager')
ax.set_ylabel("Sum of covariance matrix norms")
ax.set_xlabel("Time")
ax.legend()

# sphinx_gallery_thumbnail_number = 7

# %%
# This metric shows that the uncertainty in the tracks generated by the :class:`RandomManager` is much greater
# than for those generated by the :class:`UncertaintyManager`. This is also reflected by the uncertainty ellipses
# in the initial plots of tracks and truths.
#
# The uncertainty for the :class:`UncertaintyManager` starts poor and then improves initially as
# observations are made. This initial uncertainty is because the priors given are not correct. The uncertainty
# then increases slowly over time. This is likely because the targets are moving further away from the
# :class:`~.RadarBearingRange` sensors so the uncertainty in the observations made increases.
#
# There appears to be a periodicity in the uncertainty metric for the :class:`UncertaintyManager`. This is due to
# the fact that this simulation is quite clean and the uncertainty of each track increases by the same amount if
# left unobserved. Since the sensor manager is then making observations based on this uncertainty, it rotates
# through observing each of the targets in the same order creating this pattern.

# %%
# References
# ----------
#
# .. [#] *D. Schuhmacher, B. Vo and B. Vo*, **A Consistent Metric for Performance Evaluation of
#    Multi-Object Filters**, IEEE Trans. Signal Processing 2008
# .. [#] *Votruba, Paul & Nisley, Rich & Rothrock, Ron and Zombro, Brett.*, **Single Integrated Air
#    Picture (SIAP) Metrics Implementation**, 2001

