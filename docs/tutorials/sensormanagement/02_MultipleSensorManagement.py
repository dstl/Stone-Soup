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
# First generate transition model and ground truths as in Tutorial 1.
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
# increase in combinations to consider in the uncertainty based sensor manager. This is discussed further later.

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
for j in range(0, ntruths):
    tracksA.append(Track(id=f"id{j}"))
    tracksA[j].append(priors[j])

# Initialise tracks from the UncertaintyManager
tracksB = []
for j in range(0, ntruths):
    tracksB.append(Track(id=f"id{j}"))
    tracksB[j].append(priors[j])

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
# The first method :class:`RandomManager` is exactly yhe same as in Tutorial 1. The sensor manager chooses
# a target to observe randomly. To do this the :meth:`choose_actions` function uses :meth:`np.random.uniform()`
# to draw random samples from a uniform distribution between 0 and 1, and multiply by the number of targets
# in the simulation. This means a target to observe is selected randomly each time the :meth:`choose_actions`
# function is run.


class RandomManager(Base):
    action_list: list = Property(doc="List of possible actions")

    def choose_actions(self):
        return self.action_list[int(np.floor(len(self.action_list) * np.random.uniform()))]


# %%
# UncertaintyManager class
# """"""""""""""""""""""""
#
# The second method :class:`UncertaintyManager` chooses the target observation configuration which results
# in the largest difference between the uncertainty covariances of the target predictions and posteriors
# assuming a predicted measurement corresponding to that prediction. This means the sensor manager chooses
# to observe the targets such that the uncertainty will be reduced the most by making that observation.
#
# For each target in an observation configuration the :meth:`calculate_reward function` calculates the
# difference between the covariance matrix norms of the prediction and the posterior assuming a predicted
# measurement corresponding to that prediction. The sum of these differences are returned as a metric for
# that observation configuration.
#
# The :meth:`reward_list` function generates a list of this metric for all possible target observation configurations,
# given the number of sensors.
#
# The :meth:`choose_actions` function takes in the variables `action_list_metric` generated by the
# :meth:`reward_list` function and chooses the configuration with the greatest value. This identifies the
# choice of targets which results in the greatest reduction in uncertainty and returns the target numbers to be
# observed.

import itertools as it
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange


class UncertaintyManager(Base):
    action_list: list = Property(doc="List of possible actions")
    predictor: KalmanPredictor = Property()
    updater: ExtendedKalmanUpdater = Property()
    sensor_set: set = Property(doc="Set of sensors in use")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.possible_configs = list(it.product(self.action_list,
                                                repeat=len(self.sensor_set)))
        self.measurement_models = dict()
        for sensor in self.sensor_set:
            measurement_model = CartesianToBearingRange(
                ndim_state=4,
                mapping=(0, 2),
                noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                                      [0, 0.75 ** 2]]),
                translation_offset=sensor.position)
            self.measurement_models[sensor] = measurement_model
        # Generates measurement models for each sensor
        # - for use before any measurements have been made

    def choose_actions(self, tracks_list, metric_time):
        action_list_metric = self.reward_list(tracks_list, metric_time)
        config = self.possible_configs[np.argmax(action_list_metric)]

        target_choices = []
        for choice in config:
            target_choices.append(self.action_list[choice])
        return target_choices

    def reward_list(self, tracks_list, metric_time):
        metric_list = []
        for config in self.possible_configs:
            metric = self.calculate_reward(tracks_list, config, metric_time)
            metric_list.append(metric)
        return metric_list

    def calculate_reward(self, tracks_list, config, metric_time):
        self.sensor_list = list(sensor_set)
        config_metric = 0

        # For each observation in the configuration
        for obs in set(config):
            sensor = self.sensor_list[config.index(obs)]
            self.updater.measurement_model = self.measurement_models[sensor]

            # Do the prediction and store covariance matrix
            prediction = self.predictor.predict(tracks_list[obs][-1],
                                                timestamp=metric_time)
            pred_cov_norm = np.linalg.norm(prediction.covar)

            # Calculate predicted measurement
            predicted_measurement = self.updater.predict_measurement(prediction)

            # Generate detection from predicted measurement
            detection = Detection(predicted_measurement.state_vector,
                                  timestamp=prediction.timestamp)

            # Generate hypothesis based on prediction and detection
            hypothesis = SingleHypothesis(prediction, detection)

            # Do the update based on this hypothesis and store covariance matrix
            update = self.updater.update(hypothesis)
            update_cov_norm = np.linalg.norm(update.covar)

            # Calculate metric
            metric = pred_cov_norm - update_cov_norm
            config_metric = config_metric + metric

        return config_metric


# %%
# Create an instance of the sensor manager
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Create an instance of each sensor manager class. Each class takes in an `action_list`, a list of the possible actions
# to select from. Here this is the possible target numbers the manager can choose to observe. The
# :class:`UncertaintyManager` also requires a predictor, an updater and the sensor set.

actions = range(0, ntruths)

randommanager = RandomManager(actions)
uncertaintymanager = UncertaintyManager(actions, predictor, updater, sensor_set)

# %%
# Run the sensor managers
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# From here the method differs slightly for each sensor manager. :class:`RandomManager` does not require any other input
# variables whereas :class:`UncertaintyManager` requires a tracks list and a time step.
#
# For both sensor management methods, at each time step a prediction is made for each of the targets except the chosen
# target,  which is updated. These states are appended to the tracks list.
#
# The ground truths, tracks and uncertainty ellipses are then plotted.
#
# RandomManager
# """""""""""""
#
# Here the chosen target for observation is selected randomly using the method :meth:`choose_actions()` from the class
# :class:`RandomManager`. This is repeated for each sensor in the set of sensors.

# Generate list of timesteps from ground truth timestamps
timesteps = []
for state in truths[0]:
    timesteps.append(state.timestamp)

for timestep in timesteps[1:]:

    chosen_targets = []
    measurementsA = []

    for sensor in sensor_set:
        # Activate the sensor manager and make a measurement
        chosen_target = randommanager.choose_actions()
        chosen_targets.append(chosen_target)

        # The ground truth will therefore be:
        selected_truth = truths[chosen_target][timestep]

        # Observe this
        measurement = sensor.measure([selected_truth], noise=True)
        detection = list(measurement)
        measurementsA.append(detection[0])

    # Do the prediction (for all targets) and the update for those observed
    for target_id, track in enumerate(tracksA):
        prediction = predictor.predict(track[-1],
                                       timestamp=measurementsA[-1].timestamp)

        if target_id in chosen_targets:  # Update the prediction
            # Association - just a single hypothesis at present
            measurement_index = chosen_targets.index(target_id)
            hypothesis = SingleHypothesis(prediction,
                                          measurementsA[measurement_index])  # Group a prediction and measurement

            # Update and add to track
            post = updater.update(hypothesis)
        else:
            post = prediction

        track.append(post)

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
# First a list is created of all possible sensor configurations. Each possible configuration is a list the
# length of the number of sensors, containing target numbers to be observed by the sensors.
#
# At each time step, for each target in each configuration:
#
# * A prediction is made and the covariance matrix norms stored
# * A predicted measurement is made
# * A synthetic detection is generated from this predicted measurement
# * A hypothesis generated based on the detection and prediction
# * This hypothesis is used to do an update and the covariance matrix norms of the update are stored.
#
# The metric `action_list_metric` is calculated as the sum of the differences between these covariance matrix norms
# for the targets in the possible configuration.
#
# The list of metrics for each possible configuration is passed into :class:`UncertaintyManager` using the
# method :meth:`choose_actions`. The sensor manager is also passed the list of possible configurations and uses
# this information to identify the configuration which results in the largest reduction in uncertainty.
#
# The sensor manager returns a list of targets corresponding to the optimum configuration as the chosen targets
# to observe.
#
# The prediction for each target is appended to the tracks list at each time step, except for the chosen
# targets for which an update is appended.

for timestep in timesteps[1:]:

    chosen_targets = uncertaintymanager.choose_actions(tracksB, timestep)
    measurementsB = []

    for target in chosen_targets:
        # The ground truth will therefore be:
        selected_truth = truths[target][timestep]

        # Observe this
        measurement = sensor.measure([selected_truth], noise=True)
        detection = list(measurement)
        measurementsB.append(detection[0])

    # Do the prediction (for all targets) and the update for those
    for target_id, track in enumerate(tracksB):
        # Do the prediction
        prediction = predictor.predict(track[-1],
                                       timestamp=measurementsB[-1].timestamp)

        if target_id in chosen_targets:  # Update the prediction
            # Association - just a single hypothesis at present
            measurement_index = chosen_targets.index(target_id)
            hypothesis = SingleHypothesis(prediction,
                                          measurementsB[measurement_index])  # Group a prediction and measurement

            # Update and add to track
            post = updater.update(hypothesis)
        else:
            post = prediction

        track.append(post)

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

# %%
# This metric shows that the uncertainty in the tracks generated by the :class:`RandomManager` is much greater
# than for those generated by the :class:`UncertaintyManager`. This is also reflected by the uncertainty ellipses
# in the initial plots of tracks and truths.
#
# The uncertainty for the :class:`UncertaintyManager` starts poor and then improves initially as
# observations are made. This initial uncertainty is because the priors given are not correct. The uncertainty
# then increases slowly over time. This is likely because the targets are moving further away from the
# :class:`~.RadarBearingRange` sensors so the uncertainty in the observations made increases.

# %%
# References
# ----------
#
# .. [#] *D. Schuhmacher, B. Vo and B. Vo*, **A Consistent Metric for Performance Evaluation of
#    Multi-Object Filters**, IEEE Trans. Signal Processing 2008
# .. [#] *Votruba, Paul & Nisley, Rich & Rothrock, Ron and Zombro, Brett.*, **Single Integrated Air
#    Picture (SIAP) Metrics Implementation**, 2001

