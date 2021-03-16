#!/usr/bin/env python
# coding: utf-8

"""
==========================================================
1 - Single Sensor Management
==========================================================
"""

# %%
# 
# This tutorial introduces how Stone Soup classes can be used to build simple sensor management
# algorithms for tracking and state estimation. The intention is to further develop the methods explored
# here in order to build sensor management classes that can be added to the Stone Soup framework.
# This tutorial exposes some of the logic which would otherwise be hidden by a sensor manager class.
# 
# Background
# ----------
# 
# Sensor management is the process of deciding and executing the actions that a sensor or group of sensors
# will take in a specific scenario and with a particular objective, or objectives in mind. The process
# involves using information about the scenario to determine an appropriate action for the sensing system
# to take. An observation of the state of the system is then made using the sensing configuration decided
# by the sensor manager. The observations are used to update the estimate of the collective states and this
# update is used (if necessary) to determine the next action for the sensing system to take.
# 
# A simple example can be imagined using a sensor with a limited field of view which must decide which direction
# it should point at each time step. Alternatively, we might construct an objective based example by imagining
# that the desired target is fast moving and the sensor can only observe one target at a time. If there are
# multiple targets which could be observed the sensor manager could choose to observe the target that had the
# greatest estimated velocity at the current time.
# 
# The example in this notebook considers two simple sensor management methods and applies them to the same
# ground truths in order to quantify the difference in behaviour. The scenario simulates 10 targets moving on
# nearly constant velocity trajectories and a sensor that can only observe one target at each time step.
# 
# The first method, "RandomManager" chooses a target randomly with equal probability. The second method,
# "UncertaintyManager" aims to reduce the total uncertainty of the track estimates at each time step. To achieve
# this the sensor manager chooses to look at the target for which the estimated uncertainty - as represented by
# the Frobenius norm of the covariance matrix, can be reduced the most by making an observation.
#
# Sensor management as a POMDP
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Sensor management problems can be considered as Partially Observable Markov Decision Processes (POMDPs) where
# observations provide information about the current state of the system but there is uncertainty in the estimate
# of the underlying state due to noisy sensors and imprecise models of target evaluation.
# 
# POMDPs consist of:
#  * :math:`X_k`, the finite set of possible states for each stage index :math:`k`.
#  * :math:`A_k`, the finite set of possible actions for each stage index :math:`k`.
#  * :math:`R_k(x, a)`, the reward function.
#  * :math:`Z_k`, the finite set of possible observations for each stage index :math:`k`.
#  * :math:`f_k(x_{k}|x_{k-1})`, a (set of) state transition function(s). (Note that actions are excluded from
#    the function at the moment. It may be necessary to include them if prior sensor actions cause the targets to
#    modify their behaviour.)
#  * :math:`h_k(z_k | x_k, a_k)`, a (set of) observation function(s).
#  * :math:`\{x\}_k`, the set of states at :math:`k` to be estimated.
#  * :math:`\{a\}_k`, a set of actions at :math:`k` to be chosen.
#  * :math:`\{z\}_k`, the observations at :math:`k` returned by the sensor.
#  * :math:`\Psi_{k-1}`, denotes the complete set of 'intelligence' available to the sensor manager before deciding
#    on an action at :math:`k`. This includes the prior set of state estimates :math:`\{x\}_{k-1}`, but may also
#    encompass contextual information, sensor constraints or mission parameters.
#
# Figure 1: Illustration of sequential actions and measurements. [#]_
#
# .. image:: ../../_static/sensor_management_flow_diagram.png
#   :width: 800
#   :alt: Illustration of sequential actions and measurements
#
# :math:`\Psi_k` is the intelligence available to the sensor manager at stage index :math:`k`, to help
# select the action :math:`a_k` for the system to take. An observation :math:`z_k` is made by the sensing system,
# giving information on the state :math:`x_k`. The action :math:`a_k` and observation :math:`z_k` are added to the
# intelligence set to generate :math:`\Psi_{k+1}`, the intelligence available at stage index :math:`k+1`.
#
# Comparing sensor management methods using metrics
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The performance of the two sensor management methods explored in this notebook can be assessed using metrics
# available from the Stone Soup framework. The metrics used to assess the performance of the different methods
# are the OPSA metric [#]_, SIAP metrics [#]_ and an uncertainty metric. Demonstration of the OSPA and SIAP metrics
# can be found in the Metrics Example.
# 
# The uncertainty metric computes the covariance matrices of all target states at each time step and calculates the
# sum of their norms. This gives a measure of the total uncertainty across all tracks at each time step.

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
# Following the methods from previous Stone Soup tutorials we generate a series of combined linear Gaussian transition
# models and generate ground truths. Each ground truth is offset in the y-direction by 10.
# 
# Ground truths are assigned an ID. This is later used by the data associator.
# 
# The number of targets in this simulation is defined by `n_truths` - here there are 10 targets. The time the
# simulation is observed for is defined by `time_max`.
# 
# We can fix our random number generator in order to probe a particular example repeatedly. This can be undone by
# commenting out the first line in the next cell.

np.random.seed(1991)

# Generate transition model
# i.e. fk(xk|xk-1)
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                          ConstantVelocity(0.005)])

yps = range(0, 100, 10)  # y value for prior state
truths = []
ntruths = 10  # number of ground truths in simulation
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
# Create a measurement model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Assign a measurement model. This notebook explores the use of either the :class:`~.LinearGaussian` measurement model
# or :class:`~.CartesianToBearingRange` measurement model. This is changed by setting `nonlinear_example`.

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange

nonlinear_example = False  # Change to True for nonlinear demonstration

# Generate measurement model
# i.e. hk(zk|xk, ak)
if nonlinear_example:
    measurement_model = CartesianToBearingRange(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                              [0, 0.75 ** 2]]),
        translation_offset=np.array([[10], [50]])  # Moves sensor location from default (origin)
    )

else:
    measurement_model = LinearGaussian(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.array([[0.75, 0],
                              [0, 0.75]])
    )

# %%
# Create the Kalman predictor and updater
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Construct a predictor and updater using the :class:`~.KalmanPredictor` and :class:`~.ExtendedKalmanUpdater`
# components from Stone Soup. The :class:`~.ExtendedKalmanUpdater` is used because it can be used for both linear
# and nonlinear measurement models.

from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

from stonesoup.updater.kalman import ExtendedKalmanUpdater
updater = ExtendedKalmanUpdater(measurement_model)

# %%
# Run the Kalman filters
# ^^^^^^^^^^^^^^^^^^^^^^
#
# First create `ntruths` priors which estimate the targets’ initial states, one for each target. In this example
# each prior is offset by 5 in the y direction meaning the position of the track is initially not very accurate. The
# velocity is also systematically offset by +0.5 in both the x and y directions.

from stonesoup.types.state import GaussianState

priors = []
for j in range(0, ntruths):
    priors.append(
        GaussianState([[0], [1.5], [yps[j] + 5], [1.5]], np.diag([1.5, 0.25, 1.5, 0.25] + np.random.normal(0, 5e-4, 4)),
                      timestamp=start_time))

# %%
# Initialise the tracks by creating an empty list and appending the priors generated. This needs to be done separately
# for both sensor manager methods as they will generate different sets of tracks.
#
# (NB: Tracks are also assigned an ID, used later for data association)

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
# Next we create our sensor manager classes. Two sensor manager classes are built - :class:`RandomManager` and
# :class:`UncertaintyManager`.
# 
# RandomManager
# """""""""""""
# 
# The first method :class:`RandomManager`, chooses a target to observe randomly. To do this the :meth:`choose_actions`
# function uses :meth:`np.random.uniform()` to draw random samples from a uniform distribution between 0 and 1, and
# multiply by the number of targets in the simulation. This means a single target to observe is selected randomly at
# each time step.


class RandomManager(Base):
    action_list = Property(list, doc="List of possible actions")

    def choose_actions(self):
        return self.action_list[int(np.floor(len(self.action_list) * np.random.uniform()))]

# %%
# UncertaintyManager
# """"""""""""""""""
# 
# The second method :class:`UncertaintyManager` selects a chosen target to observe based on the difference between the
# covariance matrices of the prediction an the update of predicted measurement. This means the sensor manager chooses
# to observe the target for which the total uncertainty will be reduced the most by making that observation.
# 
# The :meth:`calculate_reward` function calculates the difference between the covariance matrix norms of the
# prediction and the posterior assuming a predicted measurement corresponding to that prediction. The
# :meth:`reward_list` function then generates a list of this metric for every track.
# 
# The :meth:`choose_actions` function takes in the variable `action_list_metric` generated by the :meth:`reward_list`
# function and chooses the target with the largest value of this metric to observe.
# 
# The number of the target which is to be observed, :math:`N` is found using the following equation:
# 
# .. math::
#           N = \underset{n}{\operatorname{argmax}}(m_n)
#
# 
# where :math:`n \in \lbrace{1, 2, ..., \eta}\rbrace` and :math:`\eta` is the number of targets. The metric,
# :math:`m_n` is calculated for each track using the following equation.
# 
# .. math::
#           m_n = \begin{Vmatrix}P_{k|k-1}\end{Vmatrix}-\begin{Vmatrix}P_{k|k}\end{Vmatrix}
#
# 
# where :math:`P_{k|k-1}` and :math:`P_{k|k}` are the covariance matrices for the prediction and update of the track
# respectively. Note that :math:`\begin{Vmatrix}P_{k|k-1}\end{Vmatrix}` and :math:`\begin{Vmatrix}P_{k|k}\end{Vmatrix}`
# represent the Frobenius norms of these covariance matrices.


class UncertaintyManager(Base):
    action_list = Property(list, doc="List of possible actions")
    predictor = Property(KalmanPredictor)
    updater = Property(ExtendedKalmanUpdater)

    def choose_actions(self, tracks_list, metric_time):
        action_list_metric = self.reward_list(tracks_list, metric_time)
        return self.action_list[np.argmax(action_list_metric)]

    def reward_list(self, tracks_list, metric_time):
        metric_list = []
        for track in tracks_list:
            metric = self.calculate_reward(track, metric_time)
            metric_list.append(metric)
        return metric_list

    def calculate_reward(self, track, metric_time):
        # i.e. Rk(x, a)

        # Do the prediction and store covariance matrix
        prediction = self.predictor.predict(track[-1],
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
        return metric

# %%
# Create an instance of the sensor manager
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Create an instance of each sensor manager class. Each class takes in a `action_list`, a list of the possible actions
# to select from. Here this is the possible target numbers the manager can choose to observe. The
# :class:`UncertaintyManager` also requires a predictor and an updater.

actions = range(0, ntruths)

randommanager = RandomManager(actions)
uncertaintymanager = UncertaintyManager(actions, predictor, updater)

# %%
# Run the sensor managers
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# From here the method differs slightly for each sensor manager. :class:`RandomManager` does not require any other input
# variables whereas :class:`UncertaintyManager` requires a tracks list at each time step.
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
# :class:`RandomManager`.

# Generate list of time steps from ground truth timestamps
timesteps = []
for state in truths[0]:
    timesteps.append(state.timestamp)

predictionsA = []
measurementsA = []

for timestep in timesteps[1:]:

    # Activate the sensor manager and make a measurement
    chosen_target = randommanager.choose_actions()

    # The ground truth will therefore be:
    selected_truth = truths[chosen_target][timestep]

    # Observe this
    measurement = measurement_model.function(selected_truth, noise=True)
    measurementsA.append(Detection(measurement,
                                   timestamp=selected_truth.timestamp))

    # Do the prediction (for all targets) and the update for those
    for target_id, track in enumerate(tracksA):
        prediction = predictor.predict(track[-1],
                                       timestamp=measurementsA[-1].timestamp)

        if target_id == chosen_target:  # Update the prediction       
            # Association - just a single hypothesis at present
            hypothesis = SingleHypothesis(prediction, measurementsA[-1])  # Group a prediction and measurement

            # Update and add to track
            post = updater.update(hypothesis)
        else:
            post = prediction

        track.append(post)

# %%
# Plot ground truths, tracks and uncertainty ellipses for each target. 

plotterA = Plotter()
plotterA.ax.axis('auto')
plotterA.plot_ground_truths(truths_set, [0, 2])
plotterA.plot_tracks(set(tracksA), [0, 2], uncertainty=True)

# %%
# UncertaintyManager
# """"""""""""""""""
#
# Here the chosen target for observation is selected based on the difference between the covariance matrices of the
# prediction and posterior, based upon the observation of that target.
# 
# The :meth:`choose_actions` function from the :class:`UncertaintyManager` is called at each time step. This means
# that at each time step, for each target:
# 
#  * A prediction is made and the covariance matrix norms stored
#  * A predicted measurement is made
#  * A synthetic detection is generated from this predicted measurement
#  * A hypothesis generated based on the detection and prediction
#  * This hypothesis is used to do an update and the covariance matrix norms of the update are stored
#  * The difference between these covariance matrix norms is calculated
# 
# The sensor manager then returns the target with the largest value of this metric as the chosen target to observe.
# 
# The prediction for each target is appended to the tracks list at each time step, except for the chosen target for
# which an update is appended.

predictionsB = []
measurementsB = []

for timestep in timesteps[1:]:

    # Activate the sensor manager and choose a target to observe
    # i.e. {a}k 
    chosen_target = uncertaintymanager.choose_actions(tracksB, timestep)

    # The ground truth will therefore be:
    selected_truth = truths[chosen_target][timestep]

    # Observe this
    # i.e. {z}k
    measurement = measurement_model.function(selected_truth, noise=True)
    measurementsB.append(Detection(measurement,
                                   timestamp=selected_truth.timestamp))

    # Do the prediction (for all targets) and the update for those
    for target_id, track in enumerate(tracksB):
        # Do the prediction
        # i.e. {x}k
        prediction = predictor.predict(track[-1], timestamp=measurementsB[-1].timestamp)

        if target_id == chosen_target:  # Update the prediction       
            # Association - just a single hypothesis at present
            hypothesis = SingleHypothesis(prediction, measurementsB[-1])  # Group a prediction and measurement

            # Update and add to track
            post = updater.update(hypothesis)
        else:
            post = prediction

        track.append(post)

# %%
# Plot ground truths, tracks and uncertainty ellipses for each target.

plotterB = Plotter()
plotterB.ax.axis('auto')
plotterB.plot_ground_truths(truths_set, [0, 2])
plotterB.plot_tracks(set(tracksB), [0, 2], uncertainty=True)

# %%
# The smaller uncertainty ellipses in this plot suggest that the :class:`UncertaintyManager` provides a much
# better track than the :class:`RandomManager`.
#
# Metrics
# -------
# 
# Metrics can be used to compare how well different sensor management techniques are working. 
#
# Generating Metrics
# ^^^^^^^^^^^^^^^^^^
#
# The Optimal SubPattern Assignment (OSPA) metric generator computes the OSPA distance between ground truths and tracks. 

from stonesoup.metricgenerator.ospametric import OSPAMetric

ospa_generator = OSPAMetric(c=40,  # 'Max distance for possible association'
                            p=1)  # 'norm associated t distance'

# %%
# Single Integrated Air Picture (SIAP) metrics are made up of multiple individual metrics. In this example we are only
# interested in the kinematic metrics of positional accuracy and velocity accuracy.

from stonesoup.metricgenerator.tracktotruthmetrics import SIAPMetrics

siap_generator = SIAPMetrics(position_mapping=[0, 2], velocity_mapping=[1, 3])

# %%
# The SIAP metrics require an associator to associate tracks to ground truths. This is done using the
# :class:`~.TrackIDbased` associator. This associator uses the track ID to associate each track to the ground truth
# with the same ID. The associator is initiated and later used in the metric manager.

from stonesoup.dataassociator.tracktotrack import TrackIDbased

associator = TrackIDbased()

# %%
# The OSPA and SIAP don't take the uncertainty of the track into account. The initial plots of the tracks and ground
# truths show by plotting the uncertainty ellipses that there is generally less uncertainty in the tracks generated
# by the :class:`UncertaintyManager`.
# 
# To attempt to capture this we can look use an uncertainty metric to look at the sum of covariance matrix norms at
# each time step. This gives a representation of the overall uncertainty of the tracking over time.

from stonesoup.metricgenerator.uncertaintymetric import SumofCovarianceNormsMetric

uncertainty_generator = SumofCovarianceNormsMetric()

# %%
# A metric manager is used for the generation of metrics on multiple :class:`~.GroundTruthPath` and
# :class:`~.Track` objects. This takes in the OSPA generator, SIAP generator and uncertainty generator,
# as well as the associator required for the SIAP metrics.
# 
# We must use a different metric manager for each sensor management method. This is because each sensor manager
# generates different track data which is then used in the metric manager.

from stonesoup.metricgenerator.manager import SimpleManager

metric_managerA = SimpleManager([ospa_generator, siap_generator, uncertainty_generator],
                                associator=associator)

metric_managerB = SimpleManager([ospa_generator, siap_generator, uncertainty_generator],
                                associator=associator)

# %%
# For each time step, data is added to the metric manager on truths and tracks. The metrics themselves can then be
# generated from the metric manager.

metric_managerA.add_data(truths, tracksA)
metric_managerB.add_data(truths, tracksB)

metricsA = metric_managerA.generate_metrics()
metricsB = metric_managerB.generate_metrics()

# %%
# OSPA metric
# ^^^^^^^^^^^
#
# First we look at the OSPA metric. This is plotted over time for each sensor manager method.

import matplotlib.pyplot as plt

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
# SIAP metrics
# ^^^^^^^^^^^^
#
# Next we look at SIAP metrics. This can be done by generating a table which displays all the SIAP metrics computed,
# as seen in the Metrics Example.
# 
# Completeness, ambiguity and spuriousness are not relevant for this example because we are not initiating and
# deleting tracks and we have one track corresponding to each ground truth.

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
# axes[0].tick_params(length=1)
axes[0].plot(times, [metric.value for metric in pa_metricA.value],
             label='RandomManager')
axes[0].plot(times, [metric.value for metric in pa_metricB.value],
             label='UncertaintyManager')
axes[0].legend()

axes[1].set(title='Velocity Accuracy', xlabel='Time', ylabel='VA')
# axes[1].tick_params(length=1)
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
# The uncertainty for the :class:`UncertaintyManager` peaks initially then remains at a constant value. This peak
# is because the priors given are offset but have a small uncertainty meaning uncertainty increases when the first
# observations are made. This simulation is quite clean and the uncertainty of each track increases by the same
# amount if left unobserved. Since the sensor manager is then making observations based on this uncertainty, it is
# reducing it by the same amount each time. This means the total uncertainty in the system is constant.

# %%
# References
# ----------
#
# .. [#] *Hero, A.O., Castanon, D., Cochran, D. and Kastella, K.*, **Foundations and Applications of Sensor
#    Management**. New York: Springer, 2008.
# .. [#] *D. Schuhmacher, B. Vo and B. Vo*, **A Consistent Metric for Performance Evaluation of
#    Multi-Object Filters**, IEEE Trans. Signal Processing 2008
# .. [#] *Votruba, Paul & Nisley, Rich & Rothrock, Ron and Zombro, Brett.*, **Single Integrated Air
#    Picture (SIAP) Metrics Implementation**, 2001
