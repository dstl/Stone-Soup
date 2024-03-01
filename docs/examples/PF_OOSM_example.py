#!/usr/bin/env python
# coding: utf-8

"""
====================================================
Particle filtering with Out-of-sequence Measurements
====================================================
"""

# %%
# Out-of-sequence measurements (OOSM) are a common and bothersome problem in tracking,
# arising from delays in the processing chain and frequent when dealing with
# multi-sensor scenarios.
#
# In literature, there are a number of approaches to deal with this problem (see [#]_ and [#]_)
# and in a series of examples we have presented some algorithms that deal with such measurements.
# These examples were adopted using a Kalman filter approach, here instead we consider the
# case of Particle filters (the algorithm implementation can be found here [#]_).
#
# Particle filters (PF) work in a different manner compared to the Kalman filters because they sequentially
# update the distribution while the latter updates only the filtered distribution.
# In the PF application we have each trajectory, defined as particles, have an associated weight.
# Weights can vary as new information arrives (measurements or clutter).
#
# The algorithm presented follows this logic: a measurement arrives at a time :math:`t_{k}`, if
# :math:`t_{k} > t_{k-1}` we apply the stadard particle filter tracking method.
# If :math:`t_{k} < t_{k-1}`, so it is a delayed measurement, then we look
# in the list of measurements where :math:`t_{k}` belongs. In detail, we look for two indices, :math:`a` and
# :math:`b`, such that :math:`t_{a} > t_{k} > t_{b}`. In this manner, we are able to insert the measurement in each
# particle tracjectory history.
#
# We have obtained a time location for the new measurement, then we sample the particles from the
# state at :math:`t_{b}` with new weights (un-normalised), then we apply the prediction and update steps with the
# delayed measurement :math:`t_{k}`, normalising the particle weights. To finalise the track we
# re-order the existing data such that :math:`t_{n} > t_{m}, \forall (n, m)`.
#
# However, it is important to note that there is the risk of accumulating some disparity over time,
# as we deal with these measurements, this disparity can significantly impact the tracking performances
# causing some degeneracy. To solve this issue, we can use a resampling step, where the
# particles are probabilistically replicated or discarded, resulting in a shift of the
# degeneracy to the end of the track.
#
# In this example we consider a simple single target scenario with negligible level of clutter.
# The scans mimic the data obtained by a sensor, and over time some of these have a delay in their arrival time,
# which we consider as OOSM data. To evaluate the improvement of applying this algorithm,
# we consider an implementation where we ignore the delayed measurements at all.
#
# This example follows this structure:
#   1. Create ground truth and detections;
#   2. Instantiate the tracking components;
#   3. Run the tracker, apply the algorithm and visualise the results.

# %%
# General imports
# ^^^^^^^^^^^^^^^
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import multivariate_normal
from copy import deepcopy

# Simulation parameters
start_time = datetime.now().replace(microsecond=0)
np.random.seed(1908)  # fix the seed
num_steps = 65  # simulation steps
number_particles= 256  # number of particles

# %%
# Stone Soup Imports
# ^^^^^^^^^^^^^^^^^^
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.state import GaussianState, State, StateVector

# instantiate the transition model
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),
                                                          ConstantVelocity(0.05)])

# Create a list of timesteps
timestamps = [start_time]

# create a set of truths
truths = set()

# %%
# 1. Create ground truth and detections;
# --------------------------------------
# In this example we consider a target moving on a nearly constant velocity transition model,
# the detections are obtained by a :class:`~.CartesianToBearingRange` measurement model.
# In this scenario we consider a negligible level of clutter.
#
# The detections are stored in scans, which contain also the time of arrival,
# when a delayed measurement appears, the timestamp of arrival will be modified, but not the
# timestamp attached to the measurement.
# As we collect all the scans, we order them by their arrival time.

# Instantiate the groundtruth for the first target
truth = GroundTruthPath([GroundTruthState([0, 1, -100, 0.3], timestamp=start_time)])

# Generate the ground truth
for k in range(1, num_steps):
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True,
                                  time_interval=timedelta(seconds=5)),
        timestamp=start_time + timedelta(seconds=5*k)))
    timestamps.append(truth.timestamp)

truths.add(truth)

# Create the measurements models
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange

measurement_model = CartesianToBearingRange(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar = np.diag([np.radians(10), 50]),
    translation_offset = np.array([[500], [-500]]))

# Collect the measurements using scans
scans = []

# Create the detections
from stonesoup.types.detection import TrueDetection

# Loop over the timesteps and collect the detections
for k in range(num_steps):
    detections = set()

    # Introduce the delay
    if k%5==0 and k>0:
        delay = 25
    else:
        delay = 0

    measurement = measurement_model.function(truth[k], noise=True)
    detections.add(TrueDetection(state_vector=measurement,
                                 groundtruth_path=truth,
                                 measurement_model=measurement_model,
                                 timestamp=truth[k].timestamp))

    # Scans for tracking and reordering
    scans.append((truth[k].timestamp + timedelta(seconds=delay), detections))

# Reorder the scans by their arrival time
arrival_time_ordered = sorted(scans, key=lambda dscan: dscan[0])

# %%
# 2. Instantiate the tracking components;
# ---------------------------------------
# We load the various tracking components using the particle filter
# :class:`~.ParticleUpdater` and :class:`~.ParticlePredictor`. For the
# resampler we consider :class:`~.ESSResampler`. Then, to initialise the
# tracks we start defining a :class:`~.GaussianState`, we sample
# the particles using a Multivariate Normal distribution around the prior state.
# We assign a weight to each particle, at the beginning they will have the same weight.
# Finally we create a :class:`~.ParticleState` with the new particles and their weights.

# Load the particle filter components
from stonesoup.updater.particle import ParticleUpdater
from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.resampler.particle import ESSResampler

predictor = ParticlePredictor(transition_model)
resampler = ESSResampler()
updater = ParticleUpdater(measurement_model=measurement_model,
                          resampler=resampler)

# Load the Particle state priors
from stonesoup.types.state import GaussianState
from stonesoup.types.state import ParticleState
from stonesoup.types.numeric import Probability
from stonesoup.types.particle import Particle

prior_state = GaussianState(
    StateVector([0, 1, -100, 0.3]),
    np.diag([10, 1, 10, 1]))

samples = multivariate_normal.rvs(
    np.array(prior_state.state_vector).reshape(-1),
    prior_state.covar,
    size=number_particles)

# create the particles
particles = [Particle(sample.reshape(-1, 1),
                      weight=Probability(1.))
             for sample in samples]

particle_prior = ParticleState(state_vector=None,
                               particle_list=particles,
                               timestamp=timestamps[0])

# %%
# 3. Run the tracker, apply the algorithm and visualise the results.
# ------------------------------------------------------------------
# We have now all the components ready. To understand the benefits of
# this algorithm we compute a track where the delayed detections are ignored.
# The algorithm is applied only when the detection timestamp (:math:`t_{k}`) is not
# greater than the previous recorded timestamp (:math:`t_{k-1}`).

# Load the tracking components
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

track = Track(particle_prior)

# Track and prior for the comparison track
track2 = deepcopy(track)
particle_prior2 = deepcopy(particle_prior)

for k in range(1, len(arrival_time_ordered)):  # loop over the scans

    scan = arrival_time_ordered[k][1]
    for detection in scan:  # get the detection time
        arrival_time = detection.timestamp

    if arrival_time > timestamps[k-1]:  # if true, the detections are in order

        # in this scenario we employ a standard particle filter
        for detection in scan:
            prediction = predictor.predict(particle_prior, timestamp=detection.timestamp)
            hypothesis = SingleHypothesis(prediction, detection)
            post = updater.update(hypothesis)
            track.append(post)
            particle_prior = track[-1]

        # track ignoring the OOSm
        for detection in scan:
            prediction = predictor.predict(particle_prior2, timestamp=detection.timestamp)
            hypothesis = SingleHypothesis(prediction, detection)
            post = updater.update(hypothesis)
            track2.append(post)
            particle_prior2 = track2[-1]

    else:  # if not, then the detection is delayed, apply the algorithm

        # find the index where the arrival time belongs
        scan_time_arrival = [arrival_time_ordered[kk][0] for kk in range(0, k-1)]  # get all the times

        # Find the index of the t_b < t_k < t_a < t_{k-1}
        delta_time = [np.abs(entry-arrival_time) for entry in scan_time_arrival]

        # identify the index
        tb_index = delta_time.index(np.min(delta_time))

        # Create the sample state
        sample_state = [track[tb_index].state_vector[ii].mean() for ii in
                        range(len(track[tb_index].state_vector))]

        # draw the samples
        samples = multivariate_normal.rvs(
            np.array(sample_state).reshape(-1),
            np.diag([10, 1, 10, 1]),
            size=number_particles)

        # get the new particles
        particles = [Particle(sample.reshape(-1, 1),
                              weight=Probability(1.))
                     for sample in samples]

        particle_prior = ParticleState(state_vector=None,
                                       particle_list=particles,
                                       timestamp=scan_time_arrival[tb_index])

        # apply the particle filter with the new particle prior and detection
        for detection in scan:
            prediction = predictor.predict(particle_prior, timestamp=detection.timestamp)
            hypothesis = SingleHypothesis(prediction, detection)
            post = updater.update(hypothesis)

        # re-order the track with the new state correct
        new_track = Track()
        for itrack in range(len(track)):
            if itrack == tb_index:
                new_track.append(track[itrack])
                new_track.append(post)
            else:
                new_track.append(track[itrack])

        # Re-assign to the track the newly ordered tracks
        track=new_track

        # resample the particles, if necessary
        particle_prior = resampler.resample(new_track[-1])

# %%
# Visualiase the tracks and measurements
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Create the detections scans for the plotting
scans_detections = [item[1] for item in arrival_time_ordered]

# Load the plotter
from stonesoup.plotter import AnimatedPlotterly

plotter = AnimatedPlotterly(timestamps)
plotter.plot_ground_truths(truths, [0, 2])

plotter.plot_measurements(scans_detections, [0, 2])
plotter.plot_tracks(track, [0, 2], track_label='Track with OOSM',
                    line= dict(color='blue'))
plotter.plot_tracks(track2, [0, 2], track_label='Track without OOSM')
plotter.fig

# %%
# Conclusion
# ----------
# In this example, we have presented a method on how to deal with OOSM using particle filters.
# This algorithm, that works by inserting the delayed measurements in the particle history track, allows
# to have better tracking performances and not discard any information from the sensors, to validate that
# we made a 1-to-1 comparison with a tracker which systematically ignores OOSM.

# %%
# References
# ----------
# .. [#] M. Orton and A. Marrs, 2005, Particle filters for tracking with out-of-sequence measurements,
#        IEEE Transactions on Aerospace and Electronic Systems.
#
# .. [#] S. R. Maskell, R. G. Everitt, R. Wright, M. Briers, 2005,
#        Multi-target out-of-sequence data association: Tracking using
#        graphical models, Information Fusion.
#
# .. [#] M. Orton and A. Marrs, 2001, A Bayesian approach to multi-target tracking and data fusion
#        with out-of-sequence measurements, IEE Target Tracking: Algorithms and Applications.
