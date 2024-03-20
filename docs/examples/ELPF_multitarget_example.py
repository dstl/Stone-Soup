#!/usr/bin/env python
# coding: utf-8

"""
====================================
Expected Likelihood Particle Filters
====================================
"""

# %%
# Target tracking in cluttered environments is always a complex problem, in particular when dealing with
# non linear or non gaussian trajectories.
#
# This example shows the implementation of the expected likelihood particle tracker (ELPF) for
# a multi target case in a cluttered scenario (see [#]_ for details). This particle filter method is based
# on the expected likelihood obtained from the mixture model of the target measurement noise and clutter
# statistics, and it has proven to be lighter in computational needs and with higher accuracy
# in cluttered scenarios.
#
# This example uses the probabilistic data association (J)PDA to assign detections to tracks,
# this implementation is not limited to a probabilistic data associator, indeed it is possible
# to use the a distance based data associator as (G)NN (nearest neighbour).
# In comparison we consider a distance based particle filter implementation to understand the benefit of
# this approach.
#
# In this example we employ a JPDA probabilistic data associator, however as shown in other examples it is
# possible to use Efficient Hypothesis Management (EHM) via the Stone Soup plugin
# PyEHM (`PyEHM <https://github.com/sglvladi/pyehm>`_).
#
# The layout of this example follows:
#
# 1. The ground truths are created using the transition model;
# 2. Generate scans containing detections and clutter;
# 3. Setup the particle filter, data associators and tracker components;
# 4. Final tracks results are plotted.
#

# %%
# General imports
# ^^^^^^^^^^^^^^^
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import uniform

# Simulation parameters
np.random.seed(1908)  # Set a random seed
simulation_start = datetime.now().replace(microsecond=0)
prob_detect = 0.95  # detection probability
simulation_timesteps = 100

# Clutter parameters
clutter_rate = 1
clutter_area = np.array([[-1, 1], [-1, 1]]) * 50
surveillance_area = ((clutter_area[0][1] - clutter_area[0][0])*
                     (clutter_area[1][1] - clutter_area[1][0]))
clutter_spatial_density = clutter_rate/surveillance_area

# %%
# Stone Soup imports
# ^^^^^^^^^^^^^^^^^^
from stonesoup.models.transition.linear import \
    CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.types.state import GaussianState
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import TrueDetection
from stonesoup.types.detection import Clutter

# %%
# 1. Create Groundtruth
# ^^^^^^^^^^^^^^^^^^^^^
# Firstly we initialise the transition and measurement models

# transition model for the particle filer
transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.1), ConstantVelocity(0.1)])

# a noiseless transition model for the ground truths
gnd_transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.),
                                                              ConstantVelocity(0.)])

# Define a detection models
measurement_model = LinearGaussian(ndim_state=4,
                                   mapping=(0, 2),
                                   noise_covar=np.diag([10, 10]))

# Generate ground-truths
truths = set()

# Instantiate the first entry
truth = GroundTruthPath([GroundTruthState([-10, 0.25, -10, 0.50], timestamp=simulation_start)])
for k in range(1, simulation_timesteps):
    truth.append(GroundTruthState(
        gnd_transition_model.function(truth[k-1], noise=True,
                                      time_interval=timedelta(seconds=1)),
        timestamp=simulation_start + timedelta(seconds=k)))
truths.add(truth)

truth = GroundTruthPath([GroundTruthState([-10, 0.25, 20, -0.50], timestamp=simulation_start)])
for k in range(1, simulation_timesteps):
    truth.append(GroundTruthState(
        gnd_transition_model.function(truth[k-1], noise=True,
                                      time_interval=timedelta(seconds=1)),
        timestamp=simulation_start + timedelta(seconds=k)))
truths.add(truth)

# generate the timestamps
timestamps = [simulation_start]
for k in range(1, simulation_timesteps):
    timestamps.append(simulation_start + timedelta(seconds=k))

# %%
# 2. Generate scans containing detections and clutter;
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Create a series of scans to collect the detections and clutter
# from the targets.

scans = []
for k in range(simulation_timesteps):
    measurement_set = set()

    # detections
    for truth in truths:
        if np.random.rand() <= prob_detect:
            measurement = measurement_model.function(truth[k], noise=True)
            measurement_set.add(TrueDetection(state_vector=measurement,
                                              groundtruth_path=truth,
                                              timestamp=truth[k].timestamp,
                                              measurement_model=measurement_model))

    # Clutter detections
    for _ in range(np.random.poisson(clutter_rate)):
        x = uniform.rvs(-20, 40)
        y = uniform.rvs(-30, 70)
        measurement_set.add(Clutter(np.array([[x], [y]]), timestamp=truth[k].timestamp,
                                    measurement_model=measurement_model))
    scans.append(measurement_set)

# %%
# Visualise the detections and the ground truth
# ---------------------------------------------

from stonesoup.plotter import AnimatedPlotterly

plotter = AnimatedPlotterly(timesteps=timestamps)
plotter.plot_measurements(scans, [0, 2], measurements_label='Scans')
plotter.plot_ground_truths(truths, [0, 2])
plotter.fig

# %%
# 3. Set up the Particle filter;
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We have a series of scans containing the detections and clutter, now we can load the
# tracker components starting from :class:`~.ParticlePredictor`, :class:`~.ParticleUpdater` and
# resampler which we employ a :class:`~.ESSResampler`. In the ELPF implementation
# the resampler should not be passed to the updater, as usual, but needs to be loaded in the
# tracker object later otherwise it will result in an error.
#
# In addition to the ELPF implementation we present a standard Particle filter
# implementation with a distance based data associator for visual comparison about track
# accuracy.

# Load the predictor
from stonesoup.predictor.particle import ParticlePredictor
predictor = ParticlePredictor(transition_model)

# load the resampler
from stonesoup.resampler.particle import ESSResampler
resampler = ESSResampler()

# load the updater
from stonesoup.updater.particle import ParticleUpdater
updater = ParticleUpdater(measurement_model)

# In the standard implementation we pass the resampler to the updater
pf_updater = ParticleUpdater(measurement_model, resampler)

# %%
# After having initialised the predictor, updater and resampler we consider the data associator.
# This example uses a joint probabilistic data association (:class:`~.JPDA`, or :class:`~.PDA`
# in the single target case) for the ELPF.
# The ELPF can also use a distance-based data associator :class:`~.NearestNeighbour`
# (:class:`~.GNNWith2DAssignment` in the multi-target case) using the distances of the points
# from the tracks.

# load the hypothesiser
from stonesoup.hypothesiser.probability import PDAHypothesiser
hypothesiser = PDAHypothesiser(predictor=predictor,
                               updater=updater,
                               clutter_spatial_density=clutter_spatial_density,
                               prob_detect=prob_detect,
                               prob_gate=0.9999)

from stonesoup.dataassociator.probability import JPDA
data_associator = JPDA(hypothesiser)

# %%
# With the data associator we need to initialise the tracks by using an initiator and a deleter
# to pair the detections with the tracks. We consider a time based
# deleter using :class:`~.UpdateTimeStepsDeleter`.
# We adopt :class:`~.MultiMeasurementInitiator` that behaves as initial tracker, using a
# distance based data associator and Extended Kalman predictor and updater.

from stonesoup.deleter.time import UpdateTimeStepsDeleter
deleter = UpdateTimeStepsDeleter(3)

from stonesoup.initiator.simple import GaussianParticleInitiator, MultiMeasurementInitiator
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis

from stonesoup.updater.kalman import ExtendedKalmanUpdater
initiator_updater = ExtendedKalmanUpdater(measurement_model)
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
initiator_predictor = ExtendedKalmanPredictor(gnd_transition_model)

# Prior state
prior_state = GaussianState(np.array([-10, 0, 0, 0]),
                            np.diag([5, 1, 8, 1])**2,
                            timestamp=simulation_start)

initiator_part = MultiMeasurementInitiator(
    prior_state,
    measurement_model=None,
    deleter=deleter,
    data_associator=GNNWith2DAssignment(
        DistanceHypothesiser(initiator_predictor, initiator_updater, Mahalanobis(), missed_distance=3)),
    updater=initiator_updater,
    min_points=8
    )

initiator = GaussianParticleInitiator(initiator=initiator_part,
                                      number_particles=500)

# %%
# 4. Run the tracker;
# ^^^^^^^^^^^^^^^^^^^
# We have initialised all the relevant components needed for the tracker,
# we can now pass these all to the :class:`~.MultiTargetExpectedLikelihoodParticleFilter`
# tracker and generate the various tracks.
#
# The particle filter uses a :class:`~.MultiTargetTracker` to perform the tracking and
# a :class:`~.GNNWith2DAssignment` data associator and adopts the the same initiator and deleter as
# the ELPF tracker.

# Load the ELPF tracker
from stonesoup.tracker.particle import MultiTargetExpectedLikelihoodParticleFilter
# Load the PF tracker
from stonesoup.tracker.simple import MultiTargetTracker

# ELPF Tracker
tracker = MultiTargetExpectedLikelihoodParticleFilter(
    initiator=initiator,
    deleter=deleter,
    detector=zip(timestamps, scans),
    data_associator=data_associator,
    updater=updater,
    resampler=resampler)  # if the resampler is not specified, ~SystematicResampler is used

# PF tracker
pf_tracker = MultiTargetTracker(
    initiator=GaussianParticleInitiator(initiator=initiator_part,
                                      number_particles=500),
    deleter=deleter,
    detector=zip(timestamps, scans),
    data_associator=GNNWith2DAssignment(
        DistanceHypothesiser(predictor, pf_updater, Mahalanobis(), missed_distance=5)),
    updater=pf_updater)

# loop over the tracker generate the final tracks
tracks = set()
for time, current_tracks in tracker:
    tracks.update(current_tracks)

# gather the tracks from the PF
pf_tracks = set()
for time, current_tracks in pf_tracker:
   pf_tracks.update(current_tracks)

# Visualise the final tracks obtained
plotter.plot_tracks(tracks, [0, 2], track_label="ELPF tracks")
plotter.plot_tracks(pf_tracks, [0, 2], track_label="PF tracks", line=dict(dash='longdash'))
plotter.fig

# %%
# Conclusion
# ----------
# In this example we have presented how to implement and use the Expected Likelihood
# Particle filter in a cluttered multi-target scenario.
#
# This Particle filter implementation allows to use a probabilistic data associator with
# greater accuracy when dealing with tracking in cluttered environments with non-linear detections.
#
# By comparing a simpler distance based particle filter performances, the ELPF implementation
# provides more consistent tracks, even in a simplistic scenario as the one considered.

# %%
# References
# ----------
# .. [#] Marrs, A., Maskell, S., and Bar-Shalom, Y., “Expected likelihood for tracking in clutter with
#        particle filters”, in Signal and Data Processing of Small Targets 2002, 2002, vol. 4728,
#        pp. 230–239. doi:10.1117/12.478507
