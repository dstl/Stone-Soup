#!/usr/bin/env python

"""
Expected Likelihood Particle Filters
====================================
The problem of target tracking in clutter is always a complex problem, in particular when dealing with
non linear or non gaussian  trajectories.

This example shows the implementation of a Particle tracker for multi target case in a cluttered scenario.
This example uses the probabilistic data association (J)PDA to assign particles to tracks, however it is possible
to use (G)NN (nearest neighbour) data association methods. The Tracker Particle class has also the
method for single target case.

To run this example we have to set up the ground truths and the measurements before setting up a simplistic
particle filter.

"""

# %%
# Layout
# ^^^^^^
# The layout of this example is as follows:
#
# 1) The ground truth is created using a simple transition model
# 2) The non-linear detections are generated
# 3) Setup the particle filter, data associators and various tracker components
# 4) The tracks results are plotted.
#

# Load various packages
import datetime
import numpy as np

# Set a random seed
np.random.seed(1908)

# Load transition models
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, ConstantVelocity)
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import GaussianState


# %%
# 1) Create Groundtruth
# ^^^^^^^^^^^^^^^^^^^^^
# Firstly we initialise the transition and measurement models


# Models - initialise the models for the Groundtruth
transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.05), ConstantVelocity(0.05)])

measurement_model = LinearGaussian(4, mapping=[0, 2], noise_covar=np.diag([0.5, 0.5]))

# %%
# Load the simulator - generate truths
from stonesoup.simulator.simple import SimpleDetectionSimulator, MultiTargetGroundTruthSimulator

groundtruth_sim = MultiTargetGroundTruthSimulator(
    transition_model=transition_model,
    initial_state=GaussianState(
        StateVector([[0], [0], [0], [0]]),
        CovarianceMatrix(np.diag([0.5, 1, 0.5, 1]))),  # initial state
    timestep=datetime.timedelta(seconds=5),
    number_steps=100,  # number of steps
    birth_rate=0.1,  # probability of birth of a track
    death_probability=0.05  # probability of death of track
)

prob_detect = 0.9  # detection probability

# %%
# we have generated the groundtruth simulator, now use the simulator to generate measurements
detection_sim = SimpleDetectionSimulator(
    groundtruth=groundtruth_sim,
    measurement_model=measurement_model,
    meas_range=np.array([[-1, 1], [-1, 1]]) * 600,  # Area to generate clutter
    detection_probability=prob_detect,
    clutter_rate=3
)

# %%
detections = set()
ground_truth = set()

for time, dets in detection_sim:
    detections |= dets
    ground_truth |= groundtruth_sim.groundtruth_paths

# %%
# Show the detections and the ground truth
from stonesoup.plotter import Plotterly

plotter = Plotterly()
plotter.plot_measurements(detections, [0,2])
plotter.plot_ground_truths(ground_truth, [0,2])
plotter.fig.show()

# $$
# 3) set up the Particle filter
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# As presented in previous tutorials and examples, for the filters we need a predictor, an updater and,
# in the specific case of particle filters a resampler.

# %%
# Load the particle predictor
from stonesoup.predictor.particle import ParticlePredictor
predictor = ParticlePredictor(transition_model)

# %%
# Load the resampler
from stonesoup.resampler.particle import SystematicResampler
resampler = SystematicResampler()

# %%
# Load the particle updater
from stonesoup.updater.particle import ParticleUpdater
updater = ParticleUpdater(None, resampler)  # We can also use the measurement model instead of None

# %%
# After having initialised the predictor, updater and resampler we consider the data associator.
# This example uses a joint probabilistic data association (JPDA, or PDA in the single target case).
# The tracker can also use a distance data associator as Nearest Neighbour (Global in the multi-target case) using
# the distances of the points from the expected tracks.

from stonesoup.hypothesiser.probability import PDAHypothesiser
hypothesiser = PDAHypothesiser(predictor=predictor,
                               updater=updater,
                               clutter_spatial_density=0.150,
                               prob_detect=prob_detect
                               )
# %%
from stonesoup.dataassociator.probability import JPDA
data_associator = JPDA(hypothesiser)

# %%
# Initiator & Deleter
# With the data associator we need to initialise the track initiator and the deleter
# to match the particles with the tracks. For the deleter we consider a variance based deleter.
# For the initiator, in general multi-target examples we usually use a MultiMeasurementInitiator however
# in this case, since we are using particles we can easily adopt a GaussianParticleInitiator

# %%
from stonesoup.deleter.error import CovarianceBasedDeleter
deleter = CovarianceBasedDeleter(covar_trace_thresh=10, delete_last_pred=True)

# %%
from stonesoup.initiator.simple import SimpleMeasurementInitiator, GaussianParticleInitiator
prior_state = GaussianState(
    StateVector([0,0,0,0]),
    np.diag([0.5, 1, 0.5, 1])**1)

initiator_part = SimpleMeasurementInitiator(prior_state, measurement_model=None,
                                            skip_non_reversible=True)

initiator = GaussianParticleInitiator(initiator=initiator_part,
                                      number_particles=500,
                                      use_fixed_covar=False)

# %%
# 4) Run the tracker
# ^^^^^^^^^^^^^^^^^^
# At this point we have initialised all the relevant components needed for the tracker,
# we can now pass these all to the ELPF tracker and generate the various tracks.

# %%
from stonesoup.tracker.particle import MultiTargetExpectedLikelihoodParticleFilter

tracker = MultiTargetExpectedLikelihoodParticleFilter(
    initiator=initiator,
    deleter=deleter,
    detector=detection_sim,
    data_associator=data_associator,
    updater=updater,
    resampler=resampler)  # if the resampler is not specified, ~SystematicResampler is used

# %%
# Generate the plots
tracks = set()
for step, (time, current_tracks) in enumerate(tracker, 1):
    tracks.update(current_tracks)


plotter.plot_tracks(tracks, [0, 2], track_label="ELPF tracks", line=dict(color="brown"),
                    uncertainty=False)
plotter.fig.show()


# %%
# Key points
# ^^^^^^
# 1)
# 2)
# 3)

"""
References
----------
.. [#] Marrs, A., Maskell, S., and Bar-Shalom, Y., “Expected likelihood for tracking in clutter with 
   particle filters”, in Signal and Data Processing of Small Targets 2002, 2002, vol. 4728, 
   pp. 230–239. doi:10.1117/12.478507
"""