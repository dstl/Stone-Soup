#!/usr/bin/env python

"""
Expected Likelihood Particle Filters
====================================
The problem of target tracking in clutter is always a complex problem, in particular when dealing with
non-linear or non-gaussian trajectories.

This example shows the implementation of a Particle tracker for multi target case in a cluttered scenario.
This example uses the probabilistic data association (J)PDA to assign particles to tracks, however it is possible
to use (G)NN (nearest neighbour) data association methods. The Tracker Particle ~class has also the
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
# 3) Setup the particle filter, data associator and the tracker components
# 4) The tracks results are plotted.
#

# Load various packages
import numpy as np
from datetime import datetime
from datetime import timedelta
from matplotlib import pyplot as plt
from scipy.stats import uniform

# Load transition models
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, ConstantVelocity)
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.array import StateVector
from stonesoup.types.state import GaussianState


# %%
# 1) Create Groundtruth
# ^^^^^^^^^^^^^^^^^^^^^
# Firstly we initialise the transition and measurement models

# Define some general parameters before running the simulation
start_time = datetime.now()  # simulation start time
prob_detect = 0.9            # Probability of detection
num_iter = 100               # Number of timesteps to run
clutter_rate = .01           # clutter rate
surveillance_region = [[-10, 30], [0, 30]]  # The surveillance region x=[-10, 30], y=[0, 30]
surveillance_area = (surveillance_region[0][1] - surveillance_region[0][0]) \
                    * (surveillance_region[1][1] - surveillance_region[1][0])
clutter_intensity = clutter_rate / surveillance_area

# Models - initialise the models for the Groundtruth
transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.05), ConstantVelocity(0.05)])

measurement_model = LinearGaussian(4, mapping=[0, 2], noise_covar=np.diag([0.5, 0.5]))

# %%

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import TrueDetection
from stonesoup.types.detection import Clutter

# defined the ground truth transtion model, in this case we have two linear tracks
gnd_transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.),
                                                              ConstantVelocity(0.)])

truths = set()
# first object
truth = GroundTruthPath([GroundTruthState([0, 0.2, 0, 0.2], timestamp=start_time)])
for k in range(1, num_iter + 1):
    truth.append(GroundTruthState(
        gnd_transition_model.function(truth[k - 1], noise=True,
                                      time_interval=timedelta(seconds=1)),
        timestamp=start_time + timedelta(seconds=k)))
truths.add(truth)
# second object
truth = GroundTruthPath([GroundTruthState([0, 0.2, 20, -0.2], timestamp=start_time)])
for k in range(1, num_iter + 1):
    truth.append(GroundTruthState(
        gnd_transition_model.function(truth[k - 1], noise=True,
                                      time_interval=timedelta(seconds=1)),
        timestamp=start_time + timedelta(seconds=k)))
truths.add(truth)

# create a list of timestamsp
timestamps = [start_time]
for k in range(1, num_iter + 1):
    timestamps.append(start_time + timedelta(seconds=k))

# Simulate measurements
# =====================
scans = []

for k in range(num_iter):
    measurement_set = set()

    # True detections
    for truth in truths:
        # Generate actual detection from the state with a 10% chance that no detection is received.
        if np.random.rand() <= prob_detect:
            measurement = measurement_model.function(truth[k], noise=True)
            measurement_set.add(TrueDetection(state_vector=measurement,
                                              groundtruth_path=truth,
                                              timestamp=truth[k].timestamp,
                                              measurement_model=measurement_model))

        # Generate clutter at this time-step
        truth_x = truth[k].state_vector[0]
        truth_y = truth[k].state_vector[2]

    # Clutter detections
    for _ in range(np.random.poisson(0.1)):
        x = uniform.rvs(-10, 30)
        y = uniform.rvs(0, 25)
        measurement_set.add(Clutter(np.array([[x], [y]]), timestamp=truth[k].timestamp,
                                    measurement_model=measurement_model))
    scans.append((timestamps[k], measurement_set))

# %%
# At this point we have generated a series of GroundTruth measurements
# for the two tracks, some Clutter detections and detections now we can create a detector
# and prepare the pieces for the particle filter

# %%
# Generate a simple detector
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.reader import DetectionReader

class SimpleDetector(DetectionReader):
    @BufferedGenerator.generator_method
    def detections_gen(self):
        for timestamp, measurement_set in scans:
            yield timestamp, measurement_set

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
# Load the particle updater, at this stage we don't need the Resampler inside the particle updater
from stonesoup.updater.particle import ParticleUpdater
updater = ParticleUpdater(measurement_model)

# %%
# After having initialised the predictor, updater and resampler we consider the data associator.
# This example uses a joint probabilistic data association (JPDA, or PDA in the single target case).
# The tracker can also use a distance data associator as Nearest Neighbour (Global in the multi-target case) using
# the distances of the points from the expected tracks.

from stonesoup.hypothesiser.probability import PDAHypothesiser
hypothesiser = PDAHypothesiser(predictor=predictor,
                               updater=updater,
                               clutter_spatial_density=clutter_intensity,
                               prob_detect=prob_detect,
                               prob_gate=0.9999
                               )
# %%
# In this case we use a JPDA data associator, it is possible to extend this using efficient
# hypothesys management (EHM) like from PyEHM.
from stonesoup.dataassociator.probability import JPDA
data_associator = JPDA(hypothesiser)

# %%
# Initiator & Deleter
# With the data associator we need to initialise the track initiator and the deleter
# to match the particles with the tracks. For the deleter we consider a variance based deleter.
# For the initiator, in general multi-target examples we usually use a MultiMeasurementInitiator however
# in this case, since we are using particles we can easily adopt a GaussianParticleInitiator

# %%
# We consider a time based deleter.
from stonesoup.deleter.time import UpdateTimeStepsDeleter
deleter = UpdateTimeStepsDeleter(time_steps_since_update=5)

# %%
# Initialise the prior state
from stonesoup.initiator.simple import SimpleMeasurementInitiator, GaussianParticleInitiator
prior_state = GaussianState(
    StateVector(np.array([10., 0.0, 10., 0.0])),
    np.diag([10., 1., 10., 1.])**2
)

initiator_part = SimpleMeasurementInitiator(prior_state=prior_state,
                                            measurement_model=measurement_model)

initiator = GaussianParticleInitiator(initiator=initiator_part,
                                      number_particles=1000)

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
    detector=SimpleDetector(),
    data_associator=data_associator,
    updater=updater,
    resampler=resampler)  # if the resampler is not specified, ~SystematicResampler is used


# %%
# Generate the plots
fig1 = plt.figure(figsize=(13, 7))
ax1 = plt.gca()

for k, (timestamp, tracks) in enumerate(tracker):
    ax1.cla()
    for i, truth in enumerate(truths):
        data = np.array([s.state_vector for s in truth[:k + 1]])
        ax1.plot(data[:, 0], data[:, 2], '--', label=f'Groundtruth Track {i + 1}')
    for i, track in enumerate(tracks):
        data = np.array([s.mean for s in track[:k + 1]])
        ax1.plot(data[:, 0], data[:, 2], '--', label=f'Track {i + 1}')
        ax1.plot(track.state.state_vector[0, :], track.state.state_vector[2, :],
                 'r.', label='Particles')
    plt.axis([*surveillance_region[0], *surveillance_region[1]])
    plt.legend(loc='center right')
    plt.pause(0.01)

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