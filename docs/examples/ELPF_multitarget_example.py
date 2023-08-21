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
particle filter. This example follows the implementation that can be found in [#]_.

"""

# %%
# Layout
# ------
# The layout of this example follows:
#
# 1) Create the targets ground truths transition models;
# 2) Generate the targets detections;
# 3) Set up the simulation using a particle filter, data associator and the tracker components;
# 4) Tun the simulation and create the plots.
#

# Some general imports
import numpy as np
from datetime import datetime
from datetime import timedelta
from matplotlib import pyplot as plt
from scipy.stats import uniform

# Load transition models and GroundTruths models
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, ConstantVelocity)
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.array import StateVector
from stonesoup.types.state import GaussianState
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import TrueDetection
from stonesoup.types.detection import Clutter

# %%
# 1) Create Ground truth model, detections and clutter
# ----------------------------------------------------
# Firstly we initialise the transition and measurement models. We initialise the transition model using a
# linear gaussian model and we create the tracks of the targets.

# Define some general parameters before running the simulation
start_time = datetime.now()  # simulation start time
np.random.seed(1908)         # set the random seed
prob_detect = 0.9            # Probability of detection
num_iter = 100               # Number of timesteps to run
clutter_rate = .01           # clutter rate
surveillance_region = [[-10, 30], [-10, 30]]  # The surveillance region x=[-10, 30], y=[-10, 30]
surveillance_area = (surveillance_region[0][1] - surveillance_region[0][0]) \
                    * (surveillance_region[1][1] - surveillance_region[1][0])
clutter_intensity = clutter_rate / surveillance_area

# Models - initialise the models for the Groundtruth
transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.05), ConstantVelocity(0.05)])

measurement_model = LinearGaussian(4, mapping=[0, 2], noise_covar=np.diag([0.5, 0.5]))

# defined the ground truth transition model, in this case we have two linear tracks
gnd_transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.),
                                                              ConstantVelocity(0.)])

# Create a set for the true detections
truths = set()

# First target track
truth = GroundTruthPath([GroundTruthState([0, 0.2, 0, 0.2], timestamp=start_time)])
for k in range(1, num_iter + 1):
    truth.append(GroundTruthState(
        gnd_transition_model.function(truth[k - 1], noise=True,
                                      time_interval=timedelta(seconds=1)),
        timestamp=start_time + timedelta(seconds=k)))
truths.add(truth)

# Second target track
truth = GroundTruthPath([GroundTruthState([0, 0.2, 20, -0.2], timestamp=start_time)])
for k in range(1, num_iter + 1):
    truth.append(GroundTruthState(
        gnd_transition_model.function(truth[k - 1], noise=True,
                                      time_interval=timedelta(seconds=1)),
        timestamp=start_time + timedelta(seconds=k)))
truths.add(truth)

# Create a list of timestamps for the tracks
timestamps = [start_time]
for k in range(1, num_iter + 1):
    timestamps.append(start_time + timedelta(seconds=k))

# %%
# 2) Simulate the detections
# --------------------------
#
# With the ground truth tracks defined now we have to store the various true detections
# and the clutter present in each scan that will be used by the tracker.

# Define the temporal scans
scans = []

# Loop over the timesteps to collect the true detections and the clutter
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
        y = uniform.rvs(-10, 30)
        measurement_set.add(Clutter(np.array([[x], [y]]), timestamp=truth[k].timestamp,
                                    measurement_model=measurement_model))
    scans.append((timestamps[k], measurement_set))

# $$
# 3) set up the Particle filter components and the detector
# ---------------------------------------------------------
# At this stage we have generated a series of GroundTruth detections
# for the two targets and some Clutter detections.
# Now we can create a measurement detector and
# prepare the particle filter components.
# As presented in previous tutorials and examples, we need a predictor,
# an updater and, in the specific case of particle filters, a resampler for
# the filter.


# Load the components for a Buffered detection reader
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.reader import DetectionReader

class SimpleDetector(DetectionReader):
    @BufferedGenerator.generator_method
    def detections_gen(self):
        for timestamp, measurement_set in scans:
            yield timestamp, measurement_set


# Load the particle predictor
from stonesoup.predictor.particle import ParticlePredictor
predictor = ParticlePredictor(transition_model)

# Load the resampler
from stonesoup.resampler.particle import SystematicResampler
resampler = SystematicResampler()

# Load the particle updater, at this stage we don't need the resampler component
# inside the particle updater. We can, also, not specify the measurement model in the
# updater
from stonesoup.updater.particle import ParticleUpdater
updater = ParticleUpdater(measurement_model)

# %%
# After having initialised the predictor, updater and resampler we consider the data associator.
# This example uses a joint probabilistic data association (JPDA, or PDA in the single target case).
# The tracker can use a distance data associator as Nearest Neighbour (Global in the multi-target case) using
# the distances of the points from the expected tracks.
# In this case we use a JPDA data associator, it is possible to extend this using efficient
# hypothesys management (EHM) like from PyEHM.

from stonesoup.hypothesiser.probability import PDAHypothesiser
hypothesiser = PDAHypothesiser(predictor=predictor,
                               updater=updater,
                               clutter_spatial_density=clutter_intensity,
                               prob_detect=prob_detect,
                               prob_gate=0.9999
                               )
# load the data associator probability
from stonesoup.dataassociator.probability import JPDA
data_associator = JPDA(hypothesiser)

# %%
# Initiator & Deleter
# With the data associator component we need to initialise the track initiator and the deleter
# to match the particles with the tracks. For the deleter we consider a :class:`~.UpdateTimeDeleter`.
# For the initiator, in general multi-target examples we usually use a :class:`~.MultiMeasurementInitiator`
# however we are using particles, we can adopt a :class:`~.GaussianParticleInitiator`.

# We consider a time based deleter.
from stonesoup.deleter.time import UpdateTimeStepsDeleter
deleter = UpdateTimeStepsDeleter(time_steps_since_update=4)

# Initialise the prior state
from stonesoup.initiator.simple import SimpleMeasurementInitiator, GaussianParticleInitiator
prior_state = GaussianState(
    StateVector(np.array([10., 0.0, 10., 0.0])),
    np.diag([5, 0.5, 5, 0.5])**2
)

initiator_part = SimpleMeasurementInitiator(prior_state=prior_state,
                                            measurement_model=measurement_model)

# Gaussian particle initiator
initiator = GaussianParticleInitiator(initiator=initiator_part,
                                      number_particles=500)

# %%
# 4) Run the simulation
# ---------------------
# We have initialised all the relevant components needed for the tracker,
# we can now pass these all to the :class:`~.MultiTargetExpectedLikelihoodParticleFilter`
# tracker and generate the various tracks and finally plot them.

from stonesoup.tracker.particle import MultiTargetExpectedLikelihoodParticleFilter

tracker = MultiTargetExpectedLikelihoodParticleFilter(
    initiator=initiator,
    deleter=deleter,
    detector=SimpleDetector(),
    data_associator=data_associator,
    updater=updater,
    resampler=resampler)  # if the resampler is not specified, :class:`~.SystematicResampler` is used

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
# This concludes this short tutorial on how to use expected likelihood particle filter
# in the multi-targets case using a particle filter using Stone Soup components.

# %%
# References
# ----------
# [1] Marrs, A., Maskell, S., and Bar-Shalom, Y., “Expected likelihood for tracking in clutter with
# particle filters”, in Signal and Data Processing of Small Targets 2002, 2002, vol. 4728,
# pp. 230–239. doi:10.1117/12.478507
