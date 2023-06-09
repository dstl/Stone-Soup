"""
This example demonstrates a simple hierarchical tracker:

                     |----------------|
                     |  Top Tracker   |
                     |----------------|
                             |
                    --------------------
                   |                   |
          |----------------|  |----------------|
          | Fuse Tracker 1 |  | Fuse Tracker 2 |
          |----------------|  |----------------|
                  |                   |
          -------------------         ------------
         |                  |                    |
|----------------|  |----------------|  |----------------|
| Leaf Tracker 1 |  | Leaf Tracker 2 |  | Leaf Tracker 3 |
|----------------|  |----------------|  |----------------|
        |                   |                    |
|----------------|  |----------------|   |----------------|
| Sensor 1       |  | Sensor 2       |   | Sensor 3       |
|----------------|  |----------------|   |----------------|

"""
import numpy as np
from datetime import datetime, timedelta
from copy import deepcopy, copy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from stonesoup.custom.sensor.movable import MovableUAVCamera
from stonesoup.custom.tracker import SMCPHD_JIPDA
from stonesoup.custom.types.tracklet import SensorTracks
from stonesoup.custom.initiator.twostate import TwoStateInitiator
from stonesoup.types.numeric import Probability
from stonesoup.types.state import State, GaussianState
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.platform.base import MovingPlatform
from stonesoup.models.transition.linear import (CombinedLinearGaussianTransitionModel,
                                                ConstantVelocity, KnownTurnRate,
                                                NthDerivativeDecay,
                                                OrnsteinUhlenbeck)
from stonesoup.platform.base import MultiTransitionMovingPlatform
from stonesoup.simulator.simple import DummyGroundTruthSimulator
from stonesoup.types.track import Track
from stonesoup.types.update import Update, GaussianStateUpdate
from stonesoup.gater.distance import DistanceGater
from stonesoup.plugins.pyehm import JPDAWithEHM2
from stonesoup.measures import Mahalanobis

from utils import plot_cov_ellipse

from stonesoup.custom.hypothesiser.probability import PDAHypothesiser
from stonesoup.custom.simulator.platform import PlatformTargetDetectionSimulator
from stonesoup.custom.predictor.twostate import TwoStatePredictor
from stonesoup.custom.updater.twostate import TwoStateKalmanUpdater
from stonesoup.custom.reader.tracklet import TrackletExtractor, PseudoMeasExtractor
from stonesoup.custom.tracker.fuse import FuseTracker2


def to_single_state(tracks):
    """ Converts a set of tracks with two-state vectors to a set of tracks with one-state vectors"""
    new_tracks = set()
    for track in tracks:
        states = []
        for state in track.states:
            if isinstance(state, Update):
                new_state = GaussianStateUpdate(state.state_vector[6:], state.covar[6:, 6:],
                                                hypothesis=state.hypothesis,
                                                timestamp=state.timestamp)
            else:
                new_state = GaussianState(state.state_vector[6:], state.covar[6:, 6:],
                                          timestamp=state.timestamp)
            states.append(new_state)
        new_tracks.add(Track(id=track.id, states=states))
    return new_tracks


# Parameters
np.random.seed(1000)
clutter_rate = 1  # Mean number of clutter points per scan
max_range = 50  # Max range of sensor (meters)
surveillance_area = np.pi * max_range ** 2  # Surveillance region area
clutter_density = clutter_rate / surveillance_area  # Mean number of clutter points per unit area
prob_detect = 0.9  # Probability of Detection
num_timesteps = 151  # Number of simulation timesteps
PLOT = True  # Plot the results or not

# Simulation components
# ---------------------
# In this simulation, we have 3 platforms, each with a sensor. The sensors are mounted on the
# platforms and can move with them. The platforms are moving in a straight line at constant
# velocity. There also exists a (non-cooperative) target that is also moving in a straight line at
# constant velocity.

# Simulation start time
start_time = datetime.now()

# Define transition model and position for 3D platform
platform_transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.),
                                                                   ConstantVelocity(0.),
                                                                   ConstantVelocity(0.)])

# Create platforms. Each platform has a sensor and a transition model. The platform's sensor can
# only detect targets within its field of view (FOV), but not itself.
init_states = [State(StateVector([-50, 0, -25, 1, 0, 0]), start_time),
               State(StateVector([50, 0, -25, 1, 0, 0]), start_time),
               State(StateVector([-25, 1, 50, 0, 0, 0]), start_time)]
platforms = []
for i, init_state in enumerate(init_states):
    # Platform
    platform = MovingPlatform(states=init_state,
                              position_mapping=(0, 2, 4),
                              velocity_mapping=(1, 3, 5),
                              transition_model=platform_transition_model)

    # Sensor
    sensor = MovableUAVCamera(ndim_state=6, mapping=[0, 2, 4],
                              noise_covar=np.diag([.1, .1, .1]),
                              mounting_offset=StateVector([0, 0, 0]),
                              rotation_offset=StateVector([0, 0, 0]),
                              fov_radius=max_range,
                              limits=None,
                              fov_in_km=False)
    platform.add_sensor(sensor)
    platforms.append(platform)

# The (non-cooperative) target
cv_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.), ConstantVelocity(0.), ConstantVelocity(0.)])
init_state_gnd = State(StateVector([25, -1, 25, -1, 0, 0]), start_time)
target = MovingPlatform(transition_model=cv_model,
                        states=init_state_gnd,
                        position_mapping=(0, 2, 4),
                        velocity_mapping=(1, 3, 5),
                        sensors=None)

# Simulation timestamps
times = np.arange(0, num_timesteps, 1)
timestamps = [start_time + timedelta(seconds=float(elapsed_time)) for elapsed_time in times]

# A dummy ground truth simulator, which simply acts as a clock
gnd_simulator = DummyGroundTruthSimulator(times=timestamps)

# Detection simulators (1 for each platform)
detector1 = PlatformTargetDetectionSimulator(groundtruth=gnd_simulator, platforms=[platforms[0]],
                                             targets=[platforms[1], platforms[2], target])
detector2 = PlatformTargetDetectionSimulator(groundtruth=gnd_simulator, platforms=[platforms[1]],
                                             targets=[platforms[0], platforms[2], target])
detector3 = PlatformTargetDetectionSimulator(groundtruth=gnd_simulator, platforms=[platforms[2]],
                                             targets=[platforms[0], platforms[1], target])
all_detectors = [detector1, detector2, detector3]

# Hierarchical tracking components
# --------------------------------
# In this section, we define the components of the hierarchical trackers. Recall that the
# hierarchy is as follows:
# 1. 3 Leaf trackers (one for each sensor)
# 2. 2 Branch fuse trackers:
#   a. One that fuses the tracks from leaf trackers 1 and 2
#   b. One that fuses the tracks from leaf tracker 3
# 3. The root (top) fuse tracker that fuses the tracks from the branch trackers

# Leaf trackers (one for each sensor)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Each leaf tracker is a JIPDA tracker that uses a SMC-PHD filter to initialise tracks.
leaf_trackers = []
for i, detector in enumerate(all_detectors):
    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(1)] * 3)
    birth_density = GaussianState(StateVector([0, 0, 0, 0, 0, 0]),
                                  CovarianceMatrix(np.diag([50, 2, 50, 2, 0, 0])))
    prob_death = Probability(0.01)  # Probability of death
    prob_birth = Probability(0.1)  # Probability of birth
    prob_survive = Probability(0.99)  # Probability of survival
    birth_rate = 0.02
    num_particles = 2 ** 11
    birth_scheme = 'mixture'
    tracker = SMCPHD_JIPDA(birth_density=birth_density, transition_model=transition_model,
                           measurement_model=None, prob_detection=prob_detect,
                           prob_death=prob_death, prob_birth=prob_birth,
                           birth_rate=birth_rate, clutter_intensity=clutter_density,
                           num_samples=num_particles, birth_scheme=birth_scheme,
                           start_time=start_time, detector=detector, use_ismcphd=True)
    leaf_trackers.append(tracker)

# Fusion Tracker components
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# The fusion trackers are JPDA trackers, that use the tracks from the leaf trackers as inputs.
# The transition model, predictor, updater, hypothesiser, data associator, and initiator can
# be shared between the fusion trackers.
#
# On the contrary, the tracklet and pseudo-measurement extractors must be defined separately for
# each fusion tracker. This is because, these components perform caching of the tracks and
# pseudo-measurements, generated at each time step.

# Transition model
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(1)] * 3)

# Predictors and updaters
two_state_predictor = TwoStatePredictor(transition_model)
two_state_updater = TwoStateKalmanUpdater(None, True)

# Hypothesiser
hypothesiser1 = PDAHypothesiser(predictor=None,
                                updater=two_state_updater,
                                clutter_spatial_density=Probability(-80, log_value=True),
                                prob_detect=Probability(prob_detect),
                                prob_gate=Probability(0.99),
                                predict=False,
                                per_measurement=True)
hypothesiser1 = DistanceGater(hypothesiser1, Mahalanobis(), 10)  # Uncomment to use JPDA+EHM2

# Data associator
fuse_associator = JPDAWithEHM2(hypothesiser1)  # in Fuse tracker

# Initiator
prior = GaussianState(StateVector([0, 0, 0, 0, 0, 0]),
                      CovarianceMatrix(np.diag([50, 5, 50, 5, 0, 0])))        # Uncomment for GNN in Fuse Tracker
initiator1 = TwoStateInitiator(prior, transition_model, two_state_updater)

# Fuse tracker 1 (fuses tracks from leaf trackers 1 and 2)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tracklet_extractor = TrackletExtractor(transition_model=transition_model,
                                       fuse_interval=timedelta(seconds=2))
pseudomeas_extractor = PseudoMeasExtractor()
fuse_tracker1 = FuseTracker2(initiator=initiator1, predictor=two_state_predictor,
                             updater=two_state_updater, associator=fuse_associator,
                             tracklet_extractor=tracklet_extractor,
                             pseudomeas_extractor=detector, death_rate=1e-4,
                             prob_detect=Probability(prob_detect),
                             delete_thresh=Probability(0.1))

# Fuse tracker 2 (fuses tracks from leaf tracker 3)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tracklet_extractor2 = TrackletExtractor(transition_model=transition_model,
                                        fuse_interval=timedelta(seconds=2))
pseudomeas_extractor2 = PseudoMeasExtractor()
fuse_tracker2 = FuseTracker2(initiator=initiator1, predictor=two_state_predictor,
                             updater=two_state_updater, associator=fuse_associator,
                             tracklet_extractor=tracklet_extractor,
                             pseudomeas_extractor=detector, death_rate=1e-4,
                             prob_detect=Probability(prob_detect),
                             delete_thresh=Probability(0.1))

# Root tracker (fuses tracks from the branch trackers)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tracklet_extractor3 = TrackletExtractor(transition_model=transition_model,
                                        fuse_interval=timedelta(seconds=4))
pseudomeas_extractor3 = PseudoMeasExtractor()
fuse_tracker3 = FuseTracker2(initiator=initiator1, predictor=two_state_predictor,
                             updater=two_state_updater, associator=fuse_associator,
                             tracklet_extractor=tracklet_extractor2,
                             pseudomeas_extractor=detector, death_rate=1e-4,
                             prob_detect=Probability(prob_detect),
                             delete_thresh=Probability(0.1))

# Run the simulation
# ------------------
sim_start_time = datetime.now()
tracks = set()
if PLOT:
    plt.figure(figsize=(10, 10))
    plt.ion()

# We use the leaf trackers as our clock for the simulation. Each leaf tracker provides an
# iterator over the tracks it generates at each time step. We use the `zip` function to
# iterate over the tracks from all the leaf trackers simultaneously.
for (timestamp, tracks1), (_, tracks2), (_, tracks3) in zip(*leaf_trackers):

    # Run Fuse tracker 1
    # ~~~~~~~~~~~~~~~~~~~~
    # Group tracks from leaf trackers 1 and 2
    alltracks1 = [SensorTracks(tracks, i, transition_model) for i, tracks
                  in enumerate([tracks1, tracks2])]
    # Extract tracklets
    tracklets1 = tracklet_extractor.extract(alltracks1, timestamp)
    # Extract pseudo-measurements
    scans1 = pseudomeas_extractor.extract(tracklets1, timestamp)
    # Generate fused tracks
    ctracks1 = fuse_tracker1.process_scans(scans1)

    # Run Fuse tracker 2
    # ~~~~~~~~~~~~~~~~~~~~
    # Group tracks from leaf tracker 3
    alltracks2 = [SensorTracks(tracks3, 2, transition_model)]
    # Extract tracklets
    tracklets2 = tracklet_extractor2.extract(alltracks2, timestamp)
    # Extract pseudo-measurements
    scans2 = pseudomeas_extractor2.extract(tracklets2, timestamp)
    # Generate fused tracks
    ctracks2 = fuse_tracker2.process_scans(scans2)

    # Run Root tracker
    # ~~~~~~~~~~~~~~~~
    # Convert two-state tracks to single-state tracks
    ctracks11 = to_single_state(ctracks1)
    ctracks22 = to_single_state(ctracks2)
    # Group tracks from Fuse trackers 1 and 2
    alltracks3 = [SensorTracks(tracks, i, transition_model) for i, tracks
                  in enumerate([ctracks11, ctracks22])]
    # Extract tracklets
    tracklets3 = tracklet_extractor3.extract(alltracks3, timestamp)
    # Extract pseudo-measurements
    scans3 = pseudomeas_extractor3.extract(tracklets3, timestamp)
    # Generate fused tracks
    ctracks3 = fuse_tracker3.process_scans(scans3)

    # Store tracks
    tracks.update(ctracks3)

    # Print progress
    print(f'{timestamp - start_time} - No. Tracks: {len(ctracks3)}')

    # Plot
    if PLOT:
        plt.clf()
        colors = ['r', 'g', 'b']

        # Plot groundtruth
        data = np.array([state.state_vector for state in target])
        plt.plot(data[:, 0], data[:, 2], '--k', label='Groundtruth (Target)')
        for i, (platform, color) in enumerate(zip(platforms, colors)):
            data = np.array([state.state_vector for state in platform])
            plt.plot(data[:, 0], data[:, 2], f'--{color}')

        # Plot sensor FOVs
        ax1 = plt.gca()
        for j, platform in enumerate(platforms):
            sensor = platform.sensors[0]
            circle = plt.Circle((sensor.position[0], sensor.position[1]), radius=sensor.fov_radius,
                                color=colors[j],
                                fill=False,
                                label=f'Sensor {j + 1}')
            ax1.add_artist(circle, )

        # Plot detections
        all_detections = [detector.detections for detector in all_detectors]
        for i, (detections, color) in enumerate(zip(all_detections, colors)):
            for detection in detections:
                model = detection.measurement_model
                x, y = detection.state_vector[0], detection.state_vector[1]
                plt.plot(x, y, f'{color}x')

        # Plot tracks from Fuse tracker 1
        for track in ctracks1:
            data = np.array([state.state_vector for state in track])
            plot_cov_ellipse(track.covar[[6, 8], :][:, [6, 8]], track.state_vector[[6, 8], :],
                             edgecolor='r', facecolor='none', ax=ax1)
            plt.plot(data[:, 6], data[:, 8], '-*c')

        # Plot tracks from Fuse tracker 2
        for track in ctracks2:
            data = np.array([state.state_vector for state in track])
            plot_cov_ellipse(track.covar[[6, 8], :][:, [6, 8]], track.state_vector[[6, 8], :],
                             edgecolor='r', facecolor='none', ax=ax1)
            plt.plot(data[:, 6], data[:, 8], '-*r')

        # Plot tracks from Root tracker
        for track in tracks:
            data = np.array([state.state_vector for state in track])
            plot_cov_ellipse(track.covar[[6, 8], :][:, [6, 8]], track.state_vector[[6, 8], :],
                             edgecolor='r', facecolor='none', ax=ax1)
            plt.plot(data[:, 6], data[:, 8], '-*m')

        # Add legend info
        for i, color in enumerate(colors):
            plt.plot([], [], f'--{color}', label=f'Groundtruth (Sensor {i + 1})')
            plt.plot([], [], f':{color}', label=f'Tracklets (Sensor {i + 1})')
            plt.plot([], [], f'x{color}', label=f'Detections (Sensor {i + 1})')
        plt.plot([], [], f'-*m', label=f'Fused Tracks')

        plt.legend(loc='upper right')
        plt.xlim((-200, 200))
        plt.ylim((-200, 200))
        plt.pause(0.01)

print(datetime.now() - sim_start_time)
