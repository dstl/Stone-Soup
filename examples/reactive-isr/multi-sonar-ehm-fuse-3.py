"""
multi-sonar-ehm-fuse.py

This example script simulates 3 moving platforms, each equipped with a single active sonar sensor 
(StoneSoup does not have an implementation of an active sonar so a radar is used instead), and 1 
target. Each sensor generates detections of all other objects (excluding itself).

The tracking configuration is as follows:
- For each sensor whose index in the 'all_detectors' list is not in 'bias_tracker_idx', a
  local tracker is configured that acts like a contact follower and generates Track objects. The
  outputs of these trackers are the fed into the Fusion engine.
- For all other sensors, their data is fed directly into the Fusion engine. Note that the
  TrackletExtractorWithTracker is used here, meaning that a (local) bias estimation tracker is run
  on the data read from each sensor, before it is fed into the main Fuse Tracker (i.e. the
  component of the Fusion Engine that produces the fused tracks).
- The data association algorithm used for both the local and fuse trackers is JPDA with EHM.

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
                                                ConstantVelocity, KnownTurnRate, NthDerivativeDecay,
                                                OrnsteinUhlenbeck)
from stonesoup.platform.base import MultiTransitionMovingPlatform
from stonesoup.simulator.simple import DummyGroundTruthSimulator
from stonesoup.types.update import Update
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

# Parameters
np.random.seed(1000)
clutter_rate = 1                                    # Mean number of clutter points per scan
max_range = 130                                     # Max range of sensor (meters)
surveillance_area = np.pi*max_range**2              # Surveillance region area
clutter_density = clutter_rate/surveillance_area    # Mean number of clutter points per unit area
prob_detect = 0.9                                   # Probability of Detection
num_timesteps = 101                                 # Number of simulation timesteps
PLOT = True

# Simulation start time
start_time = datetime.now()

# Define transition model and position for 3D platform
platform_transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.),
                                                                   ConstantVelocity(0.),
                                                                   ConstantVelocity(0.)])

# Create platforms
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
                              noise_covar=np.diag([3, 3, 3]),
                              mounting_offset=StateVector([0, 0, 0]),
                              rotation_offset=StateVector([0, 0, 0]),
                              fov_radius=max_range,
                              limits=None,
                              fov_in_km=False)
    platform.add_sensor(sensor)
    platforms.append(platform)


# Simulation components

# The target
cv_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.), ConstantVelocity(0.), ConstantVelocity(0.)])
ct_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.), ConstantVelocity(0.), ConstantVelocity(0.)])
manoeuvres = [cv_model, ct_model]
manoeuvre_times = [timedelta(seconds=4), timedelta(seconds=4)]
init_state_gnd = State(StateVector([25, -1, 25, -1, 0, 0]), start_time)
target = MultiTransitionMovingPlatform(transition_models=manoeuvres,
                                       transition_times=manoeuvre_times,
                                       states=init_state_gnd,
                                       position_mapping=(0, 2, 4),
                                       velocity_mapping=(1, 3, 5),
                                       sensors=None)

times = np.arange(0, num_timesteps, 1)
timestamps = [start_time + timedelta(seconds=float(elapsed_time)) for elapsed_time in times]

gnd_simulator = DummyGroundTruthSimulator(times=timestamps)

# Detection simulators (1 for each platform)
detector1 = PlatformTargetDetectionSimulator(groundtruth=gnd_simulator, platforms=[platforms[0]],
                                             targets=[platforms[1], platforms[2], target])
detector2 = PlatformTargetDetectionSimulator(groundtruth=gnd_simulator, platforms=[platforms[1]],
                                             targets=[platforms[0], platforms[2], target])
detector3 = PlatformTargetDetectionSimulator(groundtruth=gnd_simulator, platforms=[platforms[2]],
                                             targets=[platforms[0], platforms[1], target])

all_detectors = [detector1, detector2, detector3]

# Multi-Target Trackers (1 per platform)
base_trackers = []
for i, detector in enumerate(all_detectors):
    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(1)]*3)
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
    base_trackers.append(tracker)

# Fusion Tracker
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(1)]*3)
prior = GaussianState(StateVector([0, 0, 0, 0, 0, 0]),
                      CovarianceMatrix(np.diag([50, 5, 50, 5, 0, 0])))
tracklet_extractor = TrackletExtractor(trackers=None,
                                       transition_model=transition_model,
                                       fuse_interval=timedelta(seconds=3))
pseudomeas_extractor = PseudoMeasExtractor(None, state_idx_to_use=None, use_prior=False)

two_state_predictor = TwoStatePredictor(transition_model)
two_state_updater = TwoStateKalmanUpdater(None, True)
hypothesiser1 = PDAHypothesiser(predictor=None,
                                updater=two_state_updater,
                                clutter_spatial_density=Probability(-80, log_value=True),
                                prob_detect=Probability(prob_detect),
                                prob_gate=Probability(0.99),
                                predict=False,
                                per_measurement=True)
hypothesiser1 = DistanceGater(hypothesiser1, Mahalanobis(), 10)   # Uncomment to use JPDA+EHM2
fuse_associator = JPDAWithEHM2(hypothesiser1)                     # in Fuse tracker
# fuse_associator = GNNWith2DAssignment(hypothesiser1)          # Uncomment for GNN in Fuse Tracker
initiator1 = TwoStateInitiator(prior, transition_model, two_state_updater)
fuse_tracker = FuseTracker2(initiator=initiator1, predictor=two_state_predictor,
                            updater=two_state_updater, associator=fuse_associator,
                            tracklet_extractor=tracklet_extractor,
                            pseudomeas_extractor=detector, death_rate=1e-4,
                            prob_detect=Probability(prob_detect),
                            delete_thresh=Probability(0.1))

sim_start_time = datetime.now()
tracks = set()

if PLOT:
    plt.figure(figsize=(10, 10))
    plt.ion()
for (timestamp, tracks1), (_, tracks2), (_, tracks3) in zip(*base_trackers):

    alltracks = [SensorTracks(tracks, i, transition_model) for i, tracks
                 in enumerate([tracks1, tracks2, tracks3])]

    # Perform fusion

    # _, ctracks = fuse_tracker.process_tracks(alltracks, timestamp)

    # Extract tracklets
    tracklets = tracklet_extractor.extract(alltracks, timestamp)

    # Extract pseudo-measurements
    scans = pseudomeas_extractor.extract(tracklets, timestamp)

    # Process pseudo-measurements
    ctracks = fuse_tracker.process_scans(scans)

    # Update tracks
    tracks.update(ctracks)

    print(f'{timestamp-start_time} - No. Tracks: {len(ctracks)}')
    tracks.update(ctracks)
    # Plot
    if PLOT:
        plt.clf()
        all_detections = [detector.detections for detector in all_detectors]
        colors = ['r', 'g', 'b']
        data = np.array([state.state_vector for state in target])
        plt.plot(data[:, 0], data[:, 2], '--k', label='Groundtruth (Target)')
        for i, (platform, color) in enumerate(zip(platforms, colors)):
            data = np.array([state.state_vector for state in platform])
            plt.plot(data[:, 0], data[:, 2], f'--{color}')

        ax1 = plt.gca()
        for j, platform in enumerate(platforms):
            sensor = platform.sensors[0]
            circle = plt.Circle((sensor.position[0], sensor.position[1]), radius=sensor.fov_radius,
                                color=colors[j],
                                fill=False,
                                label=f'Sensor {j+1}')
            ax1.add_artist(circle, )

        for i, (detections, color) in enumerate(zip(all_detections, colors)):
            for detection in detections:
                model = detection.measurement_model
                x, y = detection.state_vector[0], detection.state_vector[1]
                plt.plot(x, y, f'{color}x')

        for i, (tracklets, color) in enumerate(zip(tracklet_extractor.current[1], colors)):
            for tracklet in tracklets:
                data = np.array([s.mean for s in tracklet.states if isinstance(s, Update)])
                plt.plot(data[:, 6], data[:, 8], f':{color}')

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

        # state_smc = non_bias_trackers[0]._initiator._state
        # plt.plot(state_smc.state_vector[0, :], state_smc.state_vector[2, :], 'r.')

        plt.legend(loc='upper right')
        plt.xlim((-200, 200))
        plt.ylim((-200, 200))
        plt.pause(0.01)

print(datetime.now() - sim_start_time)