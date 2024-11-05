"""
 Generate a multi-sensor / multi-target simulation and tracking scenario
 The scenario simulates 8 radar sensors positioned on a 4x2 grid of overlapping FOVs, organised
 in 4 hierarchical levels of 8 trackers, 4 of 2 each, 2 of 4 each and 1 of all 8 sensors passed
 up the fusion tree.

                                   |----------------|
                                   |  Top Tracker   |
                                   |----------------|
                                           |
                    --------------------------------------------
                   |                                           |
          |-----------------|                         |-----------------|
          | Fuse Tracker 2a |                         | Fuse Tracker 2b |
          |-----------------|                         |-----------------|
                  |                                           |
          -------------------                        -------------------
         |                  |                       |                  |
|------------------|  |------------------|  |------------------|  |------------------|
|  Fuse Tracker 1a |  |  Fuse Tracker 1b |  |  Fuse Tracker 1c |  |  Fuse Tracker 1d |
|------------------|  |------------------|  |------------------|  |------------------|
    |          |          |          |          |          |          |          |
|-------|  |-------|  |-------|  |-------|  |-------|  |-------|  |-------|  |-------|
| Leaf  |  | Leaf  |  | Leaf  |  | Leaf  |  | Leaf  |  | Leaf  |  | Leaf  |  | Leaf  |
|Sens. 1|  |Sens. 2|  |Sens. 3|  |Sens. 4|  |Sens. 5|  |Sens. 6|  |Sens. 7|  |Sens. 8|
|-------|  |-------|  |-------|  |-------|  |-------|  |-------|  |-------|  |-------|
"""



import matplotlib
from itertools import tee

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

import numpy as np
from datetime import datetime, timedelta

from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import GaussianState, State
from stonesoup.types.numeric import Probability
from stonesoup.types.track import Track
from stonesoup.types.update import Update, GaussianStateUpdate
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator

from stonesoup.platform import FixedPlatform, MovingPlatform
from stonesoup.sensor.radar import RadarBearingRange
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.simulator.simple import DummyGroundTruthSimulator
from stonesoup.simulator.platform import PlatformDetectionSimulator
#from stonesoup.custom.simulator.platform import PlatformTargetDetectionSimulator

from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import UnscentedKalmanUpdater
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.tracker.simple import MultiTargetTracker
from stonesoup.plugins.pyehm import JPDAWithEHM2
from stonesoup.gater.distance import DistanceGater

from stonesoup.custom.predictor.twostate import TwoStatePredictor
from stonesoup.custom.updater.twostate import TwoStateKalmanUpdater
from stonesoup.custom.initiator.twostate import TwoStateInitiator
from stonesoup.custom.types.tracklet import SensorTracks
from stonesoup.custom.reader.tracklet import TrackletExtractor, PseudoMeasExtractor, TrackletExtractorTwoState
from stonesoup.custom.tracker.fuse import FuseTracker2

from stonesoup.custom.hypothesiser.probability import \
    PDAHypothesiser as PDAHypothesiserLyud  # Lyudmil's custom PDA which doesn't have to predict?
from stonesoup.hypothesiser.probability import PDAHypothesiser  # replaced with Lyudmil's custom code

from utils import plot_cov_ellipse#compute_ellipse

from prh_funcs import tile_with_circles, merge_position_and_velocity, to_single_state, fit_normal_to_uniform,\
    merge_position_and_velocity_covariance
from fusion import FusionNode

from output_files import output_tracks, output_meas, output_truth


np.random.seed(1991)

# Whether to use one or two state trackers at the upper hierarchy levels
use_two_state_tracks = True
# Whether to divide out the prior distribution when calculating pseudomeasurements
use_prior = True


# A function to plot the output tracks, including the lower level tracks fed into the higher level trackers
def plot_tracks(all_tracks, all_detections, truth, numxy, fusion_level):
    """
    Plot results for a level of trackers
    """

    numx, numy = numxy[0], numxy[1]
    fig, axs = plt.subplots(numy, numx, squeeze = False)

    for row in range(numy):
        for col in range(numx):

            i = col * numy + row
            ax = axs[numy - row - 1, col]
            fusion_node = fusion_level[i]

            leaf_platforms = [p for x in fusion_node.get_leaf_trackers() for p in x.detector.platforms]
            platform_states = [np.array(p.state_vector.flatten()) for p in leaf_platforms]
            platform_ranges = [p.sensors[0].max_range for p in leaf_platforms]

            # Tracks
            for track in all_tracks[i]:  # leaf:  # [i]:
                track_x = np.array([x.state_vector.flatten() for x in track.states])
                ax.plot(track_x[:, -4], track_x[:, -2], color='r', marker='+')
                for state in track:
                    mn, cv = state.mean[np.ix_([-4, -2])], state.covar[np.ix_([-4, -2], [-4, -2])]
                    # path = compute_ellipse(mn, cv)
                    plot_cov_ellipse(cv, mn,
                                     ax=ax)
            # Ground truth
            for target in truth:
                truth_x = np.array([x.state_vector.flatten() for x in target.states])
                ax.plot(truth_x[:, 0], truth_x[:, 2], color='g', marker='')

            # Platform range
            for (platform_state, platform_range) in zip(platform_states, platform_ranges):#platform_positions[i]:
                circle = plt.Circle(platform_state[np.ix_([0, 2])], platform_range, color='k', fill=False)
                ax.add_patch(circle)

            # Detections (will have problems if the pseudomeasurements are inhomogeneous due to having different
            # rank)
            if all_detections and all_detections[i]:
                if fusion_node.is_leaf():
                    d = np.array([z.measurement_model.inverse_function(z).flatten() for z in all_detections[i]])
                    meas_x = d[:,0]
                    meas_y = d[:,2]
                else:
                    d = np.array([z.state_vector.flatten() for z in all_detections[i]])
                    meas_x = d[:,-4]
                    meas_y = d[:,-2]
                ax.plot(meas_x, meas_y, color='b', marker='.', linestyle='')

            # Axes
            ax.set_xlim([minpos[0], maxpos[0]])
            ax.set_ylim([minpos[1], maxpos[1]])


# Create a leaf-level tracker (i.e. one which processes raw measurements and provides tracks for higher level
# trackers to process
def create_leaf_tracker(platforms, gnd_sim, transition_model, prior_state):

    # Create detection simulator
    detection_sim = PlatformDetectionSimulator(
        groundtruth=gnd_sim,
        platforms=platforms)

    this_meas_model = platforms[0].sensors[0].measurement_model # will this work with multiple platforms/sensors per tracker?
    predictor = KalmanPredictor(transition_model)
    updater = UnscentedKalmanUpdater(this_meas_model)

    # Covariance-based deleted
    deleter = CovarianceBasedDeleter(covar_trace_thresh=10000.0)

    # Create GNN data association
    hypothesiser = PDAHypothesiser(predictor, updater, prob_detect=0.9, prob_gate=0.999,
                                   clutter_spatial_density=1.0e-6)
    data_associator = GNNWith2DAssignment(hypothesiser)

    # Create initiator for this sensor
    min_detections = 1
    initiator = MultiMeasurementInitiator(
        prior_state=prior_state,
        measurement_model=this_meas_model,
        deleter=deleter,
        data_associator=data_associator,
        updater=updater,
        min_points=min_detections)

    # Set up the tracker
    return MultiTargetTracker(
        initiator=initiator,
        deleter=deleter,
        detector=detection_sim,
        data_associator=data_associator,
        updater=updater)

# Create a fusion tracker, i.e. one which takes in tracks, produces pseudomeasurements from them and carries out
# tracking on the pseudomeasurements
# use_two_state specifies if the tracks to be fused consist of single target state distributions (as is the case for
# tracks produced by leaf trackers, or fusion tracks which have been converted to single state), or distributions on
# pairs of states for a track over an intervaL)
def create_fuse_tracker(fusion_time_interval, transition_model, prior_state, use_two_state):

    # Fusion tracker components
    two_state_predictor = TwoStatePredictor(transition_model)
    two_state_updater = TwoStateKalmanUpdater(None, True)
    fuse_initiator = TwoStateInitiator(prior_state, transition_model, two_state_updater)
    # Lyudmil's code uses PDAHypothesiser from stonesoup.custom.hypothesiser.probability
    fuse_hypothesiser = PDAHypothesiserLyud(predictor=None,
                                            updater=two_state_updater,
                                            clutter_spatial_density=Probability(-80, log_value=True),#Probability(-80, log_value=True),
                                            prob_detect=Probability(prob_detect),
                                            prob_gate=Probability(0.99),
                                            predict=False,
                                            per_measurement=True)
    #fuse_associator = JPDAWithEHM2(fuse_hypothesiser)  # in Fuse tracker
    fuse_associator = GNNWith2DAssignment(fuse_hypothesiser)  # in Fuse tracker
    if use_two_state:
        tracklet_extractor = TrackletExtractorTwoState(transition_model=transition_model,
                                                       fuse_interval=fusion_time_interval)
    else:
        tracklet_extractor = TrackletExtractor(transition_model=transition_model,
                                               fuse_interval=fusion_time_interval)

    pseudomeas_extractor = PseudoMeasExtractor(use_prior=True)
    return FuseTracker2(initiator=fuse_initiator, predictor=two_state_predictor,
                        updater=two_state_updater, associator=fuse_associator,
                        tracklet_extractor=tracklet_extractor,
                        pseudomeas_extractor=pseudomeas_extractor, death_rate=1e-4,
                        prob_detect=Probability(prob_detect),
                        delete_thresh=Probability(0.5)) # delete_thresh=Probability(0.0))

# Create a hierarchy of trackers with leaf trackers at the bottom, passing up to multiple layers of fusion trackers
def create_fusion_hierarchy(platforms, gnd_sims, tracker_hierarchy_indices, fusion_times,
                            transition_model, prior_state):
    """
    Create hierarchy of fusion trackers (all the same for now, with different fusion times)
    """
    fusion_hierarchy = []

    # Generate leaf nodes
    fusion_hierarchy.append([])
    for i_node, idx in enumerate(tracker_hierarchy_indices[0]):
        tracker = create_leaf_tracker([platforms[i] for i in idx], gnd_sims[i_node],
                                      transition_model, prior_state)
        fusion_hierarchy[-1].append(FusionNode(tracker, [], statedim, use_two_state_tracks))

    # Generate fusion nodes
    for i_level, (fusion_time, level) in enumerate(zip(fusion_times[1:], tracker_hierarchy_indices[1:])):
        fusion_hierarchy.append([])
        for child_idx in level:
            is_two_state = (i_level > 0 and use_two_state_tracks)
            tracker = create_fuse_tracker(fusion_time, transition_model, prior_state, is_two_state)
            children = [fusion_hierarchy[-2][i] for i in child_idx]
            fusion_hierarchy[-1].append(FusionNode(tracker, children, statedim, use_two_state_tracks))

    return fusion_hierarchy


# Specify maximum and minimum extect of surveillance region
minpos = np.array([-1000, -1000])
maxpos = np.array([3000, 1000])
posdim = len(minpos)
statedim = 2 * posdim
position_mapping = (0, 2)
velocity_mapping = (1, 3)

# Specify sensor noise covariance and measurement times
noise_covar = CovarianceMatrix(np.diag([0.01 ** 2, 10 ** 2]))
start_time = datetime(year=1970, month=1, day=1)#datetime.now().replace(microsecond=0)#
timestep_size = timedelta(seconds=1.0)
simulation_length = timedelta(seconds=300)
number_of_steps = int(simulation_length / timestep_size) + 1
prob_detect = 0.9  # Probability of Detection

# Specify initial velocity mean and covariance
init_velocity_mean = StateVector([[0.0], [0.0]])
init_velocity_covariance = CovarianceMatrix(np.diag([10.0 ** 2, 10.0 ** 2]))

# Get initial state distribution by fitting a Gaussian to the uniform position distribution and merging with
# the velocity distribution
init_position_mean, init_position_covariance = fit_normal_to_uniform(minpos, maxpos)
s_prior_state_mean = merge_position_and_velocity(init_position_mean, init_velocity_mean, statedim,
                                                 position_mapping, velocity_mapping)
s_prior_state_covariance = merge_position_and_velocity_covariance(init_position_covariance, init_velocity_covariance,
                                                                  statedim, position_mapping, velocity_mapping)
# Specify prior Gaussian distribution of a new track
s_prior_state = GaussianState(s_prior_state_mean, s_prior_state_covariance, start_time)

# 2-d Constant Velocity model for target
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.1)] * 2)

# Generate the target initial states
number_of_targets = 20
init_target_states = []
for _ in range(number_of_targets):
    init_pos = np.random.uniform(minpos, maxpos)
    init_vel = StateVector(np.random.multivariate_normal(init_velocity_mean.flatten(), init_velocity_covariance))
    init_pos_vel = merge_position_and_velocity(init_pos, init_vel, statedim, position_mapping, velocity_mapping)
    init_target_states.append(State(init_pos_vel, start_time))

# Ground truth simulator
ground_truth_simulator = MultiTargetGroundTruthSimulator(
    transition_model=transition_model,
    initial_state=s_prior_state,
    birth_rate=0.0,
    death_probability=0.0,
    preexisting_states = [x.state_vector for x in init_target_states],
    timestep=timestep_size,
    number_steps=number_of_steps)

# Generate grid of sensors which tile a box
numx, numy = 4, 2
[platform_positions, max_range] = tile_with_circles(minpos, maxpos, numx, numy)
# Size of tracker grid for each level in the hierarchy
numxy_hierarchy = [(numx, numy), (numx, numy//2), (numx//2, numy//2), (1,1)]

# Create sensor platforms
# (how do I add clutter?)
platforms = []
for i_platform, x in enumerate(platform_positions):

    sensor = RadarBearingRange(ndim_state=statedim, noise_covar=noise_covar,
                               position_mapping=position_mapping, max_range=max_range)

    # Create fixed sensor platform (we need to have 4-dimensional states for the sensors even if they are stationary
    # because the other sensors try to measure them as if they were targets)
    platform_state = merge_position_and_velocity(x, [0, 0], statedim,
                                                 position_mapping, velocity_mapping)
    platform = FixedPlatform(State(platform_state, timestamp=start_time), position_mapping=position_mapping)
    platform.add_sensor(sensor)
    platforms.append(platform)

# Create hierarchy of fusion trackers (all the same for now, with different fusion times)
fusion_times = [timestep_size, 4*timestep_size, 24*timestep_size, 48*timestep_size]
tracker_hierarchy_indices = [[(i,) for i, _ in enumerate(platforms)], [(0,1),(2,3),(4,5),(6,7)], [(0,1),(2,3)], [(0,1)]]
gnd_sims = tee(ground_truth_simulator, len(tracker_hierarchy_indices[0]))
fusion_hierarchy = create_fusion_hierarchy(platforms, gnd_sims, tracker_hierarchy_indices, fusion_times,
                                           transition_model, s_prior_state)

# Run the trackers:

all_tracks = [[set() for _ in level] for level in fusion_hierarchy]
all_detections = [[set() for _ in level] for level in fusion_hierarchy]
root_node = fusion_hierarchy[-1][0]
leaf_trackers = root_node.get_leaf_trackers()

for leaf_time_and_tracks in zip(*leaf_trackers):

    timestamp = leaf_time_and_tracks[0][0]
    leaf_tracks = [x[1] for x in leaf_time_and_tracks]

    print("Time: " + str(timestamp))

    # Run fusion level trackers
    for i_level, level in enumerate(fusion_hierarchy[1:]):
        for i_node, node in enumerate(level):
            node.process_tracks(timestamp)

    # Get tracks and detections from fusion hierarchy
    for i_level, level in enumerate(fusion_hierarchy):
        for i_node, node in enumerate(level):
            all_tracks[i_level][i_node].update(node.tracker.tracks)
            all_detections[i_level][i_node].update(node.detections)

    ntracks_hierarchy = [[len(node.tracks) for node in level] for level in fusion_hierarchy]
    print(ntracks_hierarchy)

truth = ground_truth_simulator.groundtruth_paths

outdir = "TestData/Large/"

output_meas(outdir + "outputfile.txt", start_time, platform_positions, all_detections)
output_tracks(outdir + "outputtracks.txt", start_time, all_tracks)
output_truth(outdir + "outputtruth.txt", start_time, truth)

# Plot results:
for i in range(len(all_tracks)):
    # (don't plot the detections because they will have different dimensions)
    plot_tracks(all_tracks[i], None, truth, numxy_hierarchy[i], fusion_hierarchy[i])
    fig = plt.gcf()
    fig.suptitle(f'Level {i+1}')
    #plot_tracks(all_tracks[i], all_detections[i], truth, numxy_hierarchy[i], fusion_hierarchy[i])

plt.show()
