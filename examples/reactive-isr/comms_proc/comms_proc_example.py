from copy import copy, deepcopy
from datetime import datetime, timedelta
from uuid import uuid4

import warnings

from matplotlib import pyplot as plt
from ordered_set import OrderedSet

from stonesoup.custom.functions.rollout import enumerate_action_configs, extract_rois, get_sensor, \
    queue_actions, ActionTupleType
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import numpy as np
from matplotlib.path import Path
from shapely import unary_union

from reactive_isr_core.data import Node, AvailableAlgorithms, Algorithm, ProcessingStatistics, \
    TraversalTime, Edge, NetworkTopology, Availability, Storage, ImageStore, ProcessingAction, \
    ActionList, CommunicateAction, CollectAction, ActionStatus, Image, GeoLocation
from stonesoup.custom.sensor.movable import MovableUAVCamera
from stonesoup.custom.tracker import SMCPHD_JIPDA, SMCPHD_IGNN
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.array import StateVector
from stonesoup.types.detection import TrueDetection
from stonesoup.types.numeric import Probability
from stonesoup.types.state import GaussianState, ParticleState

from evaluator import CommsAndProcEvaluator
from utils import setup_network, setup_rfis, plot_cov_ellipse


def _prob_detect_func(fovs, prob_detect):
    """Closure to return the probability of detection function for a given environment scan"""
    # Get the union of all field of views
    fovs_union = unary_union(fovs)
    if fovs_union.geom_type == 'MultiPolygon':
        fovs = [poly for poly in fovs_union]
    else:
        fovs = [fovs_union]

    paths = [Path(poly.boundary.coords) for poly in fovs]

    # Probability of detection nested function
    def prob_detect_func(state):
        for path_p in paths:
            if isinstance(state, ParticleState):
                prob_detect_arr = np.full((len(state),), Probability(0.1))
                points = state.state_vector[[0, 2], :].T
                inside_points = path_p.contains_points(points)
                prob_detect_arr[inside_points] = prob_detect
                return prob_detect_arr
            else:
                points = state.state_vector[[0, 2], :].T
                return prob_detect if np.alltrue(path_p.contains_points(points)) \
                    else Probability(0)

    return prob_detect_func

seed = 2001
np.random.seed(seed)

# Parameters
# ==========
start_time = datetime.now()  # Simulation start time
prob_detect = Probability(.9)  # 90% chance of detection.
prob_death = Probability(0.01)  # Probability of death
prob_birth = Probability(0.1)  # Probability of birth
prob_survive = Probability(0.99)  # Probability of survival
birth_rate = 0.02  # Birth-rate (Mean number of new targets per scan)
clutter_rate = 10  # Clutter-rate (Mean number of clutter measurements per scan)
surveillance_region = [[-5, -2],  # The surveillance region
                       [50.1, 53.2]]
surveillance_area = (surveillance_region[0][1] - surveillance_region[0][0]) \
                    * (surveillance_region[1][1] - surveillance_region[1][0]) # Surveillance volume
clutter_intensity = clutter_rate / surveillance_area  # Clutter intensity per unit volume/area
birth_density = GaussianState(
    StateVector(np.array([-2.5, 0.0, 51, 0.0, 0.0, 0.0])),
    np.diag([3. ** 2, .01 ** 2, 3. ** 2, .01 ** 2, 0., 0.]))  # Birth density
birth_scheme = 'mixture'  # Birth scheme. Possible values are 'expansion' and 'mixture'
num_particles = 2 ** 8  # Number of particles used by the PHD filter
num_iter = 400  # Number of simulation steps
PLOT = True  # Set [True | False] to turn plotting [ON | OFF]
MANUAL_RFI = True  # Set [True | False] to turn manual RFI [ON | OFF]
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Colors for plotting

sensor_position = StateVector([-4.5, 51.5, 100.])
network_topology, assets = setup_network(sensor_position, start_time)
image_store = ImageStore(
    images=[]
)

# Ongoing actions is a dictionary of lists of actions. The keys are the action types and the values
# are the lists of actions of that type. The action types are 'collect', 'comms' and 'proc'.
# This dictionary is used to keep track of ongoing actions.
ongoing_actions = {
    'collect': [],
    'comms': [],
    'proc': [],
}

rfis = setup_rfis(start_time, num_rois=2, time_varying=True)

# Simulate Groundtruth
# ====================
gnd_transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.),
                                                              ConstantVelocity(0.),
                                                              ConstantVelocity(0.)])

timestamps = []
for k in range(0, num_iter + 1, 2):
    timestamps.append(start_time + timedelta(seconds=k))
truths = set()
rois = extract_rois(rfis)
for roi in rois:
    lon_min, lat_min = roi.corners[0].longitude, roi.corners[0].latitude
    lon_max, lat_max = roi.corners[1].longitude, roi.corners[1].latitude
    for i in range(2):
        lat = np.random.uniform(lat_min, lat_max)
        lon = np.random.uniform(lon_min, lon_max)
        truth = GroundTruthPath([GroundTruthState([lon, 0.00, lat, 0.00, 0, 0],
                                                  timestamp=start_time)])
        for timestamp in timestamps[1:]:
            truth.append(GroundTruthState(
                gnd_transition_model.function(truth[-1], noise=False,
                                              time_interval=timedelta(seconds=1)),
                timestamp=timestamp))
        truths.add(truth)

# Plot groundtruth, sensors and rois
# ============================
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111)
# ax.set_xlim(surveillance_region[0][0]-1, surveillance_region[0][1]+1)
# ax.set_ylim(surveillance_region[1][0], surveillance_region[1][1])
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
# ax.set_title('Groundtruth and initial sensor locations')
# ax.set_aspect('equal')
# for i, track in enumerate(truths):
#     ax.plot([state.state_vector[0] for state in track],
#             [state.state_vector[2] for state in track],
#             color=colors[i], linestyle='--', linewidth=2, label=f'Truth {i+1}')
#     ax.plot(track[-1].state_vector[0], track[-1].state_vector[2],
#             color=colors[i], marker='o', markersize=5)
# asset = assets.assets[0]
# sensor = get_sensor(asset.asset_status.location, asset.asset_description.fov_radius)
# footprint = sensor.footprint
# x, y = footprint.exterior.xy
# ax.plot(x, y, color='r', label=f'Sensor')
# for i, roi in enumerate(rois):
#     lon_min, lat_min = roi.corners[0].longitude, roi.corners[0].latitude
#     lon_max, lat_max = roi.corners[1].longitude, roi.corners[1].latitude
#     ax.plot([lon_min, lon_max, lon_max, lon_min, lon_min],
#             [lat_min, lat_min, lat_max, lat_max, lat_min],
#             color='k', linestyle='--', linewidth=0.1, label=f'ROI {i+1}')
# ax.legend()
# plt.show()

# Tracking Components
# ===================
# Transition model
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.000001),
                                                          ConstantVelocity(0.000001),
                                                          ConstantVelocity(0.000001)])

# Main tracker
tracker = SMCPHD_JIPDA(birth_density=birth_density, transition_model=transition_model,
                       measurement_model=None, prob_detection=prob_detect,
                       prob_death=prob_death, prob_birth=prob_birth,
                       birth_rate=birth_rate, clutter_intensity=clutter_intensity,
                       num_samples=num_particles, birth_scheme=birth_scheme,
                       start_time=start_time)

# Evaluator tracker
eval_tracker = SMCPHD_IGNN(birth_density=birth_density, transition_model=transition_model,
                           measurement_model=None, prob_detection=prob_detect,
                           prob_death=prob_death, prob_birth=prob_birth,
                           birth_rate=birth_rate, clutter_intensity=clutter_intensity,
                           num_samples=num_particles, birth_scheme=birth_scheme,
                           start_time=start_time)


# Evaluator
num_samples = 40                    # Number of monte-carlo samples for Monte-Carlo Rollout
num_timesteps = 5                   # Number of timesteps for Monte-Carlo Rollout
interval = timedelta(seconds=1)     # Interval between timesteps for Monte-Carlo Rollout
evaluator = CommsAndProcEvaluator(
    tracker=eval_tracker,
    num_timesteps=num_timesteps,
    interval=interval,
    num_samples=num_samples,
)

def optimise(tracks, image_store, network_topology, assets, rfis, ongoing_actions, timestamp):
    # Get all possible action configurations
    configs = enumerate_action_configs(image_store, network_topology, assets, rfis,
                                       ongoing_actions, timestamp)

    # For each action configuration
    rewards = []
    for config in configs:
        # Evaluate the action configuration
        reward = evaluator(config, tracks, image_store, network_topology, assets, rfis,
                           ongoing_actions, timestamp)

        rewards.append(reward)

    print(f'\nRewards: \n--------------------------------------------------------')
    for i, config in enumerate(configs):
        print(f'{rewards[i]:.2f} - {config}')
    # Find the best action configuration
    max_reward = np.max(rewards)
    best_inds = np.argwhere(rewards == max_reward).flatten()
    best_ind = np.random.choice(best_inds)
    best_config = configs[best_ind]
    return best_config


if PLOT:
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

tracks = set()
processed_images = list()
for k, timestamp in enumerate(timestamps):
    print(f'\n\nIter: {k+1} - Timestamp: {timestamp}\n ===========================================')
    truth_states = OrderedSet(truth[timestamp] for truth in truths)

    # Update ongoing actions
    completed_comms_actions = []
    completed_proc_actions = []
    for comms_action in ongoing_actions['comms']:
        if timestamp >= comms_action.end_time:
            completed_comms_actions.append(comms_action)
            comms_action.image.node_id = comms_action.target_node_id
    for proc_action in ongoing_actions['proc']:
        if timestamp >= proc_action.end_time:
           completed_proc_actions.append(proc_action)
    for comms_action in completed_comms_actions:
        ongoing_actions['comms'].remove(comms_action)
    for proc_action in completed_proc_actions:
        ongoing_actions['proc'].remove(proc_action)
        try:
            image_store.images.remove(proc_action.image)
        except ValueError:
            pass

    # Optimise actions
    chosen_actions = optimise(tracks, image_store, network_topology, assets, rfis,
                              ongoing_actions, timestamp)
    print(f'Chosen actions: \n--------------------------------------------------------')
    print(chosen_actions)


    # Perform chosen actions
    queue_actions(chosen_actions, image_store, ongoing_actions)
    sensor_action = chosen_actions[0]
    if sensor_action:
        coll_action = sensor_action[0]
        sensor = get_sensor(coll_action.image.location, coll_action.image.fov_radius)
    proc_actions = [action.proc_action for action in chosen_actions if action]
    proc_actions.sort(key=lambda x: x.image.collection_time)
    for i, proc_action in enumerate(proc_actions):
        if proc_action.image in processed_images:
            continue
        else:
            processed_images.append(proc_action.image)
        sub_sensor = get_sensor(proc_action.image.location, proc_action.image.fov_radius,
                                proc_action.algorithm.prob_detection,
                                proc_action.algorithm.false_alarm_density)
        p = sub_sensor.footprint
        tracker.prob_detect = _prob_detect_func([p], proc_action.algorithm.prob_detection)
        tracker.clutter_intensity = proc_action.algorithm.false_alarm_density/p.area

        # Observe the ground truth
        detections = sensor.measure(truth_states, noise=True)
        detections = list(detections)
        # Track using main tracker
        tracks = tracker.track(detections, proc_action.image.collection_time)

        # Print debug info
        tracks = list(tracks)
        print(f'\n Action {i + 1} --------------------------------------------------------')
        for track in tracks:
            print(f'Track {track.id} - Exist prob: {track.exist_prob}')

    if PLOT:
        ax.cla()
        ax.set_xlim(surveillance_region[0][0] - 1, surveillance_region[0][1] + 1)
        ax.set_ylim(surveillance_region[1][0], surveillance_region[1][1])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Groundtruth and initial sensor locations')
        ax.set_aspect('equal')
        for i, track in enumerate(truths):
            ax.plot([state.state_vector[0] for state in track],
                    [state.state_vector[2] for state in track],
                    color=colors[i], linestyle='--', linewidth=2, label=f'Truth {i + 1}')
            ax.plot(track[-1].state_vector[0], track[-1].state_vector[2],
                    color=colors[i], marker='o', markersize=5)

        for track in tracks:
            plot_cov_ellipse(track.covar[[0, 2], :][:, [0, 2]], track.state_vector[[0, 2], :],
                             edgecolor='r', facecolor='none', ax=ax)
            ax.plot(track.state_vector[0, 0], track.state_vector[2, 0], 'rx', markersize=5)

        footprint = sensor.footprint
        x, y = footprint.exterior.xy
        ax.plot(x, y, color='r', label=f'Sensor')
        for i, roi in enumerate(rois):
            lon_min, lat_min = roi.corners[0].longitude, roi.corners[0].latitude
            lon_max, lat_max = roi.corners[1].longitude, roi.corners[1].latitude
            ax.plot([lon_min, lon_max, lon_max, lon_min, lon_min],
                    [lat_min, lat_min, lat_max, lat_max, lat_min],
                    color='k', linestyle='--', linewidth=0.1, label=f'ROI {i + 1}')
        ax.legend()
        plt.pause(0.1)