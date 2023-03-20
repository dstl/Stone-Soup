import uuid
from datetime import datetime, timedelta
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from ordered_set import OrderedSet
from shapely.geometry import Point, Polygon

from stonesoup.custom.functions import calculate_num_targets_dist, geodesic_point_buffer
from stonesoup.custom.sensor.movable import MovableUAVCamera
from stonesoup.custom.sensormanager.base import UniqueBruteForceSensorManager
from stonesoup.custom.sensormanager.reward import RolloutPriorityRewardFunction, \
    RolloutPriorityRewardFunction2
from stonesoup.types.angle import Angle
from stonesoup.types.array import StateVector
from stonesoup.types.numeric import Probability
from stonesoup.types.state import GaussianState, ParticleState
from stonesoup.custom.tracker import SMCPHD_JIPDA, SMCPHD_IGNN
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from reactive_isr_core.data import RFI, TaskType, GeoRegion, GeoLocation, PriorityOverTime, \
    ThresholdOverTime, TargetSpecification, TargetType

from utils import plot_cov_ellipse, _prob_detect_func

# np.random.seed(5547)
np.random.seed(95146)

# Parameters
# ==========
start_time = datetime.now()         # Simulation start time
prob_detect = Probability(.9)       # 90% chance of detection.
prob_death = Probability(0.01)      # Probability of death
prob_birth = Probability(0.1)       # Probability of birth
prob_survive = Probability(0.99)    # Probability of survival
birth_rate = 0.02                   # Birth-rate (Mean number of new targets per scan)
clutter_rate = 2                    # Clutter-rate (Mean number of clutter measurements per scan)
surveillance_region = [[-5, -2],    # The surveillance region
                       [50.1, 53.2]]
surveillance_area = (surveillance_region[0][1] - surveillance_region[0][0]) \
                    * (surveillance_region[1][1] - surveillance_region[1][0])  # Surveillance volume
clutter_intensity = clutter_rate / surveillance_area  # Clutter intensity per unit volume/area
birth_density = GaussianState(
    StateVector(np.array([-2.5, 0.0, 51, 0.0, 0.0, 0.0])),
    np.diag([3. ** 2, .01 ** 2, 3. ** 2, .01 ** 2, 0., 0.]))  # Birth density
birth_scheme = 'mixture'            # Birth scheme. Possible values are 'expansion' and 'mixture'
num_particles = 2 ** 8             # Number of particles used by the PHD filter
num_iter = 200                      # Number of simulation steps
total_no_sensors = 3                # Total number of sensors
PLOT = True                         # Set [True | False] to turn plotting [ON | OFF]
MANUAL_RFI = True                  # Set [True | False] to turn manual RFI [ON | OFF]
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Colors for plotting

# Simulate Groundtruth
# ====================
gnd_transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.),
                                                              ConstantVelocity(0.),
                                                              ConstantVelocity(0.)])
timestamps = []
for k in range(0, num_iter + 1, 2):
    timestamps.append(start_time + timedelta(seconds=k))

truths = set()
truth = GroundTruthPath([GroundTruthState([-3.7, 0.0, 52.0, 0.01, 0, 0], timestamp=start_time)])
for timestamp in timestamps[1:]:
    truth.append(GroundTruthState(
        gnd_transition_model.function(truth[-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=timestamp))
truths.add(truth)

truth = GroundTruthPath([GroundTruthState([-4.6, 0.01, 52.1, -0.01, 0, 0], timestamp=start_time)])
for timestamp in timestamps[1:]:
    truth.append(GroundTruthState(
        gnd_transition_model.function(truth[-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=timestamp))
truths.add(truth)

truth = GroundTruthPath([GroundTruthState([-3.5, 0, 51.3, -0.01, 0, 0], timestamp=start_time)])
for timestamp in timestamps[1:]:
    truth.append(GroundTruthState(
        gnd_transition_model.function(truth[-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=timestamp))
truths.add(truth)

# Create sensors
# ==============
sensors = []
for i in range(0, total_no_sensors):
    rotation_offset = StateVector(
        [Angle(0), Angle(-np.pi / 2), Angle(0)])  # Camera rotation offset
    pan_tilt = StateVector([Angle(0), Angle(-np.pi / 32)])  # Camera pan and tilt
    increment = 1.0*i
    x = -4.5 + increment
    y = 51.5 if i == 0 else 51.5 + increment
    position = StateVector([-4.5+increment, 51.5, 100.])
    resolutions = {'location_x': 1, 'location_y': 1}
    limits = {'location_x': [surveillance_region[0][0]+0.5, surveillance_region[0][1]-0.5],
              'location_y': [round(surveillance_region[1][0])+0.5, round(surveillance_region[1][1])-0.5]}
    sensor = MovableUAVCamera(ndim_state=6, mapping=[0, 2, 4],
                              noise_covar=np.diag([0.0001, 0.0001, 0.0001]),
                              location_x=position[0], location_y=position[1],
                              resolutions=resolutions,
                              position=position,
                              fov_radius=70,
                              limits=limits)
    sensors.append(sensor)
for sensor in sensors:
    sensor.timestamp = start_time

# Plot groundtruth and sensors
# ============================
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_xlim(surveillance_region[0][0]-1, surveillance_region[0][1]+1)
ax.set_ylim(surveillance_region[1][0], surveillance_region[1][1])
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Groundtruth and initial sensor locations')
ax.grid(True)
ax.set_aspect('equal')
for i, track in enumerate(truths):
    ax.plot([state.state_vector[0] for state in track],
            [state.state_vector[2] for state in track],
            color=colors[i], linestyle='--', linewidth=2, label=f'Truth {i+1}')
for j, sensor in enumerate(sensors):
    coords = geodesic_point_buffer(sensor.position[1], sensor.position[0], sensor.fov_radius).exterior.coords[:]
    ax.plot([coord[0] for coord in coords], [coord[1] for coord in coords],
            color=colors[j], linewidth=2, label=f'Sensor {j+1} FOV')
    # circle = plt.Circle((sensor.position[0], sensor.position[1]), radius=sensor.fov_radius,
    #                     color=colors[j],
    #                     fill=False,
    #                     label=f'Sensor {j + 1}')
    # ax.add_artist(circle, )
ax.legend()
plt.show()

# Tracking Components
# ===================
# Transition model
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.000001),
                                                          ConstantVelocity(0.000001),
                                                          ConstantVelocity(0.000001)])

# Main tracker
tracker = SMCPHD_IGNN(birth_density=birth_density, transition_model=transition_model,
                       measurement_model=None, prob_detection=prob_detect,
                       prob_death=prob_death, prob_birth=prob_birth,
                       birth_rate=birth_rate, clutter_intensity=clutter_intensity,
                       num_samples=num_particles, birth_scheme=birth_scheme,
                       start_time=start_time)

# Evaluator tracker
tracker2 = SMCPHD_IGNN(birth_density=birth_density, transition_model=transition_model,
                       measurement_model=None, prob_detection=prob_detect,
                       prob_death=prob_death, prob_birth=prob_birth,
                       birth_rate=birth_rate, clutter_intensity=clutter_intensity,
                       num_samples=num_particles, birth_scheme=birth_scheme,
                       start_time=start_time)
# tracker2 = copy.deepcopy(tracker)


# Sensor Management Components
# ============================
# Reward function
roi = GeoRegion(corners=[
    GeoLocation(
        longitude=surveillance_region[0][0],
        latitude=surveillance_region[1][0],
        altitude=0),
    GeoLocation(
        longitude=surveillance_region[0][1],
        latitude=surveillance_region[1][1],
        altitude=0)]
)
rfi = RFI(id=uuid.uuid4(),
          task_type=TaskType.COUNT,
          region_of_interest=roi,
          start_time=datetime.now(),
          end_time=datetime.now(),
          priority_over_time=PriorityOverTime(timescale=[datetime.now()], priority=[5]),
          targets=[], #TargetSpecification(target_type=TargetType.VEHICLE, existence_probability=0.9)
          threshold_over_time=ThresholdOverTime(timescale=[datetime.now()], threshold=[.00001]))
rfis = [rfi] if not MANUAL_RFI else []
reward_function = RolloutPriorityRewardFunction2(tracker2, 0,
                                                num_samples=100, interval=timedelta(seconds=5),
                                                rfis=rfis)
sensor_manager = UniqueBruteForceSensorManager(sensors, reward_function)


# Estimate
# ========
# Plotting setup
if PLOT:
    fig1 = plt.figure(figsize=(20, 7))
    ax1, ax2 = fig1.subplots(1, 2)
    ax1.set_title('Simulation')
    ax2.set_title('Variance')
    fig1.subplots_adjust(bottom=0.2)
    axbtn = fig1.add_axes([0.81, 0.05, 0.15, 0.075])
    btn = Button(axbtn, 'New RFI')
    def set_rfis(*args, **kwargs):
        print("Added RFI")
        reward_function.rfis = [rfi]
    btn.on_clicked(set_rfis)
    plt.ion()

# Main tracking loop
tracks = set()
vars = []
for k, timestamp in enumerate(timestamps):

    if k == 20:
        sensors.pop(1)

    sensor_detections = []
    tracks = list(tracks)
    truth_states = OrderedSet(truth[timestamp] for truth in truths)

    # Compute variance of number of targets
    region_corners = rfi.region_of_interest.corners
    xmin, ymin = region_corners[0].longitude, region_corners[0].latitude
    xmax, ymax = region_corners[1].longitude, region_corners[1].latitude
    geom = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
    _, var = calculate_num_targets_dist(tracks, geom)
    vars.append(var)

    # Check if RFI is satisfied and remove it
    if MANUAL_RFI and len(reward_function.rfis) > 0:
        if var < rfi.threshold_over_time.threshold[0]:
            reward_function.rfis.remove(rfi)

    # Generate chosen configuration
    chosen_actions = sensor_manager.choose_actions(tracks, timestamp)
    for chosen_action in chosen_actions:
        for sensor, actions in chosen_action.items():
            sensor.add_actions(actions)

    # Cue sensors
    for sensor in sensors:
        sensor.act(timestamp)

    # For each sensor
    for j, sensor in enumerate(sensors):

        # Compute probability of detection
        # center = (sensor.position[0], sensor.position[1])
        # radius = sensor.fov_radius
        # p = Point(center).buffer(radius)
        p = geodesic_point_buffer(sensor.position[1], sensor.position[0], sensor.fov_radius)
        tracker.prob_detect = _prob_detect_func(prob_detect, [p])

        # Observe the ground truth
        detections = sensor.measure(truth_states, noise=True)
        for detection in detections:
            detection.metadata['target_type_confidences'] = {
                'person': 1.0
            }
        sensor_detections.append(detections)

        detections = list(detections)
        num_tracks = len(tracks)
        num_detections = len(detections)

        # Track using main tracker
        tracks = tracker.track(detections, timestamp)

        # Print debug info
        tracks = list(tracks)
        print(f'\n Sensor {j+1} ===========================================')
        for track in tracks:
            print(f'Track {track.id} - Exist prob: {track.exist_prob}')

    # Plot output
    if PLOT:
        ax1.cla()
        ax2.cla()
        ax2.plot([i for i in range(k+1)], vars, 'r')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Variance')
        ax1.set_title('Simulation')
        ax2.set_title('Variance')
        for j, sensor in enumerate(sensors):
            coords = geodesic_point_buffer(sensor.position[1], sensor.position[0],
                                           sensor.fov_radius).exterior.coords[:]
            ax1.plot([coord[0] for coord in coords], [coord[1] for coord in coords],
                    color=colors[j], linewidth=2, label=f'Sensor {j + 1} FOV')
            # circle = plt.Circle((sensor.position[0], sensor.position[1]), radius=sensor.fov_radius,
            #                     color=colors[j],
            #                     fill=False,
            #                     label=f'Sensor {j+1}')
            # ax1.add_artist(circle, )
            detections = sensor_detections[j]
            if len(detections):
                det_data = np.array([det.state_vector for det in detections])
                ax1.plot(det_data[:, 0], det_data[:, 1], f'*{colors[j]}', label='Detections')

        for i, truth in enumerate(truths):
            data = np.array([s.state_vector for s in truth[:k + 1]])
            ax1.plot(data[:, 0], data[:, 2], '--', label=f'Groundtruth Track {i+1}')

        for i, track in enumerate(tracks):
            data = np.array([s.state_vector for s in track])
            ax1.plot(data[:, 0], data[:, 2], label=f'Track {i}')
            plot_cov_ellipse(track.covar[[0, 2], :][:, [0, 2]], track.state_vector[[0, 2], :],
                             edgecolor='r', facecolor='none', ax=ax1)
        ax1.set_aspect('equal', adjustable='box', anchor='C')
        ax1.set_xlim(np.array(surveillance_region[0]) + np.array([-1, 1]))
        ax1.set_ylim(surveillance_region[1])
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.legend(loc='upper right')
        plt.pause(0.1)

