from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from ordered_set import OrderedSet

from shapely.geometry import Point
from shapely.ops import unary_union

from stonesoup.custom.sensor.moveable import MovableUAVCamera
from stonesoup.sensormanager import BruteForceSensorManager
from stonesoup.sensormanager.reward import UncertaintyRewardFunction
from stonesoup.types.angle import Angle
from stonesoup.types.array import StateVector
from stonesoup.types.numeric import Probability
from stonesoup.types.state import GaussianState, ParticleState
from stonesoup.custom.tracker import SMCPHD_JIPDA
from matplotlib.path import Path

from datetime import datetime
from datetime import timedelta
import numpy as np

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.updater.kalman import KalmanUpdater


def plot_cov_ellipse(cov, pos, nstd=1, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta,
                    alpha=0.4, **kwargs)

    ax.add_artist(ellip)
    return ellip

# np.random.seed(1991)

def _prob_detect_func(fovs):
    """Closure to return the probability of detection function for a given environment scan"""

    # Get the union of all field of views
    fovs_union = unary_union(fovs)
    if fovs_union.geom_type == 'MultiPolygon':
        fovs = [poly for poly in fovs_union]
    else:
        fovs = [fovs_union]

    # Probability of detection nested function
    def prob_detect_func(state):
        for poly in fovs:
            if isinstance(state, ParticleState):
                prob_detect_arr = np.full((len(state),), Probability(0))
                path_p = Path(poly.boundary)
                points = state.state_vector[[0, 2], :].T
                inside_points = path_p.contains_points(points)
                prob_detect_arr[inside_points] = prob_detect
                return prob_detect_arr
            else:
                point = Point(state.state_vector[0, 0], state.state_vector[2, 0])
                return prob_detect if poly.contains(point) else Probability(0)

    return prob_detect_func

# Parameters
# ==========
start_time = datetime.now()         # Simulation start time
prob_detect = Probability(.9)       # 90% chance of detection.
prob_death = Probability(0.01)      # Probability of death
prob_birth = Probability(0.1)       # Probability of birth
prob_survive = Probability(0.99)    # Probability of survival
birth_rate = 0.02                   # Birth-rate (Mean number of new targets per scan)
clutter_rate = 2                    # Clutter-rate (Mean number of clutter measurements per scan)
surveillance_region = [[-10, 30], [0, 30]]  # The surveillance region x=[-10, 30], y=[0, 30]
surveillance_area = (surveillance_region[0][1] - surveillance_region[0][0]) \
                    * (surveillance_region[1][1] - surveillance_region[1][0])
clutter_intensity = clutter_rate / surveillance_area  # Clutter intensity per unit volume/area
birth_density = GaussianState(StateVector(np.array([10., 0.0, 10., 0.0, 0.0, 0.0])),
                              np.diag([10. ** 2, 1. ** 2, 10. ** 2, 1. ** 2, .0, .0]))  # Birth density
birth_scheme = 'mixture'  # Birth scheme. Possible values are 'expansion' and 'mixture'
num_particles = 2 ** 13  # Number of particles used by the PHD filter
num_iter = 100  # Number of simulation steps
total_no_sensors = 1
PLOT = True  # Set [True | False] to turn plotting [ON | OFF]

# Models
# ======
# Transition model
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.01),
                                                          ConstantVelocity(0.01),
                                                          ConstantVelocity(0.01)])

# Simulate Groundtruth
# ====================
gnd_transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.),
                                                              ConstantVelocity(0.),
                                                              ConstantVelocity(0.)])
truths = set()
truth = GroundTruthPath([GroundTruthState([0, 0.2, 0, 0.2, 0, 0], timestamp=start_time)])
for k in range(1, num_iter + 1):
    truth.append(GroundTruthState(
        gnd_transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time + timedelta(seconds=k)))
truths.add(truth)

truth = GroundTruthPath([GroundTruthState([0, 0.2, 20, -0.2, 0, 0], timestamp=start_time)])
for k in range(1, num_iter + 1):
    truth.append(GroundTruthState(
        gnd_transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time + timedelta(seconds=k)))
truths.add(truth)

timestamps = []
for k in range(1, num_iter + 1):
    timestamps.append(start_time + timedelta(seconds=k))


# Create sensors
# ==============
sensors = set()
for i in range(0, total_no_sensors):
    rotation_offset = StateVector(
        [Angle(0), Angle(-np.pi / 2), Angle(0)])  # Camera rotation offset
    pan_tilt = StateVector([Angle(0), Angle(-np.pi / 32)])  # Camera pan and tilt

    position = StateVector([i * 10., 10., 100.])
    resolutions = {'location_x': 5., 'location_y': 5.}
    limits = {'location_x': surveillance_region[0], 'location_y': surveillance_region[1]}
    sensor = MovableUAVCamera(ndim_state=6, mapping=[0, 2, 4],
                              noise_covar=np.diag([0.05, 0.05, 0.05]),
                              location_x=position[0], location_y=position[1],
                              resolutions=resolutions,
                              position=position,
                              fov_radius=10,
                              limits=limits)
    sensors.add(sensor)
for sensor in sensors:
    sensor.timestamp = start_time

# # Predictor & Updater
# # ===================
# predictor = KalmanPredictor(transition_model)
# updater = KalmanUpdater(None)
#
# # Hypothesiser & Data Associator
# # ==============================
# hypothesiser = IPDAHypothesiser(predictor, updater, clutter_intensity, prob_detect=prob_detect,
#                                 prob_survive=prob_survive)
# # hypothesiser = PDAHypothesiser(predictor, updater, clutter_intensity, prob_detect=prob_detect)
# hypothesiser = DistanceGater(hypothesiser, Mahalanobis(), 10)
# associator = JIPDAWithEHM2(hypothesiser)
#
# # Track Deleter
# # =============
# deleter = UpdateTimeDeleter(time_since_update=timedelta(minutes=5))
#
# # Initiator
# # =========
# # Initialise PHD Filter
# resampler = SystematicResampler()
# phd_filter = SMCPHDFilter(birth_density=birth_density, transition_model=transition_model,
#                           measurement_model=None, prob_detect=prob_detect,
#                           prob_death=prob_death, prob_birth=prob_birth,
#                           birth_rate=birth_rate, clutter_intensity=clutter_intensity,
#                           num_samples=num_particles, resampler=resampler,
#                           birth_scheme=birth_scheme)
#
# # Sample prior state from birth density
# state_vector = StateVectors(multivariate_normal.rvs(birth_density.state_vector.ravel(),
#                                                     birth_density.covar,
#                                                     size=num_particles).T)
# weight = np.ones((num_particles,)) * Probability(1 / num_particles)
# state = ParticleState(state_vector=state_vector, weight=weight, timestamp=start_time)
#
#
# initiator = SMCPHDInitiator(filter=phd_filter, prior=state)

tracker = SMCPHD_JIPDA(birth_density=birth_density, transition_model=transition_model,
                       measurement_model=None, prob_detection=prob_detect,
                       prob_death=prob_death, prob_birth=prob_birth,
                       birth_rate=birth_rate, clutter_intensity=clutter_intensity,
                       num_samples=num_particles, birth_scheme=birth_scheme,
                       start_time=start_time)

# Initialise sensor manager
# =========================
reward_function = UncertaintyRewardFunction(tracker._predictor, tracker._updater)
sensor_manager = BruteForceSensorManager(sensors, reward_function)

# Estimate
# ========

# Plot the prior
if PLOT:
    fig1 = plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    # ax1.plot(state.state_vector[0, :], state.state_vector[2, :], 'r.')

# Main tracking loop
tracks = set()
for k, timestamp in enumerate(timestamps):

    tracks = list(tracks)

    # Generate chosen configuration
    chosen_actions = sensor_manager.choose_actions(tracks, timestamp)

    # Create empty dictionary for measurements
    detections = []

    for chosen_action in chosen_actions:
        for sensor, actions in chosen_action.items():
            sensor.add_actions(actions)

    fovs = []
    truth_states = OrderedSet(truth[timestamp] for truth in truths)
    for sensor in sensors:
        sensor.act(timestamp)
        center = (sensor.position[0], sensor.position[1])
        radius = sensor.fov_radius
        p = Point(center).buffer(radius)
        fovs.append(p)

    tracker.prob_detect = _prob_detect_func(fovs)

    for sensor in sensors:

        # Observe this ground truth
        sensor_measurements = sensor.measure(truth_states, noise=True)
        detections.extend(sensor_measurements)

        detections = list(detections)
        num_tracks = len(tracks)
        num_detections = len(detections)

        tracks = tracker.track(detections, timestamp)

        print('\n===========================================')
        # print(f'Num targets: {np.sum(state.weight)} - Num new targets: {len(new_tracks)}')
        for track in tracks:
            print(f'Track {track.id} - Exist prob: {track.exist_prob}')

        # Plot resulting density
        if PLOT:
            ax1.cla()

            circle = plt.Circle((sensor.position[0], sensor.position[1]), radius=sensor.fov_radius,
                                color='r',
                                fill=False)
            ax1.add_artist(circle)
            for i, truth in enumerate(truths):
                data = np.array([s.state_vector for s in truth[:k + 1]])
                ax1.plot(data[:, 0], data[:, 2], '--', label=f'Groundtruth Track {i+1}')
            if len(detections):
                det_data = np.array([det.state_vector for det in detections])
                ax1.plot(det_data[:, 0], det_data[:, 1], '*g', label='Detections')
            # ax1.plot(state.state_vector[0, :], state.state_vector[2, :],
            #          'r.', label='Particles')

            for track in tracks:
                data = np.array([s.state_vector for s in track])
                ax1.plot(data[:, 0], data[:, 2], label=f'Track {track.id}')
                plot_cov_ellipse(track.covar[[0, 2], :][:, [0, 2]], track.state_vector[[0, 2], :],
                                 edgecolor='r', facecolor='none', ax=ax1)
            plt.axis([*surveillance_region[0], *surveillance_region[1]])
            plt.legend(loc='upper right')
            plt.pause(0.01)
