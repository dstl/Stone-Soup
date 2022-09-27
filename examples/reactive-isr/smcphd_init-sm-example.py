from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from ordered_set import OrderedSet

from stonesoup.custom.functions import get_camera_footprint
from stonesoup.custom.jipda import JIPDAWithEHM2
from stonesoup.custom.sensor.pan_tilt import PanTiltUAVCamera
from stonesoup.deleter.time import UpdateTimeDeleter
from stonesoup.functions import gm_reduce_single
from stonesoup.gater.distance import DistanceGater
from stonesoup.hypothesiser.probability import IPDAHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.resampler.particle import SystematicResampler
from stonesoup.sensormanager import BruteForceSensorManager
from stonesoup.sensormanager.reward import UncertaintyRewardFunction
from stonesoup.types.angle import Angle
from stonesoup.types.array import StateVector, StateVectors
from stonesoup.types.numeric import Probability
from stonesoup.types.state import GaussianState, ParticleState
from stonesoup.custom.smcphd import SMCPHDFilter, SMCPHDInitiator
from stonesoup.custom.tracker import SMCPHD_JIPDA

from datetime import datetime
from datetime import timedelta
import numpy as np
from scipy.stats import uniform, multivariate_normal

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
for n in range(0, total_no_sensors):
    rotation_offset = StateVector(
        [Angle(0), Angle(-np.pi / 2), Angle(0)])  # Camera rotation offset
    pan_tilt = StateVector([Angle(0), Angle(-np.pi / 32)])  # Camera pan and tilt

    sensor = PanTiltUAVCamera(ndim_state=6, mapping=[0, 2, 4],
                              noise_covar=np.diag([0.05, 0.05, 0.05]),
                              fov_angle=[np.radians(15), np.radians(10)],
                              rotation_offset=rotation_offset,
                              pan=pan_tilt[0], tilt=pan_tilt[1],
                              position=StateVector([10., 10., 100.]))
    sensors.add(sensor)
for sensor in sensors:
    sensor.timestamp = start_time

# Predictor & Updater
# ===================
predictor = KalmanPredictor(transition_model)
updater = KalmanUpdater(None)

# Hypothesiser & Data Associator
# ==============================
hypothesiser = IPDAHypothesiser(predictor, updater, clutter_intensity, prob_detect=prob_detect,
                                prob_survive=prob_survive)
# hypothesiser = PDAHypothesiser(predictor, updater, clutter_intensity, prob_detect=prob_detect)
hypothesiser = DistanceGater(hypothesiser, Mahalanobis(), 10)
associator = JIPDAWithEHM2(hypothesiser)

# Track Deleter
# =============
deleter = UpdateTimeDeleter(time_since_update=timedelta(minutes=5))

# Initiator
# =========
# Initialise PHD Filter
resampler = SystematicResampler()
phd_filter = SMCPHDFilter(birth_density=birth_density, transition_model=transition_model,
                          measurement_model=None, prob_detect=prob_detect,
                          prob_death=prob_death, prob_birth=prob_birth,
                          birth_rate=birth_rate, clutter_intensity=clutter_intensity,
                          num_samples=num_particles, resampler=resampler,
                          birth_scheme=birth_scheme)

# Sample prior state from birth density
state_vector = StateVectors(multivariate_normal.rvs(birth_density.state_vector.ravel(),
                                                    birth_density.covar,
                                                    size=num_particles).T)
weight = np.ones((num_particles,)) * Probability(1 / num_particles)
state = ParticleState(state_vector=state_vector, weight=weight, timestamp=start_time)


initiator = SMCPHDInitiator(filter=phd_filter, prior=state)

tracker = SMCPHD_JIPDA(birth_density=birth_density, transition_model=transition_model,
                       measurement_model=None, prob_detect=prob_detect,
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

    # Fov ranges (min, center, max)
    foot1 = get_camera_footprint(sensor)

    truth_states = OrderedSet(truth[timestamp] for truth in truths)
    for sensor in sensors:
        sensor.act(timestamp)

        # Observe this ground truth
        sensor_measurements = sensor.measure(truth_states, noise=True)
        detections.extend(sensor_measurements)

    foot2 = get_camera_footprint(sensor)
    
    # tracks = tracker.track(detections, timestamp)
    detections = list(detections)
    num_tracks = len(tracks)
    num_detections = len(detections)

    # Perform data association
    associations = associator.associate(tracks, detections, timestamp)

    assoc_prob_matrix = np.zeros((num_tracks, num_detections + 1))
    for i, track in enumerate(tracks):
        for hyp in associations[track]:
            if not hyp:
                assoc_prob_matrix[i, 0] = hyp.weight
            else:
                j = next(d_i for d_i, detection in enumerate(detections)
                         if hyp.measurement == detection)
                assoc_prob_matrix[i, j + 1] = hyp.weight

    rho = np.zeros((len(detections)))
    for j, detection in enumerate(detections):
        rho_tmp = 1
        if len(assoc_prob_matrix):
            for i, track in enumerate(tracks):
                rho_tmp *= 1 - assoc_prob_matrix[i, j + 1]
        rho[j] = rho_tmp

    for track, multihypothesis in associations.items():

        # calculate each Track's state as a Gaussian Mixture of
        # its possible associations with each detection, then
        # reduce the Mixture to a single Gaussian State
        posterior_states = []
        posterior_state_weights = []
        for hypothesis in multihypothesis:
            posterior_state_weights.append(hypothesis.probability)
            if hypothesis:
                posterior_states.append(updater.update(hypothesis))
            else:
                posterior_states.append(hypothesis.prediction)

        # Merge/Collapse to single Gaussian
        means = StateVectors([state.state_vector for state in posterior_states])
        covars = np.stack([state.covar for state in posterior_states], axis=2)
        weights = np.asarray(posterior_state_weights)

        post_mean, post_covar = gm_reduce_single(means, covars, weights)

        track.append(GaussianStateUpdate(
            np.array(post_mean), np.array(post_covar),
            multihypothesis,
            multihypothesis[0].prediction.timestamp))

    tracks = set(tracks)
    new_tracks = initiator.initiate(detections, timestamp, weights=rho)
    tracks |= new_tracks
    state = initiator._state

    # Delete tracks that have not been updated for a while
    del_tracks = set()
    for track in tracks:
        if track.exist_prob < 0.1:
            del_tracks.add(track)
    tracks -= del_tracks

    print('\n===========================================')
    # print(f'Num targets: {np.sum(state.weight)} - Num new targets: {len(new_tracks)}')
    for track in tracks:
        print(f'Track {track.id} - Exist prob: {track.exist_prob}')

    # Plot resulting density
    if PLOT:
        ax1.cla()
        xmin, xmax, ymin, ymax = foot1
        ax1.add_patch(
            Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor='none', edgecolor='b'))
        xmin, xmax, ymin, ymax = foot2
        ax1.add_patch(
            Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor='none', edgecolor='r'))
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
