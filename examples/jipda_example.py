from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from ordered_set import OrderedSet
from datetime import datetime, timedelta

import numpy as np
from scipy.stats import uniform

from stonesoup.dataassociator.probability import JIPDA
from stonesoup.functions import gm_reduce_single
from stonesoup.gater.distance import DistanceGater
from stonesoup.hypothesiser.probability import IPDAHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.array import StateVectors
from stonesoup.types.numeric import Probability
from stonesoup.types.state import GaussianState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import TrueDetection
from stonesoup.types.detection import Clutter
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.track import Track
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
num_iter = 21  # Number of simulation steps
PLOT = True  # Set [True | False] to turn plotting [ON | OFF]

# Models
# ======
# Transition model
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                          ConstantVelocity(0.005)])
# Measurement model
measurement_model = LinearGaussian(ndim_state=4,
                                   mapping=(0, 2),
                                   noise_covar=np.array([[0.1, 0],
                                                         [0, 0.1]]))

# Simulate Groundtruth
# ====================
np.random.seed(1991)
truths = OrderedSet()
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])
for k in range(1, num_iter):
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time+timedelta(seconds=k)))
truths.add(truth)

truth = GroundTruthPath([GroundTruthState([0, 1, 20, -1], timestamp=start_time)])
for k in range(1, num_iter):
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time+timedelta(seconds=k)))
truths.add(truth)

timestamps = []
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
    for _ in range(np.random.poisson(clutter_rate)):
        x = uniform.rvs(-10, 30)
        y = uniform.rvs(0, 25)
        measurement_set.add(Clutter(np.array([[x], [y]]), timestamp=truth[k].timestamp,
                                    measurement_model=measurement_model))
    scans.append((timestamps[k], measurement_set))

# Predictor & Updater
# ===================
predictor = KalmanPredictor(transition_model)
updater = KalmanUpdater(measurement_model)

# Hypothesiser & Data Associator
# ==============================
hypothesiser = IPDAHypothesiser(predictor, updater, clutter_intensity, prob_detect=prob_detect,
                                prob_survive=prob_survive)
hypothesiser = DistanceGater(hypothesiser, Mahalanobis(), 10)
associator = JIPDA(hypothesiser)

# Initialise the tracks
prior1 = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)
prior2 = GaussianState([[0], [1], [20], [-1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)
tracks = {Track([prior1]), Track([prior2])}
# Set the existence probability of each track to 0.5
for track in tracks:
    track.exist_prob = Probability(0.5)

# Estimate
# ========

# Plot the prior
if PLOT:
    fig1 = plt.figure(figsize=(13, 7))
    ax1 = plt.gca()

# Main tracking loop
for k, (timestamp, detections) in enumerate(scans):

    # Perform data association
    associations = associator.associate(tracks, detections, timestamp)

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

    print('\n===========================================')
    for i, track in enumerate(tracks):
        print(f'Track {i+1} - Exist prob: {track.exist_prob}')

    if PLOT:
        ax1.cla()
        for i, truth in enumerate(truths):
            data = np.array([s.state_vector for s in truth[:k + 1]])
            ax1.plot(data[:, 0], data[:, 2], '--', label=f'Groundtruth Track {i + 1}')
        if len(detections):
            det_data = np.array([det.state_vector for det in detections])
            ax1.plot(det_data[:, 0], det_data[:, 1], '*g', label='Detections')

        for track in tracks:
            data = np.array([s.state_vector for s in track])
            ax1.plot(data[:, 0], data[:, 2], label=f'Track {track.id}')
            plot_cov_ellipse(track.covar[[0, 2], :][:, [0, 2]], track.state_vector[[0, 2], :],
                             edgecolor='r', facecolor='none', ax=ax1)
        plt.axis([*surveillance_region[0], *surveillance_region[1]])
        plt.legend(loc='center right')
        plt.pause(0.1)