import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from copy import copy
from datetime import datetime, timedelta

from stonesoup.functions import gm_reduce_single

from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.predictor.imm import IMMPredictor
from stonesoup.updater.imm import IMMUpdater
from stonesoup.models.transition.linear import ConstantVelocity, \
    CombinedLinearGaussianTransitionModel, RandomWalk, OrnsteinUhlenbeck

from stonesoup.models.measurement.linear import LinearGaussian

from stonesoup.types.state import GaussianState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.detection import Detection
from stonesoup.types.state import StateVector, CovarianceMatrix, \
    GaussianMixtureState, WeightedGaussianState
from stonesoup.types.track import Track

from stonesoup.simulator.simple import SingleTargetGroundTruthSimulator

from matplotlib.patches import Ellipse
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
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
                    **kwargs)

    ax.add_artist(ellip)
    return ellip

##############################################################################
# MODELS                                                                     #
##############################################################################
transition_model_1 = CombinedLinearGaussianTransitionModel(
                        (OrnsteinUhlenbeck(0.1 ** 2, 2e-2),
                         OrnsteinUhlenbeck(0.1 ** 2, 2e-2)))
transition_model_2 = CombinedLinearGaussianTransitionModel(
                        (RandomWalk(0.0000000000000001**2),
                         RandomWalk(np.finfo(float).eps),
                         RandomWalk(0.0000000000000001**2),
                         RandomWalk(np.finfo(float).eps)))

measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2],
                                   noise_covar=np.diag([1 ** 2,
                                                        1 ** 2]))
##############################################################################
# PREDICTOR/UPDATER                                                          #
##############################################################################
model_transition_matrix = np.array([[0.95, 0.05],
                                    [0.3, 0.7]])
predictor_1 = KalmanPredictor(transition_model_1)
predictor_2 = KalmanPredictor(transition_model_2)
imm_predictor = IMMPredictor([predictor_1, predictor_2],
                             model_transition_matrix)

updater = KalmanUpdater(measurement_model)
imm_updater = IMMUpdater([updater, updater], model_transition_matrix)

##############################################################################
# GROUND-TRUTH SIMULATOR                                                     #
##############################################################################
timestamp_init = datetime.now()
state_init = WeightedGaussianState(StateVector([[1], [0], [0], [0]]),
                                   CovarianceMatrix(
                                       np.diag([10 ** 2, 20 ** 2,
                                                10 ** 2, 20 ** 2])),
                                   timestamp=timestamp_init,
                                   weight=0.5)
gndt = SingleTargetGroundTruthSimulator(transition_model_1, state_init,
                                        number_steps=500)

##############################################################################
# SIMULATION                                                                 #
##############################################################################

# Assume we know the initial position of the track
prior1 = copy(state_init)
prior = GaussianMixtureState([prior1, prior1])
track = Track([copy(prior)])

fig, (ax1, ax2) = plt.subplots(2,1)
for time, gnd_paths in gndt.groundtruth_paths_gen():
    # Extract ground-truth path and generate a noisy detection
    gnd_path = gnd_paths.pop()
    measurement = Detection(measurement_model.function(
                                gnd_path.state.state_vector,
                                measurement_model.rvs(1)),
                           time)

    # State prediction
    prediction = imm_predictor.predict(track.state, timestamp=time)
    # Measurement prediction
    meas_prediction = imm_updater.predict_measurement(prediction)
    # State update
    hyp = SingleHypothesis(prediction, measurement)
    prior = imm_updater.update(hyp)
    track.append(prior)

    # Generate plots
    ax1.cla()
    ax2.cla()
    # PLot true trajectory
    data = np.array([state.state_vector for state in gnd_path.states])
    ax1.plot(data[:, 0], data[:, 2], 'b-')
    # PLot estimated trajectory
    data = np.array([state.state_vector for state in track.states])
    ax1.plot(data[:, 0], data[:, 2], 'r-')
    # Plot innovation covariance
    plot_cov_ellipse(meas_prediction.covar,
                     meas_prediction.mean, edgecolor='b',
                     facecolor='none', ax=ax1)
    # Plot estimated covariance
    plot_cov_ellipse(track.state.covar[[0, 2], :][:, [0, 2]],
                     track.state.mean[[0, 2], :], edgecolor='r',
                     facecolor='none', ax=ax1)
    # Visualise model weights
    ax2.bar([1,2], prior.weights.ravel())
    plt.pause(0.0001)

