import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from copy import copy
from datetime import datetime, timedelta

from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater, SqrtKalmanUpdater
from stonesoup.models.transition.linear import ConstantVelocity, \
    CombinedLinearGaussianTransitionModel

from stonesoup.models.measurement.linear import LinearGaussian

from stonesoup.types.state import GaussianState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.detection import Detection
from stonesoup.types.state import StateVector, CovarianceMatrix
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


# Fix random number seed
np.random.seed(30)


##############################################################################
# MODELS                                                                     #
##############################################################################
transition_model = CombinedLinearGaussianTransitionModel(
                        (ConstantVelocity(0.1 ** 2),
                         ConstantVelocity(0.1 ** 2)))

measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2],
                                   noise_covar=np.diag([1 ** 2,
                                                        1 ** 2]))
##############################################################################
# PREDICTOR/UPDATER                                                          #
##############################################################################
predictor = KalmanPredictor(transition_model)
updater = SqrtKalmanUpdater(measurement_model)

##############################################################################
# GROUND-TRUTH SIMULATOR                                                     #
##############################################################################
timestamp_init = datetime.now()
state_init = GaussianState(StateVector([[1], [0], [0], [0]]),
                           CovarianceMatrix(np.diag([10 ** 2, 20 ** 2,
                                                     10 ** 2, 20 ** 2])),
                           timestamp=timestamp_init)
gndt = SingleTargetGroundTruthSimulator(transition_model, state_init,
                                        number_steps=100)

#------
#A = np.array([[1,2],[3,4]])
#print(A)
#[Q, R] = np.linalg.qr(A)
#------

##############################################################################
# SIMULATION                                                                 #
##############################################################################

# Assume we know the initial position of the track
prior = copy(state_init)
track = Track([copy(prior)])

fig, ax = plt.subplots()
for time, gnd_paths in gndt.groundtruth_paths_gen():
    # Extract ground-truth path and generate a noisy detection
    gnd_path = gnd_paths.pop()
    measurement = Detection(measurement_model.function(
                                gnd_path.state,
                                measurement_model.rvs(1)),
                            time)

    # State prediction
    prediction = predictor.predict(track.state, timestamp=time)
    # State update
    hyp = SingleHypothesis(prediction, measurement)
    posterior = updater.update(hyp)

    # Update the track
    track.append(posterior)

    # Generate plots
    ax.cla()
    # PLot true trajectory
    data = np.array([state.state_vector for state in gnd_path.states])
    ax.plot(data[:, 0], data[:, 2], 'b-')
    # PLot estimated trajectory
    data = np.array([state.state_vector for state in track.states])
    ax.plot(data[:, 0], data[:, 2], 'r-')
    # Plot innovation covariance
    meas_prediction = posterior.hypothesis.measurement_prediction
    plot_cov_ellipse(meas_prediction.covar,
                     meas_prediction.mean, edgecolor='b',
                     facecolor='none', ax=ax)
    # Plot estimated covariance
    plot_cov_ellipse(track.state.covar[[0, 2], :][:, [0, 2]],
                     track.state.mean[[0, 2], :], edgecolor='r',
                     facecolor='none', ax=ax)
    plt.pause(0.0001)