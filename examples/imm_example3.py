import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from copy import copy
from datetime import datetime, timedelta


from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.predictor.mixturemodelproxy import add_mixture_capability

from stonesoup.predictor.imm import IMMPredictor
from stonesoup.updater.imm import IMMUpdater
from stonesoup.models.transition.linear import ConstantVelocity, \
    CombinedLinearGaussianTransitionModel, RandomWalk, OrnsteinUhlenbeck, \
    ConstantAcceleration

from stonesoup.models.measurement.linear import LinearGaussian

from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianMixtureState, WeightedGaussianState
from stonesoup.types.track import Track
from stonesoup.types.prediction import GaussianStatePrediction, \
    GaussianMeasurementPrediction,  MeasurementPrediction
from stonesoup.types.array import StateVector, CovarianceMatrix

from stonesoup.simulator.simple import SingleTargetGroundTruthSimulator

from matplotlib.patches import Ellipse

#######################################################################
# This uses a CV and CA IMM. Inside the IMM uses the CA model
# The taarget simulator uses a CA model.
#######################################################################


def cv2ca(state):
    out = copy(state)
    # By copying the state we only have to overwrite the affected components.
    # Elements like hypothesis and timestamp are automatically copied
    if isinstance(state, MeasurementPrediction):
        # This is a measurement model conversion
        state_vector = state.state_vector

        if state.cross_covar.shape[0] == 4:
            ndim = 6
            ind = np.array([0, 1, 3, 4])
        elif state.cross_covar.shape[0] == 6:
            ndim = 9
            ind = np.array([0, 1, 3, 4, 6, 7])

        cross_covar = CovarianceMatrix(np.zeros((ndim, state_vector.shape[0])))
        cross_covar[ind, :] = state.cross_covar[:, :]
        out.cross_covar = cross_covar
    else:
        # Doing a state model conversion
        # Assume a full 3D constant acceleration model
        ndim = 9
        ind = np.array([0, 1, 3, 4, 6, 7])
        # Check if we need to switch to a 2D CA model
        if state.state_vector.shape[0] == 4:
            ndim = 6
            ind = np.array([0, 1, 3, 4])
        state_vector = StateVector(np.zeros((1, ndim)))
        covar = CovarianceMatrix(np.zeros((ndim, ndim)))
        state_vector[ind] = state.state_vector
        covar[ind[:, np.newaxis], ind] = state.covar
        out.state_vector = state_vector
        out.covar = covar
    return out


def ca2cv(state):
    out = copy(state)
    # By copying the state we only have to overwrite the affected components.
    # Elements like hypothesis and timestamp are automatically copied

    if isinstance(state, MeasurementPrediction):
        if state.cross_covar.shape[0] == 4:
            ndim = 9
            ind = np.array([0, 1, 3, 4, 6, 7])
        elif state.cross_covar.shape[0] == 6:
            ndim = 6
            ind = np.array([0, 1, 3, 4])
        cross_covar = state.cross_covar[ind, :]
        out.cross_covar = cross_covar
    else:
        # Assume a full 3D constant velocity model
        ndim = 6
        ind = np.array([0, 1, 3, 4, 6, 7])
        # Check if we need to switch to a 2D CV model
        if state.state_vector.shape[0] == 6:
            ndim = 4
            ind = np.array([0, 1, 3, 4])
        state_vector = StateVector(state.state_vector[ind])
        covar = state.covar[ind[:, np.newaxis], ind]

        out.state_vector = state_vector
        out.covar = covar
    return out


def null_convert(state):
    return state


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
trans_1 = CombinedLinearGaussianTransitionModel((ConstantVelocity(0.1),
                                                ConstantVelocity(0.1)))
trans_2 = CombinedLinearGaussianTransitionModel((ConstantAcceleration(0.1),
                                                ConstantAcceleration(0.1)))
sim_trans_model = trans_2

measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2],
                                   noise_covar=np.diag([1 ** 2,
                                                        1 ** 2]))

measurement_model2 = LinearGaussian(ndim_state=6, mapping=[0, 3],
                                    noise_covar=np.diag([1 ** 2,
                                                         1 ** 2]))

sim_measurement_model = measurement_model2
##############################################################################
# PREDICTOR/UPDATER                                                          #
##############################################################################
model_transition_matrix = np.array([[0.95, 0.05],
                                    [0.3, 0.7]])

KalmanPredictorMM = add_mixture_capability(KalmanPredictor)
predictor_1 = KalmanPredictor(transition_model_1)
predictor_2 = KalmanPredictor(transition_model_2)
predictor_cv = KalmanPredictorMM(trans_1, convert2common_state=cv2ca,
                                 convert2local_state=ca2cv)
predictor_ca = KalmanPredictorMM(trans_2, convert2common_state=null_convert,
                                 convert2local_state=null_convert)
imm_predictor = IMMPredictor([predictor_cv, predictor_ca],
                             model_transition_matrix)

KalmanUpdaterMM = add_mixture_capability(KalmanUpdater)
updater_cv = KalmanUpdaterMM(measurement_model,
                             convert2common_state=cv2ca,
                             convert2local_state=ca2cv)
updater_ca = KalmanUpdaterMM(measurement_model2)
imm_updater = IMMUpdater([updater_cv, updater_ca], model_transition_matrix)


##############################################################################
# Mixture Tracker Common State                                               #
##############################################################################
timestamp_init = datetime.now()
# Use 4 dim state - i.e. CV model
state_init = WeightedGaussianState(StateVector([[1], [0], [0], [0]]),
                                   CovarianceMatrix(
                                       np.diag([10 ** 2, 20 ** 2,
                                                10 ** 2, 20 ** 2])),
                                   timestamp=timestamp_init,
                                   weight=0.5)
# Use 6 dim state - i.e. CA model
state_init = WeightedGaussianState(StateVector([[1], [0], [0], [0], [0], [0]]),
                                   CovarianceMatrix(
                                       np.diag([10 ** 2, 20 ** 2, 5 ** 2,
                                                10 ** 2, 20 ** 2, 5 ** 2])),
                                   timestamp=timestamp_init,
                                   weight=0.5)

##############################################################################
# GROUND-TRUTH SIMULATOR                                                     #
##############################################################################
# Simulator is using a 4 dim state
sim_state_init = WeightedGaussianState(StateVector([[1], [0], [0], [0]]),
                                       CovarianceMatrix(
                                           np.diag([10 ** 2, 20 ** 2,
                                                    10 ** 2, 20 ** 2])),
                                       timestamp=timestamp_init,
                                       weight=0.5)

sim_state_init = WeightedGaussianState(StateVector(
        [[1], [0], [0], [0], [0], [0]]),
        CovarianceMatrix(
            np.diag([10 ** 2, 20 ** 2, 5 ** 2,
                     10 ** 2, 20 ** 2, 5 ** 2])),
        timestamp=timestamp_init,
        weight=0.5)
gndt = SingleTargetGroundTruthSimulator(sim_trans_model, sim_state_init,
                                        number_steps=200)

##############################################################################
# SIMULATION                                                                 #
##############################################################################

# Assume we know the initial position of the track
prior1 = copy(state_init)
prior = GaussianMixtureState([prior1, prior1])
track = Track([copy(prior)])
cur_state = []
fig, (ax1, ax2) = plt.subplots(2, 1)
for time, gnd_paths in gndt.groundtruth_paths_gen():
    # Extract ground-truth path and generate a noisy detection
    if type(cur_state) != list:
        old_state = cur_state
    gnd_path = gnd_paths.pop()
    cur_state = gnd_path.state.state_vector
    # measurement = Detection(sim_measurement_model.function(
    #         gnd_path.state.state_vector,
    #         measurement_model.rvs(1)),
    #         time)
    measurement = Detection(sim_measurement_model.function(
            gnd_path.state,
            sim_measurement_model.rvs(1)),
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

    data_t = np.array([state.state_vector for state in gnd_path.states])
    t_ind = []
    if data_t[0].shape[0] == 4:
        # Assume this a constant velocity model
        t_ind = 2
    if data_t[0].shape[0] == 6:
        # Assume this is a constant acceleration model
        t_ind = 3
    ax1.plot(data_t[:, 0], data_t[:, t_ind], 'b-')
    # PLot estimated trajectory
    data = np.array([state.state_vector for state in track.states])

    # Set the ind to take from the tracker state and covariances
    ind = []
    if data[0].shape[0] == 4:
        # Assume this a constant velocity model
        ind = 2
    if data[0].shape[0] == 6:
        # Assume this is a constant acceleration model
        ind = 3

    ax1.plot(data[:, 0], data[:, ind], 'r-')
    # Plot innovation covariance
    plot_cov_ellipse(meas_prediction.covar,
                     meas_prediction.mean, edgecolor='b',
                     facecolor='none', ax=ax1)
    # Plot estimated covariance
    plot_cov_ellipse(track.state.covar[[0, ind], :][:, [0, ind]],
                     track.state.mean[[0, ind], :], edgecolor='r',
                     facecolor='none', ax=ax1)
    # Visualise model weights
    ax2.bar([1, 2], prior.weights.ravel())
    plt.pause(0.0001)
