import numpy as np

from stonesoup.predictor.base import Predictor
from stonesoup.types import State, GaussianStatePrediction
from stonesoup.wrapper.matlab import MatlabWrapper


class MatlabKalmanPredictor(Predictor, MatlabWrapper):
    """A standard Kalman predictor using MATLAB functions to prove that you can
    Note that unlike the standard Kalman filter this does not use control \
    input covariance """

    def predict(self, prior, control_input=None, timestamp=None, **kwargs):

        # Compute time_interval
        try:
            time_interval = timestamp - prior.timestamp
        except TypeError:
            # TypeError: (timestamp or prior.timestamp) is None
            time_interval = None

        # Transition model parameters
        transition_matrix = self.transition_model.matrix(
            timestamp=timestamp,
            time_interval=time_interval,
            **kwargs)
        transition_noise_covar = self.transition_model.covar(
            timestamp=timestamp,
            time_interval=time_interval,
            **kwargs)

        # Control model parameters
        if self.control_model is None:
            control_matrix = np.zeros(prior.covar.shape)
            contol_noise_covar = np.zeros(prior.covar.shape)
            control_input = State(np.zeros(prior.state_vector.shape))
        else:
            # Extract control matrix
            control_matrix = self.control_model.matrix(
                timestamp=timestamp,
                time_interval=time_interval,
                **kwargs)
            # Extract control noise covariance
            try:
                # covar() is implemented for control_model
                contol_noise_covar = self.control_model.covar(
                    timestamp=timestamp,
                    time_interval=time_interval,
                    **kwargs)
            except AttributeError:
                # covar() is NOT implemented for control_model
                contol_noise_covar = np.zeros(self.control_model.ndim_ctrl)
            if control_input is None:
                control_input = np.zeros((self.control_model.ndim_ctrl, 1))

        x = self.matlab_array(prior.state_vector)
        P = self.matlab_array(prior.covar)
        F = self.matlab_array(transition_matrix)
        Q = self.matlab_array(transition_noise_covar)
        B = self.matlab_array(control_matrix)
        u = self.matlab_array(control_input.state_vector)
        Qu = self.matlab_array(contol_noise_covar)

        pred_mean, pred_covar = self.matlab_engine.kf_predict(x, P, F, Q,
                                                              u, B, Qu,
                                                              nargout=2)

        return GaussianStatePrediction(np.array(pred_mean),
                                       np.array(pred_covar),
                                       timestamp)
