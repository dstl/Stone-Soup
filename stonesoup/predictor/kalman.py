# -*- coding: utf-8 -*-
import copy
from collections import OrderedDict
from functools import lru_cache, partial

import numpy as np

from .base import Predictor
from ..base import Property
from ..functions import gauss2sigma, unscented_transform
from ..models.base import LinearModel
from ..models.control import ControlModel
from ..models.control.linear import LinearControlModel
from ..models.transition import TransitionModel
from ..models.transition.linear import LinearGaussianTransitionModel
from ..types.prediction import GaussianStatePrediction, ASDGaussianStatePrediction


class KalmanPredictor(Predictor):
    r"""A predictor class which forms the basis for the family of Kalman
    predictors. This class also serves as the (specific) Kalman Filter
    :class:`~.Predictor` class. Here

    .. math::

      f_k( \mathbf{x}_{k-1}) = F_k \mathbf{x}_{k-1},  \ b_k( \mathbf{x}_k) =
      B_k \mathbf{x}_k \ \mathrm{and} \ \mathbf{\nu}_k \sim \mathcal{N}(0,Q_k)


    Notes
    -----
    In the Kalman filter, transition and control models must be linear.


    Raises
    ------
    ValueError
        If no :class:`~.TransitionModel` is specified.


    """

    transition_model = Property(
        LinearGaussianTransitionModel,
        doc="The transition model to be used.")
    control_model = Property(
        LinearControlModel,
        default=None,
        doc="The control model to be used. Default `None` where the predictor "
            "will create a zero-effect linear :class:`~.ControlModel`.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # If no control model insert a linear zero-effect one
        # TODO: Think about whether it's more efficient to leave this out
        if self.control_model is None:
            ndims = self.transition_model.ndim_state
            self.control_model = LinearControlModel(ndims, [],
                                                    np.zeros([ndims, 1]),
                                                    np.zeros([ndims, ndims]),
                                                    np.zeros([ndims, ndims]))

    def _transition_matrix(self, **kwargs):
        """Return the transition matrix

        Parameters
        ----------
        **kwargs : various, optional
            These are passed to :meth:`~.LinearGaussianTransitionModel.matrix`

        Returns
        -------
        : :class:`numpy.ndarray`
            The transition matrix, :math:`F_k`

        """
        return self.transition_model.matrix(**kwargs)

    def _transition_function(self, prior, **kwargs):
        r"""Applies the linear transition function to a single vector in the
        absence of a control input, returns a single predicted state.

        Parameters
        ----------
        prior : :class:`~.State`
            The prior state, :math:`\mathbf{x}_{k-1}`

        **kwargs : various, optional
            These are passed to :meth:`~.LinearGaussianTransitionModel.matrix`

        Returns
        -------
        : :class:`~.State`
            The predicted state

        """
        return self.transition_model.matrix(**kwargs) @ prior.state_vector

    @property
    def _control_matrix(self):
        r"""Convenience function which returns the control matrix

        Returns
        -------
        : :class:`numpy.ndarray`
            control matrix, :math:`B_k`

        """
        return self.control_model.matrix()

    def _predict_over_interval(self, prior, timestamp):
        """Private function to get the prediction interval (or None)

        Parameters
        ----------
        prior : :class:`~.State`
            The prior state

        timestamp : :class:`datetime.datetime`, optional
            The (current) timestamp

        Returns
        -------
        : :class:`datetime.timedelta`
            time interval to predict over

        """

        # Deal with undefined timestamps
        if timestamp is None or prior.timestamp is None:
            predict_over_interval = None
        else:
            predict_over_interval = timestamp - prior.timestamp

        return predict_over_interval

    @lru_cache()
    def predict(self, prior, timestamp=None, **kwargs):
        r"""The predict function

        Parameters
        ----------
        prior : :class:`~.State`
            :math:`\mathbf{x}_{k-1}`
        timestamp : :class:`datetime.datetime`, optional
            :math:`k`
        **kwargs :
            These are passed, via :meth:`~.KalmanFilter.transition_function` to
            :meth:`~.LinearGaussianTransitionModel.matrix`

        Returns
        -------
        : :class:`~.State`
            :math:`\mathbf{x}_{k|k-1}`, the predicted state and the predicted
            state covariance :math:`P_{k|k-1}`

        """

        # Get the prediction interval
        predict_over_interval = self._predict_over_interval(prior, timestamp)

        # Prediction of the mean
        x_pred = self._transition_function(
            prior, time_interval=predict_over_interval, **kwargs) \
            + self.control_model.control_input()

        # As this is Kalman-like, the control model must be capable of
        # returning a control matrix (B)

        transition_matrix = self._transition_matrix(
            prior=prior, time_interval=predict_over_interval, **kwargs)
        transition_covar = self.transition_model.covar(
            time_interval=predict_over_interval, **kwargs)

        control_matrix = self._control_matrix
        control_noise = self.control_model.control_noise

        p_pred = transition_matrix @ prior.covar @ transition_matrix.T \
            + transition_covar \
            + control_matrix @ control_noise @ control_matrix.T

        return GaussianStatePrediction(x_pred, p_pred, timestamp=timestamp)


class ASDKalmanPredictor(KalmanPredictor):
    def _predict_over_interval(self, prior, timestamp):
        """Private function to get the prediction interval (or None)

        Parameters
        ----------
        prior : :class:`~.State`
            The prior state

        timestamp : :class:`datetime.datetime`, optional
            The (current) timestamp

        Returns
        -------
        : :class:`datetime.timedelta`
            time interval to predict over
            if the timestamp is in the past then it returns the interval
            to the next timestamp in the state
        : :class:`datetime.datetime`
            time from which the interval is calculated

        """

        predict_over_interval = timestamp - prior.timestamp
        timestamp_from_which_is_predicted = prior.timestamp
        if predict_over_interval.days < 0:
            predict_over_interval, timestamp_from_which_is_predicted = min([(timestamp - t, t) for t in prior.timestamps if (timestamp - t).days == 0], key=lambda x:x[0])

        return predict_over_interval, timestamp_from_which_is_predicted


    def _transition_function(self, prior, timestamp_from_which_is_predicted, **kwargs):
        r"""Applies the linear transition function to a single vector in the
        absence of a control input, returns a single predicted state.

        Parameters
        ----------
        prior : :class:`~.State`
            The prior state, :math:`\mathbf{x}_{k-1}`
        timestamp_from_which_is_predicted : :class:`datetime.datetime
            This is the timestamp from which is predicted

        **kwargs : various, optional
            These are passed to :meth:`~.LinearGaussianTransitionModel.matrix`

        Returns
        -------
        : :class:`~.State`
            The predicted state

        """
        if prior.timestamp == timestamp_from_which_is_predicted:
            return self.transition_model.matrix(**kwargs) @ prior.state_vector
        else:
            t_index = prior.timestamps.index(timestamp_from_which_is_predicted)
            return self.transition_model.matrix(**kwargs) @ prior.multi_state_vector[t_index * prior.ndim: (t_index+1) * prior.ndim]

    @lru_cache()
    def predict(self, prior, timestamp, **kwargs):
        r"""The predict function

                Parameters
                ----------
                prior : :class:`~.ASDState`
                    :math:`\mathbf{x}_{k-1}`
                timestamp : :class:`datetime.datetime`,
                    :math:`k`
                **kwargs :
                    These are passed, via :meth:`~.KalmanFilter.transition_function` to
                    :meth:`~.LinearGaussianTransitionModel.matrix`

                Returns
                -------
                : :class:`~.ASDState`
                    :math:`\mathbf{x}_{k|k-1}`, the predicted state and the predicted
                    state covariance :math:`P_{k|k-1}`

                """

        correlation_matrices = copy.deepcopy(prior.correlation_matrices)

        # Get the prediction interval
        predict_over_interval, timestamp_from_which_is_predicted = self._predict_over_interval(prior, timestamp)

        # Build the first correlation matrix just after starting the predictor the first time.
        if len(correlation_matrices) == 0:
            correlation_matrices[prior.timestamp] = {}
            correlation_matrices[prior.timestamp]['P']=prior.covar

        # Prediction of the mean
        x_pred_k = self._transition_function(
            prior, timestamp_from_which_is_predicted, time_interval=predict_over_interval, **kwargs) \
                   + self.control_model.control_input()

        # As this is Kalman-like, the control model must be capable of
        # returning a control matrix (B)

        transition_matrix = self._transition_matrix(
            prior=prior, time_interval=predict_over_interval, **kwargs)
        transition_covar = self.transition_model.covar(
            time_interval=predict_over_interval, **kwargs)

        control_matrix = self._control_matrix
        control_noise = self.control_model.control_noise



        # Consider, if the given timestep is an Out-Of-Sequence measurement
        t_index = prior.timestamps.index(timestamp_from_which_is_predicted)
        # case that it is a normal prediction
        if t_index == 0:
            # Generation of the Correlation matrices
            C_list = self.generate_C_list(prior, correlation_matrices)
            # normal case
            x_pred = np.concatenate([x_pred_k, prior.multi_state_vector])
            W_P_column = np.array([c @ prior.covar for c in C_list])
            correlated_column = np.reshape(W_P_column, (
                prior.multi_covar.shape[0], prior.ndim)) @ transition_matrix.T
            correlated_row = correlated_column.T

            # put row and col block matrices together
            p_pred_k = transition_matrix @ prior.covar @ transition_matrix.T \
                       + transition_covar \
                       + control_matrix @ control_noise @ control_matrix.T
            p_top = np.hstack((p_pred_k, correlated_row))
            p_bottom = np.hstack((correlated_column, prior.multi_covar))
            p_pred = np.vstack((p_top, p_bottom))

            # add new correlation matrix with the present time step
            correlation_matrices[timestamp_from_which_is_predicted]['P_pred'] = p_pred_k
            correlation_matrices[timestamp_from_which_is_predicted]['F'] = transition_matrix
            correlation_matrices[timestamp_from_which_is_predicted]['PFP'] = \
                correlation_matrices[timestamp_from_which_is_predicted]['P'] \
                @ correlation_matrices[timestamp_from_which_is_predicted]['F'].T \
                @ np.linalg.inv(correlation_matrices[timestamp_from_which_is_predicted]['P_pred'])

        else:
            # case of out of sequence prediction case
            print("The measurement is processed as an Out-Of-Sequence measurement.")
            next_timestamp = prior.timestamps[prior.timestamps.index(timestamp_from_which_is_predicted) - 1]
            time_interval_to_next_timestep = next_timestamp - timestamp


            transition_matrix_plus_1 = super()._transition_matrix(time_interval=time_interval_to_next_timestep, **kwargs)

            # prediction to the next timestamp
            x_pred_m_plus_1 = transition_matrix_plus_1 @ x_pred_k
            x_diff = prior.multi_state_vector[(t_index - 1) * prior.ndim:t_index * prior.ndim] - x_pred_m_plus_1

            # Normal Kalman-like prediction for the state m+1|m.
            covar_from_where_ist_predicted = prior.multi_covar[t_index * prior.ndim:(t_index + 1) * prior.ndim,
                                             t_index * prior.ndim:(t_index + 1) * prior.ndim]

            p_pred_k = transition_matrix @ covar_from_where_ist_predicted @ transition_matrix.T \
                       + transition_covar \
                       + control_matrix @ control_noise @ control_matrix.T
            p_pred_m_plus_1 = transition_matrix_plus_1 @ p_pred_k @ super()._transition_matrix(time_interval=time_interval_to_next_timestep, **kwargs).T
            p_diff = prior.multi_covar[(t_index - 1) * prior.ndim:t_index * prior.ndim, (t_index - 1) * prior.ndim:t_index * prior.ndim] - p_pred_m_plus_1

            W = p_pred_k @ transition_matrix_plus_1.T @ np.linalg.inv(p_pred_m_plus_1)
            x_pred_m = x_pred_k + W @ x_diff
            p_pred_k = p_pred_k - W @ p_diff @ W.T
            x_pred = np.concatenate([prior.multi_state_vector[0:t_index*prior.ndim], x_pred_m, prior.multi_state_vector[t_index*prior.ndim:]])

            P_right_upper = prior.multi_covar[0:t_index*prior.ndim, t_index*prior.ndim:]
            P_right_lower = prior.multi_covar[t_index*prior.ndim:, t_index*prior.ndim:]
            P_left_upper = prior.multi_covar[0:t_index*prior.ndim, 0:t_index*prior.ndim]
            P_left_lower = prior.multi_covar[t_index*prior.ndim:, 0:t_index*prior.ndim]

            # add new correlation matrix with the present time step
            old_transition_matrix = correlation_matrices[timestamp_from_which_is_predicted]['F']
            correlation_matrices[timestamp_from_which_is_predicted]['P_pred'] = p_pred_k
            correlation_matrices[timestamp_from_which_is_predicted]['F'] = transition_matrix
            correlation_matrices[timestamp_from_which_is_predicted]['PFP'] = \
                correlation_matrices[timestamp_from_which_is_predicted]['P'] \
                @ correlation_matrices[timestamp_from_which_is_predicted]['F'].T \
                @ np.linalg.inv(correlation_matrices[timestamp_from_which_is_predicted]['P_pred'])
            correlation_matrices[timestamp] = {}
            correlation_matrices[timestamp]['F'] = transition_matrix_plus_1
            correlation_matrices[timestamp]['P_pred'] = p_pred_m_plus_1
            correlation_matrices[timestamp]['P'] = p_pred_k


            # resort the dict
            correlation_matrices = OrderedDict(sorted(correlation_matrices.items()))

            # normal part of Correlation matrices
            C_list = []
            C_list.append(np.eye(prior.ndim))
            for item in [value for key, value in correlation_matrices.items() if key<timestamp][-2::-1]:
                C_list.append(C_list[-1] @ item['PFP'])
            W_P_column_normal = np.array([c @ covar_from_where_ist_predicted for c in C_list])
            correlated_column_normal = np.reshape(W_P_column_normal, (
                prior.ndim * len(C_list), prior.ndim)) @ transition_matrix.T
            correlated_row_normal = correlated_column_normal.T


            # other part of the new column/row
            C_rest_list = []
            C_rest_list.append(np.eye(prior.ndim))
            for item in [value for key, value in correlation_matrices.items() if key>timestamp][:-1]:
                C_rest_list.append(item['PFP'] @ C_rest_list[-1])


            corrs = [prior.multi_covar[i*prior.ndim:(i+1)*prior.ndim,i*prior.ndim:(i+1)*prior.ndim] for i in range(0,t_index)]

            W_P_column_rest = np.array([c @ covar  for c, covar in zip(C_rest_list, corrs)])

            correlated_column_rest = np.reshape(W_P_column_rest, (
                prior.ndim * len(C_rest_list), prior.ndim))
            correlated_row_rest = correlated_column_rest.T

            # make one matrix out of the small ones
            P = np.block([[P_left_upper, correlated_column_rest, P_right_upper],
                        [correlated_row_rest, p_pred_k, correlated_row_normal],
                        [P_left_lower, correlated_column_normal, P_right_lower]])

            # # correct of the covariance parts of the following
            # P[ (t_index-1) * prior.ndim:,(t_index-1) * prior.ndim:t_index  *prior.ndim] = \
            #     P[ (t_index-1) * prior.ndim:,(t_index-1) * prior.ndim:t_index  *prior.ndim] @ np.linalg.inv(old_transition_matrix.T) @ transition_matrix_plus_1

            p_pred = P


        timestamps = sorted(prior.timestamps +[timestamp], reverse=True)
        predicted_state = ASDGaussianStatePrediction(multi_state_vector=x_pred, multi_covar=p_pred,
                                          correlation_matrices=correlation_matrices,
                                          timestamps=timestamps, max_nstep=prior.max_nstep, act_timestamp=timestamp)
        self.prune_state(predicted_state)
        return predicted_state

    def generate_C_list(self, prior, correlation_matrices):
        prior_ndim = prior.ndim
        C_list = []
        C_list.append(np.eye(prior_ndim))
        for item in list(correlation_matrices.values())[-2::-1]:
            C_list.append(C_list[-1] @ item['PFP'])
        return C_list

    def prune_state(self, predicted_state):
        r"""Simple ASD pruning function. Deletes one timesteps from the multi state if it is longer then max_nstep

                Parameters
                ----------
                prior : :class:`~.ASDState`
                    :math:`\mathbf{x}_{k|k-1}`

                Returns
                -------
                : :class:`~.ASDState`
                    :math:`\mathbf{x}_{k|k-1}`, the pruned state and the pruned
                    state covariance :math:`P_{k|k-1}`

                """
        if predicted_state.nstep > predicted_state.max_nstep and predicted_state.max_nstep != 0:
            index = predicted_state.max_nstep * predicted_state.ndim
            predicted_state.multi_state_vector = predicted_state.multi_state_vector[0:index]
            predicted_state.multi_covar = predicted_state.multi_covar[0:index, 0:index]
            deleted_timestamps = predicted_state.timestamps[predicted_state.max_nstep:]
            predicted_state.timestamps = predicted_state.timestamps[0:predicted_state.max_nstep]
            _ = [predicted_state.correlation_matrices.pop(el) for el in deleted_timestamps]
            return predicted_state
        else:
            return predicted_state


class ExtendedKalmanPredictor(KalmanPredictor):
    """ExtendedKalmanPredictor class

    An implementation of the Extended Kalman Filter predictor. Here the
    transition and control functions may be non-linear, their transition and
    control matrices are approximated via Jacobian matrices. To this end the
    transition and control models, if non-linear, must be able to return the
    :attr:`jacobian()` function.

    """

    # In this version the models can be non-linear, but must have access to the
    # :attr:`jacobian()` function
    # TODO: Enforce the presence of :attr:`jacobian()`
    transition_model = Property(
        TransitionModel,
        doc="The transition model to be used.")
    control_model = Property(
        ControlModel,
        default=None,
        doc="The control model to be used. Default `None` where the predictor "
            "will create a zero-effect linear :class:`~.ControlModel`.")

    def _transition_matrix(self, prior, **kwargs):
        r"""Returns the transition matrix, a matrix if the model is linear, or
        approximated as Jacobian otherwise.

        Parameters
        ----------
        prior : :class:`~.State`
            :math:`\mathbf{x}_{k-1}`
        **kwargs : various, optional
            These are passed to :meth:`~.TransitionModel.matrix` or
            :meth:`~.TransitionModel.jacobian`

        Returns
        -------
        : :class:`numpy.ndarray`
            The transition matrix, :math:`F_k`, if linear (i.e.
            :meth:`TransitionModel.matrix` exists, or
            :meth:`~.TransitionModel.jacobian` if not)
        """
        if isinstance(self.transition_model, LinearModel):
            return self.transition_model.matrix(**kwargs)
        else:
            return self.transition_model.jacobian(prior.state_vector, **kwargs)

    def _transition_function(self, prior, **kwargs):
        r"""This is the application of :math:`f_k(\mathbf{x}_{k-1})`, the
        transition function, non-linear in general, in the absence of a control
        input

        Parameters
        ----------
        prior : :class:`~.State`
            The prior state, :math:`\mathbf{x}_{k-1}`
        **kwargs : various, optional
            These are passed to :meth:`~.TransitionModel.function`

        Returns
        -------
        : :class:`~.State`
            The predicted state

        """
        return self.transition_model.function(prior.state_vector, noise=0,
                                              **kwargs)

    @property
    def _control_matrix(self):
        r"""Returns the control input model matrix, :math:`B_k`, or its linear
        approximation via a Jacobian. The :class:`~.ControlModel`, if
        non-linear must therefore be capable of returning a
        :meth:`~.ControlModel.jacobian`,

        Returns
        -------
        : :class:`numpy.ndarray`
            The control model matrix, or its linear approximation
        """
        if isinstance(self.control_model, LinearModel):
            return self.control_model.matrix()
        else:
            return self.control_model.jacobian(
                self.control_model.control_vector)


class UnscentedKalmanPredictor(KalmanPredictor):
    """UnscentedKalmanFilter class

    The predict is accomplished by calculating the sigma points from the
    Gaussian mean and covariance, then putting these through the (in general
    non-linear) transition function, then reconstructing the Gaussian.
    """
    transition_model = Property(
        TransitionModel,
        doc="The transition model to be used.")
    control_model = Property(
        ControlModel,
        default=None,
        doc="The control model to be used. Default `None` where the predictor "
            "will create a zero-effect linear :class:`~.ControlModel`.")
    alpha = Property(
        float,
        default=0.5,
        doc="Primary sigma point spread scaling parameter. Default is 0.5.")
    beta = Property(
        float,
        default=2,
        doc="Used to incorporate prior knowledge of the distribution. If the "
            "true distribution is Gaussian, the value of 2 is optimal. "
            "Default is 2")
    kappa = Property(
        float,
        default=0,
        doc="Secondary spread scaling parameter. Default is calculated as "
            "3-Ns")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._time_interval = None

    def _transition_and_control_function(self, prior_state_vector, **kwargs):
        r"""Returns the result of applying the transition and control functions
        for the unscented transform

        Parameters
        ----------
        prior_state_vector : :class:`~.State`
            Prior state vector
        **kwargs : various, optional
            These are passed to :meth:`~.TransitionModel.function`

        Returns
        -------
        : :class:`numpy.ndarray`
            The combined, noiseless, effect of applying the transition and
            control
        """

        return \
            self.transition_model.function(
                prior_state_vector, noise=0, **kwargs) \
            + self.control_model.control_input()

    @lru_cache()
    def predict(self, prior, timestamp=None, **kwargs):
        r"""The unscented version of the predict step

        Parameters
        ----------
        prior : :class:`~.State`
            Prior state, :math:`\mathbf{x}_{k-1}`
        timestamp : :class:`datetime.datetime`
            Time to transit to (:math:`k`)
        **kwargs : various, optional
            These are passed to :meth:`~.TransitionModel.covar`

        Returns
        -------
        : :class:`~.GaussianStatePrediction`
            The predicted state :math:`\mathbf{x}_{k|k-1}` and the predicted
            state covariance :math:`P_{k|k-1}`
        """

        # Get the prediction interval
        predict_over_interval = self._predict_over_interval(prior, timestamp)

        # The covariance on the transition model + the control model
        # TODO: Note that I'm not sure you can actually do this with the
        # TODO: covariances, i.e. sum them before calculating
        # TODO: the sigma points and then just sticking them into the
        # TODO: unscented transform, and I haven't checked the statistics.
        total_noise_covar = \
            self.transition_model.covar(
                time_interval=predict_over_interval, **kwargs) \
            + self.control_model.control_noise

        # Get the sigma points from the prior mean and covariance.
        sigma_points, mean_weights, covar_weights = gauss2sigma(
            prior.state_vector, prior.covar, self.alpha, self.beta, self.kappa)

        # This ensures that function passed to unscented transform has the
        # correct time interval
        transition_and_control_function = partial(
            self._transition_and_control_function,
            time_interval=predict_over_interval)

        # Put these through the unscented transform, together with the total
        # covariance to get the parameters of the Gaussian
        x_pred, p_pred, _, _, _, _ = unscented_transform(
            sigma_points, mean_weights, covar_weights,
            transition_and_control_function, covar_noise=total_noise_covar
        )

        # and return a Gaussian state based on these parameters
        return GaussianStatePrediction(x_pred, p_pred, timestamp=timestamp)
