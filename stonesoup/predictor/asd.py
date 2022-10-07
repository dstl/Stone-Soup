import copy
from collections import OrderedDict
from itertools import islice
from operator import itemgetter

import numpy as np

from ._utils import predict_lru_cache
from .kalman import KalmanPredictor
from ..types.prediction import ASDGaussianStatePrediction


class ASDKalmanPredictor(KalmanPredictor):
    """Accumulated State Densities Kalman Predictor

      A linear predictor for accumulated state densities, for processing out of
      sequence measurements. This requires the state is represented in
      :class:`ASDGaussianState` multi-state.

      References
      ----------
      1.  W. Koch and F. Govaers, On Accumulated State Densities with Applications to
          Out-of-Sequence Measurement Processing in IEEE Transactions on Aerospace and
          Electronic Systems,
          vol. 47, no. 4, pp. 2766-2778, OCTOBER 2011, doi: 10.1109/TAES.2011.6034663.
    """
    def _predict_over_interval(self, prior, timestamp):
        """Private function to get the prediction interval (or None)

        Parameters
        ----------
        prior : :class:`~.ASDState`
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
        if predict_over_interval.total_seconds() < 0:
            predict_over_interval, timestamp_from_which_is_predicted = min(
                ((timestamp - t, t)
                 for t in prior.timestamps if (timestamp - t).total_seconds() > 0),
                key=itemgetter(0))

        return predict_over_interval, timestamp_from_which_is_predicted

    def _transition_function(self, prior, timestamp_from_which_is_predicted,
                             **kwargs):
        r"""Applies the linear transition function to a single vector in the
        absence of a control input, returns a single predicted state.

        Parameters
        ----------
        prior : :class:`~.ASDState`
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
        t_index = prior.timestamps.index(timestamp_from_which_is_predicted)
        t2t_plus = slice(t_index * prior.ndim, (t_index+1) * prior.ndim)
        if t_index == 0:
            return self.transition_model.matrix(**kwargs) @ prior.state_vector
        else:
            return self.transition_model.matrix(**kwargs) @ prior.multi_state_vector[t2t_plus]

    @predict_lru_cache()
    def predict(self, prior, timestamp, **kwargs):
        r"""The predict function

        Parameters
        ----------
        prior : :class:`~.ASDGaussianState`
            :math:`\mathbf{x}_{k-1}`
        timestamp : :class:`datetime.datetime`,
            :math:`k`
        **kwargs :
            These are passed, via
            :meth:`~.KalmanFilter.transition_function` to
            :meth:`~.LinearGaussianTransitionModel.matrix`

        Returns
        -------
        : :class:`~.ASDState`
            :math:`\mathbf{x}_{k|k-1}`, the predicted state and the
            predicted state covariance :math:`P_{k|k-1}`

        """

        correlation_matrices = copy.deepcopy(prior.correlation_matrices)

        # Get the prediction interval
        predict_over_interval, timestamp_from_which_is_predicted = \
            self._predict_over_interval(prior, timestamp)

        # Build the first correlation matrix just after starting the
        # predictor the first time.
        if not correlation_matrices:
            correlation_matrices.setdefault(prior.timestamp, dict())['P'] = prior.covar

        # Consider, if the given timestep is an Out-Of-Sequence measurement
        t_index = prior.timestamps.index(timestamp_from_which_is_predicted)

        def pred(transition_matrix, transition_covar, state_vector, covar):
            ctrl_model = self.control_model
            p = transition_matrix@covar@transition_matrix.T \
                + transition_covar \
                + ctrl_model.matrix()@ctrl_model.control_noise@ctrl_model.matrix().T
            x = transition_matrix@state_vector + ctrl_model.control_input()
            return x, p

        if t_index == 0:
            # case that it is a normal prediction
            # As this is Kalman-like, the control model must be capable of
            # returning a control matrix (B)

            transition_matrix = self._transition_matrix(
                prior=prior, time_interval=predict_over_interval, **kwargs)
            transition_covar = self.transition_model.covar(
                time_interval=predict_over_interval, **kwargs)

            # Prediction of the mean and covariance
            x_pred_m, p_pred_m = pred(transition_matrix, transition_covar,
                                      prior.state_vector, prior.covar)

            # Generation of the combined retrodiction matrices
            combined_retrodiction_matrices = self._generate_C_matrices(
                correlation_matrices, prior.ndim)
            # normal case
            x_pred = np.concatenate([x_pred_m, prior.multi_state_vector])
            W_P_column = np.array([c @ prior.covar for c in combined_retrodiction_matrices])
            correlated_column = np.reshape(W_P_column, (
                prior.multi_covar.shape[0], prior.ndim)) @ transition_matrix.T
            correlated_row = correlated_column.T

            p_top = np.hstack((p_pred_m, correlated_row))
            p_bottom = np.hstack((correlated_column, prior.multi_covar))
            p_pred = np.vstack((p_top, p_bottom))

            # add new correlation matrix with the present time step
            time_corr_matrices = correlation_matrices[timestamp_from_which_is_predicted]
            time_corr_matrices['P_pred'] = p_pred_m
            time_corr_matrices['F'] = transition_matrix
            time_corr_matrices['PFP'] = \
                time_corr_matrices['P'] \
                @ time_corr_matrices['F'].T \
                @ np.linalg.inv(time_corr_matrices['P_pred'])

        else:
            # case of out of sequence prediction case
            timestamp_m_plus_1 = prior.timestamps[prior.timestamps.index(
                timestamp_from_which_is_predicted) - 1]
            time_interval_m_to_m_plus_1 = timestamp_m_plus_1 - timestamp
            ndim = prior.ndim

            # transitions for timestamp m
            transition_matrix_m = self._transition_matrix(
                prior=prior, time_interval=predict_over_interval, **kwargs)
            transition_covar_m = self.transition_model.covar(
                time_interval=predict_over_interval, **kwargs)

            # transitions for timestamp m+1
            transition_matrix_m_plus_1 = self._transition_matrix(
                time_interval=time_interval_m_to_m_plus_1, **kwargs)
            transition_covar_m_plus_1 = self.transition_model.covar(
                time_interval=time_interval_m_to_m_plus_1, **kwargs)

            t2t_plus = slice(t_index * ndim, (t_index+1) * ndim)
            t_minus2t = slice((t_index-1) * ndim, t_index * ndim)
            # Normal Kalman-like prediction for the state m|m-1.
            x_m_minus_1_given_k = prior.multi_state_vector[t2t_plus]
            p_m_minus_1_given_k = prior.multi_covar[t2t_plus, t2t_plus]

            # prediction to the timestamp m|m-1
            x_pred_m, p_pred_m = pred(transition_matrix_m, transition_covar_m,
                                      x_m_minus_1_given_k, p_m_minus_1_given_k)
            # prediction to the timestamp m + 1|m-1
            x_pred_m_plus_1, p_pred_m_plus_1 = pred(transition_matrix_m_plus_1,
                                                    transition_covar_m_plus_1,
                                                    x_pred_m, p_pred_m)

            x_m_plus_1_given_k = prior.multi_state_vector[t_minus2t]
            x_diff = x_m_plus_1_given_k - x_pred_m_plus_1

            p_m_plus_1_given_k = prior.multi_covar[t_minus2t, t_minus2t]
            p_diff = p_m_plus_1_given_k - p_pred_m_plus_1

            W = p_pred_m @ transition_matrix_m_plus_1.T @ np.linalg.inv(p_pred_m_plus_1)
            x_pred_m = x_pred_m + W@x_diff
            p_pred_m_cr = p_pred_m + W@p_diff@W.T

            # build full state
            x_pred = np.concatenate([prior.multi_state_vector[:t_index * ndim],
                                     x_pred_m,
                                     prior.multi_state_vector[t_index * ndim:]])

            P_right_lower = prior.multi_covar[t_index * ndim:, t_index * ndim:]

            # add new correlation matrix with the present time step
            pred_from_corr_matrices = correlation_matrices[timestamp_from_which_is_predicted]
            pred_from_corr_matrices['P_pred'] = p_pred_m
            pred_from_corr_matrices['F'] = transition_matrix_m
            pred_from_corr_matrices['PFP'] = (
                    pred_from_corr_matrices['P'] @ transition_matrix_m.T @ np.linalg.inv(p_pred_m))

            correlation_matrices[timestamp] = {}
            correlation_matrices[timestamp]['F'] = transition_matrix_m_plus_1
            correlation_matrices[timestamp]['P_pred'] = p_pred_m_plus_1
            correlation_matrices[timestamp]['P'] = p_pred_m_cr
            correlation_matrices[timestamp]['PFP'] = \
                p_pred_m_cr @ transition_matrix_m_plus_1.T @ np.linalg.inv(p_pred_m_plus_1)

            # resort the dict
            correlation_matrices = OrderedDict(sorted(correlation_matrices.items(), reverse=True))

            # generate prediction matrix
            p_pred = np.zeros((ndim * (prior.nstep+1), ndim * (prior.nstep+1)))
            p_pred[(t_index+1) * ndim:, (t_index+1) * ndim:] = P_right_lower

            # get all timestamps which has to be recalculated beginning
            # with the newest one
            timestamps = sorted(prior.timestamps + [timestamp], reverse=True)
            timestamps_to_recalculate = [ts for ts in timestamps if ts >= timestamp]
            covars = \
                [prior.multi_covar[i * ndim:(i+1) * ndim, i * ndim:(i+1) * ndim]
                 for i in range(t_index)]
            covars.append(p_pred_m)

            for i, ts in enumerate(timestamps_to_recalculate):
                corrs = {k: v for k, v in correlation_matrices.items()
                         if k <= ts}
                combined_retrodiction_matrices = self._generate_C_matrices(corrs, ndim)
                combined_retrodiction_matrices = combined_retrodiction_matrices[1:]
                W_column = np.array([c @ covars[i] for c in combined_retrodiction_matrices])
                W_column = np.reshape(W_column, (ndim * len(combined_retrodiction_matrices), ndim))
                W_row = W_column.T

                i2i_plus = slice(i * ndim, (i + 1) * ndim)
                i_plus2end = slice((i+1) * ndim, None)
                # set covar
                p_pred[i2i_plus, i2i_plus] = covars[i]

                # set column
                p_pred[i_plus2end, i2i_plus] = W_column

                # set row
                p_pred[i2i_plus, i_plus2end] = W_row

        timestamps = sorted(prior.timestamps + [timestamp], reverse=True)
        # the act_timestamp parameter is used for the updater to
        # know for which timestamp the prediction is calculated
        predicted_state = ASDGaussianStatePrediction(
            multi_state_vector=x_pred, multi_covar=p_pred,
            correlation_matrices=correlation_matrices, timestamps=timestamps,
            max_nstep=prior.max_nstep, act_timestamp=timestamp)
        self.prune_state(predicted_state)
        return predicted_state

    def _generate_C_matrices(self, correlation_matrices, ndim):
        combined_retrodiction_matrices = [np.eye(ndim)]
        for item in islice(correlation_matrices.values(), 1, None):
            combined_retrodiction_matrices.append(
                combined_retrodiction_matrices[-1] @ item['PFP'])
        return combined_retrodiction_matrices

    def prune_state(self, predicted_state):
        r"""Simple ASD pruning function. Deletes timesteps from the multi
        state if it is longer then max_nstep

        Parameters
        ----------
        predicted_state : :class:`~.ASDState`
            :math:`\mathbf{x}_{k|k-1}`

        Returns
        -------
        : :class:`~.ASDState`
            :math:`\mathbf{x}_{k|k-1}`, the pruned state and the pruned
            state covariance :math:`P_{k|k-1}`

        """
        if predicted_state.nstep > predicted_state.max_nstep != 0:
            index = predicted_state.max_nstep * predicted_state.ndim
            predicted_state.multi_state_vector = \
                predicted_state.multi_state_vector[:index]
            predicted_state.multi_covar = \
                predicted_state.multi_covar[:index, :index]
            deleted_timestamps = \
                predicted_state.timestamps[predicted_state.max_nstep:]
            predicted_state.timestamps = \
                predicted_state.timestamps[:predicted_state.max_nstep]
            _ = [predicted_state.correlation_matrices.pop(el)
                 for el in deleted_timestamps]
            return predicted_state
        else:
            return predicted_state
