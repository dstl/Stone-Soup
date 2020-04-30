# -*- coding: utf-8 -*-

import numpy as np
from functools import lru_cache

from ..base import Property
from .base import Predictor
from ..types.prediction import InformationStatePrediction
from ..types.update import InformationStateUpdate  # To check incoming "prior" data
from ..types.state import InformationState  # To check incoming "prior" data
from ..models.transition.linear import LinearGaussianTransitionModel
from ..models.control.linear import LinearControlModel

from numpy.linalg import inv


class InfoFilterPredictor(Predictor):
    r"""A predictor class which forms the basis for the family of Kalman
    predictors. This class also serves as the (specific) Kalman Filter
    :class:`~.Predictor` class. Here the predictor model is updated from the base class

    .. math::

        \mathbf{y}_{k|k-1} = f_k(\mathbf{y}_{k-1}, \mathbf{\nu}_k) +
        b_k(\mathbf{u}_k, \mathbf{\eta}_k)

    where :math:`\mathbf{x}_{k|k-1}` has been replaced by an information-theoretic state
    :math:`\mathbf{y}_{k|k-1}`. Thus the predictor class implements

    .. math::

      f_k( \mathbf{y}_{k-1}) = [I_{identity} - \Omega_k G^T_k] F^{-T}_k \mathbf{y}_{k-1|k-1}, \
      b_k( \mathbf{y}_k) = Y_k B_k \mathbf{y}_{k|k-1} \ \mathrm{and} \ \mathbf{\nu}_k \sim
      \mathcal{N}(0,Q_k)

    where :math:`I_{identity}` is the identity matrix.

    The information prediction gain (analogous to the Kalman gain) is then calculated as,

    .. math::

        \Omega_k = M_k G_k \Sigma^{-1}_k

    where G is the "information observation" matrix which is assumed to be set to the identity
    matrix in this implementation.

    Notes
    -----
    In the Information filter (similar to the Kalman filter), transition and control models must be
    linear. Accepts InformationStateUpdate.


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

    def _noise_transition_matrix(self, **kwargs):
        """Return the noise transition matrix

        Parameters
        ----------
        **kwargs : various, optional
            These are passed to :meth:`~.LinearGaussianTransitionModel.matrix`

        Returns
        -------
        : :class:`numpy.ndarray`
            The noise transition matrix, :math:`G`

        """
        return np.identity(self.transition_model.ndim_state)

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
            The prior state, :math:`\mathbf{y}_{k-1}`

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
            :math:`\mathbf{y}_{k-1}`
        timestamp : :class:`datetime.datetime`, optional
            :math:`k`
        **kwargs :
            These are passed, via :meth:`~.InformationFilter.transition_function` to
            :meth:`~.LinearGaussianTransitionModel.matrix`

        Returns
        -------
        : :class:`~.State`
            :math:`\mathbf{y}_{k|k-1}`, the predicted state and the predicted
            Fisher information matrix :math:`Y_{k|k-1}`

        """

        # Get the prediction interval
        predict_over_interval = self._predict_over_interval(prior, timestamp)

        # As this is Kalman-like, the control model must be capable of
        # returning a control matrix (B)

        transition_matrix = self._transition_matrix(
            prior=prior, time_interval=predict_over_interval, **kwargs)
        transition_covar = self.transition_model.covar(
            time_interval=predict_over_interval, **kwargs)

#        control_matrix = self._control_matrix
#        control_noise = self.control_model.control_noise

        # p_pred = transition_matrix @ prior.info_matrix @ transition_matrix.T \
        #          + transition_covar \
        #          + control_matrix @ control_noise @ control_matrix.T

        ndims = self.transition_model.ndim

        G = self._noise_transition_matrix()
        F = transition_matrix
        # Q = transition_covar  # transition covar - not sure about this though
        if isinstance(prior, InformationStateUpdate)\
                or isinstance(prior, InformationStatePrediction)\
                or isinstance(prior, InformationState):
            Y = prior.info_matrix  # fisher information
        else:
            Y = prior.covar

        M = inv(transition_matrix.T) @ Y @ inv(transition_matrix)  # Eq 252

        Sigma = G.T @ M @ G + inv(transition_covar)  # Eq 254

        Omega = M @ G @ inv(Sigma)  # Eq 253

        Y_pred = M - Omega @ Sigma @ Omega.T  # Eq 251

        # Get the information state
        y = prior.state_vector
        y_pred = (np.identity(ndims)
                  - Omega @ G.T) @ inv(F.T) @ y + Y @ self.control_model.control_input()

        # Wikipedia method
        # M = inv(F).T @ Y @ inv(F)
        # C = M @ inv(M + inv(Q))
        # L = np.ones((ndims, ndims))-C
        # Y_pred = L @ M @ L.T + C @ inv(Q) @ C.T
        # y_pred = L @ inv(F.T) @ y
        # End wikipedia method

        return InformationStatePrediction(y_pred, Y_pred, timestamp=timestamp)
