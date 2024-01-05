import numpy as np

from ._utils import predict_lru_cache
from ..base import Property
from .kalman import KalmanPredictor
from ..types.prediction import Prediction
from ..models.transition.linear import LinearGaussianTransitionModel
from ..models.control.linear import LinearControlModel


class InformationKalmanPredictor(KalmanPredictor):
    r"""A predictor class which uses the information form of the Kalman filter. The key concept is
    that 'information' is encoded as the information matrix, and the so-called 'information state',
    which are:

      .. math::

        Y_{k-1} &= P^{-1}_{k-1}

        \mathbf{y}_{k-1} &= P^{-1}_{k-1} \mathbf{x}_{k-1}

    The prediction then proceeds as [#]_

      .. math::

        Y_{k|k-1} &= [F_k Y_{k-1}^{-1} F^T + Q_k + B_k \Gamma_k B_k^T]^{-1}

        \mathbf{y}_{k|k-1} &= Y_{k|k-1} F_k Y_{k-1}^{-1} \mathbf{y}_{k-1} + Y_{k|k-1}
        B_k\mathbf{u}_k

    where the symbols have the same meaning as in the description of the Kalman filter
    (see e.g. tutorial 1) and the prediction equations can be derived from those of the Kalman
    filter. As it stands the :class:`~ControlModel` must be capable of returning a (perhaps
    linearised) control matrix and a control noise covariance matrix. In order to cut down on
    the number of matrix inversions and to benefit from caching these can be recast as [#]_

      .. math::

        M_k &= (F_k^{-1})^T Y_{k-1} F_k^{-1}

        C_k &= (I + M_k Q_k + M_k B_k \Gamma_k B_k^T)^{-1}

        Y_{k|k-1} &= C_k M_k

        \mathbf{y}_{k|k-1} &= C_k (F_k^{-1})^T \mathbf{y}_k + Y_{k|k-1} B_k\mathbf{u}_k

    The prior state must have a state vector :math:`\mathbf{y}_{k-1}` corresponding to
    :math:`P_{k-1}^{-1} \mathbf{x}_{k-1}` and a precision matrix, :math:`Y_{k-1} = P_{k-1}^{-1}`.
    The :class:`~.InformationState` class is provided for this purpose.

    The :class:`~.TransitionModel` is queried for the existence of an
    :meth:`inverse_matrix()` method, and if not present, :meth:`matrix()` is inverted. This gives
    one the opportunity to cache :math:`F_k^{-1}` and save computational resource.

    Raises
    ------
    ValueError
        If no :class:`~.TransitionModel` is specified.

    References
    ----------
    .. [#] Kim, Y-S, Hong, K-S 2003, Decentralized information filter in federated form, SICE
        annual conference

    .. [#] Makarenko, A., Durrant-Whyte, H. 2004, Decentralized data fusion and control in active
        sensor networks, in: The 7th International Conference on Information Fusion (Fusion'04),
        pp. 479-486

    """
    transition_model: LinearGaussianTransitionModel = Property(
        doc="The transition model to be used.")
    control_model: LinearControlModel = Property(
        default=None,
        doc="The control model to be used. Default `None` where the predictor "
            "will create a zero-effect linear :class:`~.ControlModel`.")

    def _inverse_transition_matrix(self, **kwargs):
        """Return the inverse of the transition matrix

        Parameters
        ----------
        **kwargs : various, optional
            These are passed to :meth:`~.LinearGaussianTransitionModel.matrix`

        Returns
        -------
        : :class:`numpy.ndarray`
            The inverse of the transition matrix, :math:`F_k^{-1}`

        """
        if hasattr(self.transition_model, 'inverse_matrix'):
            inv_transition_matrix = self.transition_model.inverse_matrix(**kwargs)
        else:
            inv_transition_matrix = np.linalg.inv(self.transition_model.matrix(**kwargs))

        return inv_transition_matrix

    def _transition_function(self, prior, **kwargs):
        r"""Applies the linear transition function to a single vector in the
        absence of a control input, returns a single predicted state. Because in this instance
        prior is an information state and :attr:`state_vector` is :math:`\mathbf{y}_{k-1}` we
        must recover :math:`\mathbf{x}_{k-1} = Y_{k-1}^{-1} \mathbf{y}_{k-1}`.

        This method included for completeness. It's not likely to be used.

        Parameters
        ----------
        prior : :class:`~.InformationState`
            The prior state

        **kwargs : various, optional
            These are passed to :meth:`~.LinearGaussianTransitionModel.matrix`

        Returns
        -------
        : :class:`~.StateVector`
            The predicted state vector

        """
        prior_state_mean = np.linalg.inv(prior.precision) @ prior.state_vector
        return self.transition_model.matrix(**kwargs) @ prior_state_mean

    @predict_lru_cache()
    def predict(self, prior, timestamp=None, control_input=None, **kwargs):
        r"""The predict function

        Parameters
        ----------
        prior : :class:`~.InformationState`
            :math:`\mathbf{y}_{k-1}, Y_{k-1}`
        timestamp : :class:`datetime.datetime`, optional
            :math:`k`
        control_input : :class:`StateVector`, optional
            :math:`u`
        **kwargs :
            These are passed, via :meth:`~.transition_model.transition_function` to
            :meth:`~.LinearGaussianTransitionModel.matrix`

        Returns
        -------
        : :class:`~.InformationStatePrediction`
            :math:`\mathbf{y}_{k|k-1}`, the predicted information state and the predicted
            information matrix :math:`Y_{k|k-1}`

        """

        # Get the prediction interval
        predict_over_interval = self._predict_over_interval(prior, timestamp)

        # As this is Kalman-like, the control model must be capable of
        # returning a control matrix (B)
        inverse_transition_matrix = self._inverse_transition_matrix(
            prior=prior, time_interval=predict_over_interval, **kwargs)
        transition_covar = self.transition_model.covar(
            time_interval=predict_over_interval, **kwargs)

        # control noise doesn't appear in the information matrix literature. It's incorporation
        # here depends on the model being able to return a matrix and covariance.
        control_matrix = self._control_matrix(time_interval=predict_over_interval, **kwargs)
        control_noise = self.control_model.covar(time_interval=predict_over_interval, **kwargs)
        control_covar = control_matrix @ control_noise @ control_matrix.T

        Mk = inverse_transition_matrix.T @ prior.precision @ inverse_transition_matrix
        Ck = np.linalg.inv(np.eye(prior.ndim) + Mk @ transition_covar + Mk @ control_covar)
        pred_info_matrix = Ck @ Mk
        pred_info_state = Ck @ inverse_transition_matrix.T @ prior.state_vector + \
            pred_info_matrix @ self.control_model.function(control_input,
                                                           time_interval=predict_over_interval,
                                                           **kwargs)

        return Prediction.from_state(prior, pred_info_state, pred_info_matrix, timestamp=timestamp,
                                     transition_model=self.transition_model)
