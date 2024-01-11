from functools import partial

import numpy as np
import scipy.linalg as la

from .base import Predictor
from ._utils import predict_lru_cache
from ..base import Property
from ..types.prediction import Prediction, SqrtGaussianStatePrediction
from ..models.base import LinearModel
from ..models.transition import TransitionModel
from ..models.transition.linear import LinearGaussianTransitionModel
from ..models.control import ControlModel
from ..models.control.linear import LinearControlModel
from ..functions import gauss2sigma, unscented_transform


class KalmanPredictor(Predictor):
    r"""A predictor class which forms the basis for the family of Kalman
    predictors. This class also serves as the (specific) Kalman Filter
    :class:`~.Predictor` class. Here transition and control models must be linear:

    .. math::

      f_k( \mathbf{x}_{k-1}, \mathbf{\nu}_k) &= F_k \mathbf{x}_{k-1} + \mathbf{\nu}_k , \
      \mathbf{\nu}_k \sim \mathcal{N}(0,Q_k)

      \ b_k( \mathbf{u}_k, \mathbf{\eta}_k) &= B_k (\mathbf{u}_k + \mathbf{\eta}_k), \
      \mathbf{\eta}_k \sim \mathcal{N}(0,\Gamma_k).

    Raises
    ------
    ValueError
        If no :class:`~.TransitionModel` is specified.

    """

    transition_model: LinearGaussianTransitionModel = Property(
        doc="The transition model to be used.")
    control_model: LinearControlModel = Property(
        default=None,
        doc="The control model to be used. Default `None` where the predictor "
            "will create a zero-effect linear :class:`~.ControlModel`.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # If no control model insert a linear zero-effect one
        if self.control_model is None:
            ndims = self.transition_model.ndim_state
            ndimc = 2  # No control exerted so this doesn't matter.
            self.control_model = LinearControlModel(np.zeros([ndims, ndimc]),
                                                    control_noise=np.zeros([ndimc, ndimc]))

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
        prior : :class:`~.GaussianState`
            The prior state, :math:`\mathbf{x}_{k-1}`

        **kwargs : various, optional
            These are passed to :meth:`~.LinearGaussianTransitionModel.matrix`

        Returns
        -------
        : :class:`~.State`
            The predicted state

        """
        return self.transition_model.matrix(**kwargs) @ prior.state_vector

    def _control_matrix(self, **kwargs):
        r"""Convenience function which returns the control matrix

        Returns
        -------
        : :class:`numpy.ndarray`
            control matrix, :math:`B_k`

        """
        return self.control_model.matrix(**kwargs)

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

    def _predicted_covariance(self, prior, predict_over_interval, control_input=None, **kwargs):
        """Private function to return the predicted covariance. Useful in that
        it can be overwritten in children.

        Parameters
        ----------
        prior : :class:`~.GaussianState`
            The prior class
        predict_over_interval : :class`~.timedelta`
            The interval over which the covariance is predicted
        control_input : :class:`~State`
            The input control vector (optional)

        Returns
        -------
        : :class:`~.CovarianceMatrix`
            The predicted covariance matrix

        """
        prior_cov = prior.covar
        trans_m = self._transition_matrix(prior=prior, time_interval=predict_over_interval,
                                          **kwargs)
        trans_cov = self.transition_model.covar(time_interval=predict_over_interval, **kwargs)

        # As this is Kalman-like, the control model must be capable of
        # returning a control matrix (B)
        ctrl_mat = self._control_matrix(control_input=control_input,
                                        time_interval=predict_over_interval, **kwargs)
        ctrl_noi = self.control_model.control_noise

        return trans_m @ prior_cov @ trans_m.T + trans_cov + ctrl_mat @ ctrl_noi @ ctrl_mat.T

    @predict_lru_cache()
    def predict(self, prior, timestamp=None, control_input=None, **kwargs):
        r"""The predict function

        Parameters
        ----------
        prior : :class:`~.State`
            :math:`\mathbf{x}_{k-1}`
        timestamp : :class:`datetime.datetime`, optional
            :math:`k`
        control_input : :class:`StateVector`, optional
            :math:`\mathbf{u}_k`
        **kwargs :
            These are passed, via :meth:`~.KalmanFilter.transition_function()` to
            :meth:`~.LinearGaussianTransitionModel.matrix()` and
            :meth:`~.LinearControlModel.function()`

        Returns
        -------
        : :class:`~.GaussianStatePrediction`
            :math:`\mathbf{x}_{k|k-1}`, the predicted state and the predicted
            state covariance :math:`P_{k|k-1}`

        """

        # Get the prediction interval
        predict_over_interval = self._predict_over_interval(prior, timestamp)

        # Prediction of the mean
        x_pred = self._transition_function(
            prior, time_interval=predict_over_interval, **kwargs) \
            + self.control_model.function(control_input, time_interval=predict_over_interval,
                                          **kwargs)

        # Prediction of the covariance
        p_pred = self._predicted_covariance(prior, predict_over_interval,
                                            control_input=control_input)

        # And return the state in the correct form
        return Prediction.from_state(prior, x_pred, p_pred, timestamp=timestamp,
                                     transition_model=self.transition_model)


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
    transition_model: TransitionModel = Property(doc="The transition model to be used.")
    control_model: ControlModel = Property(
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
            return self.transition_model.jacobian(prior, **kwargs)

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
        return self.transition_model.function(prior, **kwargs)

    def _control_matrix(self, control_input, **kwargs):
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
            return self.control_model.matrix(**kwargs)
        else:
            return self.control_model.jacobian(control_input, **kwargs)


class UnscentedKalmanPredictor(KalmanPredictor):
    """UnscentedKalmanFilter class

    The predict is accomplished by calculating the sigma points from the
    Gaussian mean and covariance, then putting these through the (in general
    non-linear) transition function, then reconstructing the Gaussian.
    """
    transition_model: TransitionModel = Property(doc="The transition model to be used.")
    control_model: ControlModel = Property(
        default=None,
        doc="The control model to be used. Default `None` where the predictor "
            "will create a zero-effect linear :class:`~.ControlModel`.")
    alpha: float = Property(
        default=0.5,
        doc="Primary sigma point spread scaling parameter. Default is 0.5.")
    beta: float = Property(
        default=2,
        doc="Used to incorporate prior knowledge of the distribution. If the "
            "true distribution is Gaussian, the value of 2 is optimal. "
            "Default is 2")
    kappa: float = Property(
        default=None,
        doc="Secondary spread scaling parameter. Default is calculated as "
            "3-Ns")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._time_interval = None

    def _transition_and_control_function(self, prior_state, **kwargs):
        r"""Returns the result of applying the transition and control functions
        for the unscented transform

        Parameters
        ----------
        prior_state_vector : :class:`~.State`
            Prior state vector
        **kwargs : various, optional
            These are passed to :class:`~.TransitionModel.function`

        Returns
        -------
        : :class:`numpy.ndarray`
            The combined, noiseless, effect of applying the transition and
            control
        """

        return self.transition_model.function(prior_state, **kwargs) + \
            self.control_model.function(**kwargs)

    @predict_lru_cache()
    def predict(self, prior, control_input=None, timestamp=None, **kwargs):
        r"""The unscented version of the predict step

        Parameters
        ----------
        prior : :class:`~.State`
            Prior state, :math:`\mathbf{x}_{k-1}`
        control_input: :class:`~.State`
            Control input vector, :math:`\mathbf{u}_k`
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
        ctrl_mat = self.control_model.matrix(time_interval=predict_over_interval, **kwargs)
        ctrl_noi = self.control_model.covar(**kwargs)
        total_noise_covar = \
            self.transition_model.covar(time_interval=predict_over_interval, **kwargs) \
            + ctrl_mat @ ctrl_noi @ ctrl_mat.T

        # Get the sigma points from the prior mean and covariance.
        sigma_point_states, mean_weights, covar_weights = gauss2sigma(
            prior, self.alpha, self.beta, self.kappa)

        # This ensures that function passed to unscented transform has the
        # correct time interval
        transition_and_control_function = partial(
            self._transition_and_control_function,
            control_input=control_input,
            time_interval=predict_over_interval)

        # Put these through the unscented transform, together with the total
        # covariance to get the parameters of the Gaussian
        x_pred, p_pred, _, _, _, _ = unscented_transform(
            sigma_point_states, mean_weights, covar_weights,
            transition_and_control_function, covar_noise=total_noise_covar
        )

        # and return a Gaussian state based on these parameters
        return Prediction.from_state(prior, x_pred, p_pred, timestamp=timestamp,
                                     transition_model=self.transition_model)


class SqrtKalmanPredictor(ExtendedKalmanPredictor):
    r"""The version of the Kalman predictor that operates on the square root parameterisation of
    the Gaussian state, :class:`~.SqrtGaussianState`.

    The prediction is undertaken in one of two ways. The default is to work in exactly the same
    way as the parent class, with the exception that the predicted covariance is
    subject to a Cholesky factorisation prior to initialisation of the :class:`~.SqrtGaussianState`
    output. The alternative, accessible via the :attr:`qr_method = True` flag, is to predict via a
    modified Gram-Schmidt process. See [1] for details.

    If transition and control models are possessed of the square root form of the covariance (as
    :attr:`sqrt_covar` in the case of the transition model and :attr:`sqrt_control_noise` for
    control models), then these are used directly. If not then they are created from the full
    matrices using the scipy.linalg :meth:`sqrtm()` method. (Unlike the Cholesky decomposition
    this works on positive semi-definite matrices, as well as positive definite ones.

    References
    ----------
    1. Maybeck, P.S. 1994, Stochastic Models, Estimation, and Control, Vol. 1, NavtechGPS,
       Springfield, VA.

    """
    qr_method: bool = Property(
        default=False,
        doc="A switch to do the prediction via a QR decomposition, rather than using a Cholesky "
            "decomposition.")

    # This predictor returns a square root form of the Gaussian state prediction
    _prediction_class = SqrtGaussianStatePrediction

    def _predicted_covariance(self, prior, predict_over_interval, control_input=None, **kwargs):
        """Private function to return the predicted covariance.

        Parameters
        ----------
        prior : :class:`~.SqrtGaussianState`
            The prior class (which carries the covariance in square root form)
        predict_over_interval : :class`~.timedelta`
            The interval over which the model is applied
        control_input : :class:`~State`
            The input control vector (optional)

        Returns
        -------
        : :class:`~.CovarianceMatrix`
            The predicted covariance matrix

        """
        sqrt_prior_cov = prior.sqrt_covar

        trans_m = self._transition_matrix(prior=prior, time_interval=predict_over_interval,
                                          **kwargs)
        try:
            sqrt_trans_cov = self.transition_model.sqrt_covar(time_interval=predict_over_interval,
                                                              **kwargs)
        except AttributeError:
            sqrt_trans_cov = la.sqrtm(self.transition_model.covar(
                time_interval=predict_over_interval, **kwargs))

        # As this is Kalman-like, the control model must be capable of returning a control matrix
        # (B)
        ctrl_mat = self._control_matrix(control_input=control_input,
                                        time_interval=predict_over_interval)
        try:
            sqrt_ctrl_noi = self.control_model.sqrt_covar(time_interval=predict_over_interval,
                                                          **kwargs)
        except AttributeError:
            sqrt_ctrl_noi = la.sqrtm(self.control_model.covar(time_interval=predict_over_interval,
                                                              **kwargs))

        if self.qr_method:
            # Note that the control matrix aspect of this hasn't been tested
            m_sq_trans_cov = np.block([[trans_m @ sqrt_prior_cov, sqrt_trans_cov,
                                        ctrl_mat@sqrt_ctrl_noi]])
            _, pred_sqrt_cov = np.linalg.qr(m_sq_trans_cov.T)
            return pred_sqrt_cov.T
        else:
            return np.linalg.cholesky(trans_m@sqrt_prior_cov@sqrt_prior_cov.T@trans_m.T +
                                      sqrt_trans_cov@sqrt_trans_cov.T +
                                      ctrl_mat@sqrt_ctrl_noi@sqrt_ctrl_noi.T@ctrl_mat.T)
