import copy

import numpy as np

from ._utils import predict_lru_cache
from .kalman import KalmanPredictor
from ..base import Property
from ..kernel import Kernel, QuadraticKernel
from ..types.prediction import Prediction
from ..types.state import State
from ..types.update import KernelParticleStateUpdate


class AdaptiveKernelKalmanPredictor(KalmanPredictor):
    """AdaptiveKernelKalmanFilter class
    """
    kernel: Kernel = Property(
        default=None,
        doc="Kernel. Default is None. If None, the default `QuadraticKernel` is used.")
    lambda_predictor: float = Property(
        default=1e-3,
        doc=r":math:`\lambda_{\tilde{K}}`. Regularisation parameter used to stabilise the inverse "
            r"Gram matrix. Range is :math:`\left[10^{-4}, 10^{-2}\right]`")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.kernel is None:
            self.kernel = QuadraticKernel()

    @predict_lru_cache()
    def predict(self, prior, timestamp=None, proposal=None, **kwargs):
        r"""The adaptive kernel version of the predict step

        Parameters
        ----------
        prior : :class:`~.KernelParticleState`
            Prior state, :math:`\mathbf{x}_{k-1}`
        timestamp : :class:`datetime.datetime`
            Time to transit to (:math:`k`)
        proposal : :class:`~.KernelParticleState`
            Proposal state, :math:`\mathbf{x}_{k-1}`
        **kwargs : various, optional
            These are passed to :meth:`~.TransitionModel.covar`

        Returns
        -------
        : :class:`~.KernelParticleStatePrediction`
            The predicted state :math:`\mathbf{x}_{k|k-1}` and the predicted
            state covariance :math:`P_{k|k-1}`
        """
        if proposal is None:
            if isinstance(prior, KernelParticleStateUpdate):
                proposal = State(state_vector=prior.proposal)
            else:
                proposal = copy.copy(prior)

        # Get the prediction interval
        predict_over_interval = self._predict_over_interval(prior, timestamp)
        new_state_vector = self.transition_model.function(
            proposal,
            time_interval=predict_over_interval,
            **kwargs)

        k_tilde_tilde = self.kernel(proposal)
        k_tilde_nontilde = self.kernel(proposal, prior)

        I = np.identity(len(prior))  # noqa: E741
        inv_val = np.linalg.pinv(k_tilde_tilde + self.lambda_predictor * I)

        kernel_t = inv_val @ k_tilde_nontilde
        prediction_weights = kernel_t @ prior.weight
        new_val = inv_val @ k_tilde_tilde - I
        v = new_val@new_val.T / len(prior)

        prediction_covariance = kernel_t @ prior.kernel_covar @ kernel_t.T + v
        return Prediction.from_state(prior,
                                     state_vector=new_state_vector,
                                     weight=np.squeeze(prediction_weights),
                                     kernel_covar=prediction_covariance,
                                     timestamp=timestamp,
                                     transition_model=self.transition_model)
