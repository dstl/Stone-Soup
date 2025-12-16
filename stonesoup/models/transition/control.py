import numpy as np
from scipy.linalg import block_diag

from .base import TransitionModel
from .nonlinear import GaussianTransitionModel
from ..base import LinearModel
from ..control.base import ControlModel
from ...base import Property
from ...functions import jacobian as compute_jac
from ...types.state import State


class _AugmentedGaussianTransitionControlModel(GaussianTransitionModel):
    transition_model: TransitionModel = Property(doc="Transition model")
    control_model: ControlModel = Property(doc="Control model")
    control_transition_model: TransitionModel = Property(default=None)
    control_model_mapping: list[int] = Property(default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.control_model_mapping is None:
            self.control_model_mapping = list(
                range(self.transition_model.ndim,
                      self.transition_model.ndim + self.control_model.ndim)
            )

    @property
    def ndim_state(self):
        return self.transition_model.ndim + self._ndim_ctrl_est

    @property
    def _ndim_ctrl_est(self):
        if self.control_transition_model:
            return self.control_transition_model.ndim
        else:
            return self.control_model.ndim

    def _get_state_and_control(self, prior):
        state = State(
            state_vector=prior.state_vector[:self.transition_model.ndim, :],
            timestamp=prior.timestamp)
        control_input = State(
            state_vector=prior.state_vector[self.control_model_mapping, :],
            timestamp=prior.timestamp)
        return state, control_input

    def _control_matrix(self, prior=None, **kwargs):
        model_matrix = np.zeros((self.transition_model.ndim, self._ndim_ctrl_est))

        rows = np.array(list(range(self.transition_model.ndim)), dtype=np.intp)
        columns = np.array(self.control_model_mapping, dtype=np.intp) - self.transition_model.ndim

        if prior:
            state_input, _ = self._get_state_and_control(prior)
        else:
            state_input = None
        model_matrix[rows[:, np.newaxis], columns] = self.control_model.matrix(
            prior=state_input, **kwargs)

        return model_matrix


class LinearAugmentedGaussianTransitionControlModel(
        LinearModel, _AugmentedGaussianTransitionControlModel):

    def covar(self, **kwargs):
        Qt = self.transition_model.covar(**kwargs)
        B = self._control_matrix(**kwargs)
        if self.control_transition_model is None:
            Qtc = self.control_model.covar(**kwargs)
        else:
            Qtc = self.control_transition_model.covar(**kwargs)

        return block_diag(Qt + B@Qtc@B.T, Qtc)

    def matrix(self, **kwargs):
        F = self.transition_model.matrix(**kwargs)
        B = self._control_matrix(**kwargs)
        if self.control_transition_model is None:
            F_c = np.eye(self.control_model.ndim_ctrl)
        else:
            F_c = self.control_transition_model.matrix(**kwargs)

        return np.block([[F, B], [np.zeros(np.shape(B.T)), F_c]])


class NonLinearAugmentedGaussianTransitionControlModel(_AugmentedGaussianTransitionControlModel):

    def _control_model_jacobian(self, prior, **kwargs):
        def fun(state, **kwargs):
            state_input, control_input = self._get_state_and_control(state)
            return self.control_model.function(control_input, prior=state_input, **kwargs)

        jac = compute_jac(fun, prior, **kwargs)

        rows = np.array(list(range(self.transition_model.ndim)), dtype=np.intp)
        columns = np.array(self.control_model_mapping, dtype=np.intp) - self.transition_model.ndim
        model_matrix = np.zeros((self.transition_model.ndim, self._ndim_ctrl_est))
        model_matrix[rows[:, np.newaxis], columns] = jac[rows[:, np.newaxis], columns]

        return model_matrix

    def covar(self, prior, **kwargs):
        Qt = self.transition_model.covar(**kwargs)

        if isinstance(self.control_model, LinearModel):
            B = self._control_matrix(prior=prior, **kwargs)
        else:
            B = self._control_model_jacobian(prior, **kwargs)

        if self.control_transition_model is None:
            Qtc = self.control_model.covar(**kwargs)
        else:
            Qtc = self.control_transition_model.covar(**kwargs)

        return block_diag(Qt + B@Qtc@B.T, Qtc)

    def function(self, prior, noise=None, **kwargs):
        kwargs.pop('control_input', None)
        state_input, control_input = self._get_state_and_control(prior)

        Fx = self.transition_model.function(state_input, noise=noise, **kwargs)
        Bu = self.control_model.function(control_input, state_input, noise=noise, **kwargs)

        if self.control_transition_model is not None:
            control_input_est = State(
                state_vector=prior.state_vector[self.transition_model.ndim:, :],
                timestamp=prior.timestamp)
            control_input_est_vector = self.control_transition_model.function(
                control_input_est, **kwargs)
        else:
            control_input_est_vector = control_input.state_vector

        return np.vstack([Fx + Bu, control_input_est_vector])
