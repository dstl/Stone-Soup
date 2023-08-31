import copy
from typing import Sequence
import numpy as np
from scipy.linalg import block_diag

from ...types.array import StateVector, StateVectors
from .base import TransitionModel
from ..base import GaussianModel, TimeVariantModel
from ...base import Property
from ...types.array import CovarianceMatrix


class GaussianTransitionModel(TransitionModel, GaussianModel):
    pass


class ConstantTurn(GaussianTransitionModel, TimeVariantModel):
    r"""This is a class implementation of a discrete, time-variant 2D Constant
    Turn Model.

    The target is assumed to move with (nearly) constant velocity and also
    an unknown (nearly) constant turn rate.

    The model is described by the following SDEs:

        .. math::
            :nowrap:

            \begin{align}
                dx_{pos} & =  x_{vel} d  \quad | {Position \ on \
                X-axis (m)} \\
                dx_{vel} & = -\omega y_{pos} d \quad | {Speed \
                on\ X-axis (m/s)} &\\
                dy_{pos} & =  y_{vel} d  \quad | {Position \ on \
                Y-axis (m)} \\
                dy_{vel} & =  \omega x_{pos} d \quad | {Speed \
                on\ Y-axis (m/s)} \\
                d\omega & = q_\omega dt  \quad | {Position \ on \ X,Y-axes (rad/sec)}
            \end{align}

    Or equivalently:

        .. math::
            x_t = F_t x_{t-1} + w_t,\ w_t \sim \mathcal{N}(0,Q_t)

    where:

        .. math::
            x & = & \begin{bmatrix}
                        x_{pos} \\
                        x_{vel} \\
                        y_{pos} \\
                        y_{vel} \\
                        \omega
                    \end{bmatrix}

        .. math::
            F(x) & = & \begin{bmatrix}
                          x+ \frac{x_{vel}}{\omega}\sin\omega dt -
                              \frac{y_{vel}}{\omega}(1-\cos\omega dt) \\
                          x_{vel}\cos\omega dt - y_{vel}\sin\omega dt \\
                          y+ \frac{v_{vel}}{\omega}\sin\omega dt +
                              \frac{x_{vel}}{\omega}(1-\cos\omega dt) \\
                          x_{vel}\sin\omega dt + y_{vel}\sin\omega dt \\
                          \omega
                      \end{bmatrix}

        .. math::
             Q_t & = & \begin{bmatrix}
                          \frac{dt^3q_x^2}{3} & \frac{dt^2q_x^2}{2} & 0 & 0 & 0 \\
                          \frac{dt^2q_x^2}{2} & dtq_x^2 & 0 & 0 & 0 \\
                          0 & 0 & \frac{dt^3q_y^2}{3} & \frac{dt^2q_y^2}{2} & 0 \\
                          0 & 0 & \frac{dt^2q_y^2}{2} & dtq_y^2 & 0 \\
                          0 & 0 & 0 & 0 & q_\omega^2
                     \end{bmatrix}
    """
    linear_noise_coeffs: np.ndarray = Property(
        doc=r"The acceleration noise diffusion coefficients :math:`[q_x, \: q_y]^T`")
    turn_noise_coeff: float = Property(
        doc=r"The turn rate noise coefficient :math:`q_\omega`")

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of combined model state dimensions.
        """
        return 5

    def function(self, state, noise=False, **kwargs) -> StateVector:
        time_interval_sec = kwargs['time_interval'].total_seconds()
        sv1 = state.state_vector
        turn_rate = sv1[4, :]
        # Avoid divide by zero in the function evaluation
        if turn_rate.dtype != float:
            turn_rate = turn_rate.astype(float)
        turn_rate[turn_rate == 0.] = np.finfo(float).eps
        dAngle = turn_rate * time_interval_sec
        cos_dAngle = np.cos(dAngle)
        sin_dAngle = np.sin(dAngle)
        sv2 = StateVectors(
            [sv1[0, :] + sin_dAngle/turn_rate * sv1[1, :] - sv1[3, :] / turn_rate *
             (1. - cos_dAngle),
             sv1[1, :] * cos_dAngle - sv1[3, :] * sin_dAngle,
             sv1[1, :] / turn_rate * (1. - cos_dAngle) + sv1[2, :] + sv1[3, :] * sin_dAngle
             / turn_rate,
             sv1[1, :] * sin_dAngle + sv1[3, :] * cos_dAngle,
             turn_rate])
        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0
        return sv2 + noise

    def covar(self, time_interval, **kwargs):
        """Returns the transition model noise covariance matrix.

        Returns
        -------
        : :class:`stonesoup.types.state.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The process noise covariance.
        """
        q_x, q_y = self.linear_noise_coeffs
        q = self.turn_noise_coeff
        dt = time_interval.total_seconds()

        Q = np.array([[dt**3 / 3., dt**2 / 2.],
                      [dt**2 / 2., dt]])
        C = block_diag(Q*q_x**2, Q*q_y**2, q**2)

        return CovarianceMatrix(C)


class ConstantTurnSandwich(ConstantTurn):
    r"""This is a class implementation of a time-variant 2D Constant Turn
    Model. This model is used, as opposed to the normal :class:`~.ConstantTurn`
    model, when the turn occurs in 2 dimensions that are not adjacent in the
    state vector, eg if the turn occurs in the x-z plane but the state vector
    is of the form :math:`(x,y,z)`. The list of transition models are to be
    applied to any state variables that lie in between, eg if for the above
    example you wanted the y component to move with constant velocity, you
    would put a :class:`~.ConstantVelocity` model in the list.

    The target is assumed to move with (nearly) constant velocity and also
    unknown (nearly) constant turn rate.
    """
    model_list: Sequence[GaussianTransitionModel] = Property(
        doc="List of Transition Models.")

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of combined model state dimensions.
        """
        return sum(model.ndim_state for model in self.model_list) + 5

    def function(self, state, noise=False, **kwargs) -> StateVector:
        state_tmp = copy.copy(state)
        sv_in = state.state_vector
        sv1 = np.concatenate((sv_in[0:2, 0:], sv_in[-3:, 0:]))
        state_tmp.state_vector = sv1
        # Calculate state vector for CT model
        sv_ct = super().function(state_tmp, noise=False, **kwargs)

        # Calculate state vector for model list
        idx1 = 2
        sv_list = [sv_ct[0:2, 0:]]
        for model in self.model_list:
            idx2 = idx1 + model.ndim
            state_tmp.state_vector = sv_in[idx1:idx2, 0:]
            sv_list.append(model.function(state_tmp, noise=False, **kwargs))
            idx1 = idx2
        sv_list.append(sv_ct[-3:, 0:])
        sv_out = StateVectors(np.concatenate(sv_list))
        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0
        return sv_out + noise

    def covar(self, time_interval, **kwargs):
        """Returns the transition model noise covariance matrix.

        Returns
        -------
        : :class:`stonesoup.types.state.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The process noise covariance.
        """
        C_t = np.zeros([self.ndim, self.ndim])
        C_ct = super().covar(time_interval, **kwargs)
        covar_list = [model.covar(time_interval) for model in self.model_list]

        # Assemble diag block components
        C_t[2:-3, 2:-3] = block_diag(*covar_list)
        C_t[0:2, 0:2] = C_ct[0:2, 0:2]
        C_t[-3:, -3:] = C_ct[-3:, -3:]
        # Reorder offdiagonal elements
        C_t[0:2:, -3:] = C_ct[0:2, -3:]
        C_t[-3:, 0:2] = C_ct[-3:, 0:2]

        return CovarianceMatrix(C_t)
