import copy
from collections.abc import Sequence
from turtle import st

import numpy as np

from .base import TransitionModel
from ..base import GaussianModel, TimeVariantModel
from ...base import Property
from ...functions import block_diag
from ...types.array import CovarianceMatrix, StateVector, StateVectors


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
                          1 & \frac{\sin\omega dt}{\omega} & 0 & -
                            \frac{(1-\cos\omega dt)}{\omega} & 0 \\
                          0 & \cos\omega dt & 0 & - \sin\omega dt & 0 \\
                          0 & \frac{(1-\cos\omega dt)}{\omega} & 1 &
                            \frac{\sin\omega dt}{\omega} & 0 \\
                          0 & \sin\omega dt & 0 & \sin\omega dt & 0 \\
                          0 & 0 & 0 & 0 & 1
                      \end{bmatrix}

        .. math::
             Q_t & = & \begin{bmatrix}
                          q_x\frac{dt^3}{3} & q_x\frac{dt^2}{2} & 0 & 0 & 0 \\
                          q_x\frac{dt^2}{2} & q_xdt & 0 & 0 & 0 \\
                          0 & 0 & q_y\frac{dt^3}{3} & q_y\frac{dt^2}{2} & 0 \\
                          0 & 0 & q_y\frac{dt^2}{2} & q_ydt & 0 \\
                          0 & 0 & 0 & 0 & q_\omega dt
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
        dt = abs(time_interval.total_seconds())

        Q = np.array([[dt**3 / 3., dt**2 / 2.],
                      [dt**2 / 2., dt]])
        C = block_diag(Q*q_x, Q*q_y, dt*q)

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
    

class DecayTransition(TransitionModel):
    r"""This transition model simulates the decay of an active substance. During transition, a
    certain fraction of the remaining active components will decay. This depends on the half life.
    The process is stochastic with the probability of an individual decay within a time interval
    $\Delta t$ being

    .. math::

        p(1 \rightarrow 0) = 1 - \exp -\lamba \Delta t

    where $\lambda$ is the decay constant, related to the half life, $T$ via,

    .. math::

        \lambda = \frac{\log_e (2)}{T} 

    """
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The dimension is singular being a binary undecayed/decayed (1/0) indicator. 
        """
        return 1

    def decay_const(self, state):
        r"""Returns the decay constant calculated from the half life provided in the states

        Returns
        -------
        : :class:`float`
            The decay constant, $\lambda$.
        """
        # TODO: check and handle that this property exists?
        return np.log(2)/state.halflife.total_seconds()

    def prob_decay(self, state, time_interval):
        r"""Probability of a decay within a given time interval

        Returns
        -------
        : :class:`float`
            The probability of a decay within the time interval, $\Delta t$.
        """
        return 1 - np.exp(-self.decay_const(state)*time_interval.total_seconds())

    def function(self, state,  time_interval):
        """Returns a state vector after decay. A stochastic process, unless the seed is set
        (TODO: set seed?).

        Parameters
        ----------
        state : :class:`~.State`
            The state which will transition
        time_interval : :class:`~.timedelta`
            The time interval over which to transition the state

        Returns
        -------
        : :class:`~.StateVector`
            The state vector after decay
        """
        out_state_vector = copy.copy(state.state_vector)
        for i, element in enumerate(state.state_vector):
            if element == 1 and np.random.uniform() < self.prob_decay(state, time_interval):
                out_state_vector[i] *= 0

        return out_state_vector

    def covar(self, state,  time_interval):
        """Returns the covariance of the state vector over a defined time interval. Not a
        stochastic process; only depends onthe half life and the time interval. This is a single
        number repeated down the diagonal for those elements that are undecayed.

        Parameters
        ----------
        state : :class:`~.State`
            The state from which to calculate the covariance
        time_interval : :class:`~.timedelta`
            The time interval over which to calculate the covariance

        Returns
        -------
        : :class:`stonesoup.types.state.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The covariance matrix of the state vector

        """
        covar = self.prob_decay(state, time_interval) * (1 - self.prob_decay(state, time_interval))

        return CovarianceMatrix(np.diag(state.state_vector*covar))

    def pdf(self, state_fin, state_ini, time_interval):
        """What's the probability of arriving at a particular state given the current state
        and time interval? There's likely a quicker way to do this based on summing the vectors
        directly rather than looping, but this is a simple implementation that's not likely to be
        used often.

        Parameters
        ----------
        state_fin : :class:`~.State`
            The state to which the transition will be made
        state_ini : :class:`~.State`
            The state from which the transition was made
        time_interval : :class:`~.timedelta`
            The time interval over which to calculate the probability of transition

        Returns
        -------
        : :class:`float`
            The probability of transitioning from state_ini to state_fin over the time interval

        Note
        ----
        This calculates the probability of transitioning from a partiular distribution of decayed
        elements to another particular distribution of decayed elements. It's therefore likely to
        be a very small number, especially for large state vectors. It's not the same as
        calculating the probability of a particular number of decays, which may be a more useful
        metric and is goverend by standard binomial statistics.

        """
        prob = 1
        for i, element in enumerate(state_ini.state_vector):
            if element == 1:
                if state_fin.state_vector[i] == 1:
                    prob *= (1 - self.prob_decay(state_ini, time_interval))
                else:
                    prob *= self.prob_decay(state_ini, time_interval)
            else:
                if state_fin.state_vector[i] == 1:
                    # This should never happen, but if it does, we'd want to kill the loop
                    prob *= 0.
                else:
                    prob *= 1

        return prob

    def rvs(self, state, time_interval, num_samples=1):
        """Generate a random sample from the transition distribution. This is equivalent to
        calling the function method n times, but is provided for consistency with other transition
        models. The random seed cannnot yet be set.

        """
        samples = []
        for i in range(num_samples):
            samples.append(self.function(state, time_interval))

        return StateVectors(samples)
