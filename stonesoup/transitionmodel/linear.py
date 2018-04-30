# -*- coding: utf-8 -*-
# import numpy as np

import scipy as np
from scipy.stats import multivariate_normal
from ..base import Property
from .base import TransitionModel
from ..types.model import LinearModel, GaussianModel, TimeVariantModel


class LinearTransitionModel(TransitionModel):
    """Linear Transition Model

    Transitions state using a transition matrix, and optionally accepts a
    control matrix and vector to apply to the new state.

    .. math::

        \hat{\mathbf{x}}_{k\mid k-1} &=
            \mathbf{F}_k \hat{\mathbf{x}}_{k-1\mid k-1} +
            \mathbf{B}_k \mathbf{u}_k

    """

    transition_matrix = Property(
        np.ndarray, doc="Linear transition matrix :math:`\mathbf{F}_k`.")
    control_matrix = Property(
        np.ndarray, default=None, doc="Control matrix :math:`\mathbf{B}_k`.")

    def transition(self, state_vector, control_vector=None):
        """Transition state

        Parameters
        ----------
        state_vector : StateVector
            State vector :math:`\hat{\mathbf{x}}_{k-1|k-1}`.
        control_vector : StateVector, optional
            Control vector :math:`\mathbf{u}_k`. Default is None in which case
            no control vector is applied.

        Returns
        -------
        StateVector
            New state vector :math:`\hat{\mathbf{x}}_{k\mid k-1}`.
        """
        new_state_vector = self.transition_matrix @ state_vector
        if control_vector is not None:
            new_state_vector += self.control_matrix @ control_vector
        return new_state_vector


class ConstantVelocity1D(TransitionModel, LinearModel,
                         GaussianModel, TimeVariantModel):
    r"""This is a class implementation of a time-variant 1D Linear-Gaussian
    Constant Velocity Transition Model.

    The target is assumed to move with (nearly) constant velocity, where
    target acceleration is model as white noise.

    The model is described by the following SDEs:

        .. math::
            :nowrap:

            \begin{eqnarray}
                dx_{pos} & = & x_{vel}*d & | {Position \ on \
                X-axis (m)} \\
                dx_{vel} & = & q*dW_t,\ W_t \sim N(0,q^2) & | Speed \ on \
                X-axis (m/s)
            \end{eqnarray}

    Or equivalently:

        .. math::
            x_t = F_t*x_{t-1} + w_t,\ w_t \sim N(0,Q_t)

    where:

        .. math::
            x & = & \begin{bmatrix}
                        x_{pos} \\
                        x_{vel}
                \end{bmatrix}

        .. math::
            F_t & = & \begin{bmatrix}
                        1 & t\\
                        0 & 1
                \end{bmatrix}

        .. math::
            Q_t & = & \begin{bmatrix}
                        \frac{t^3}{3} & \frac{t^2}{2} \\
                        \frac{t^2}{2} & t
                \end{bmatrix}*q

    Parameters
    ----------
    ndim_state : :class:`int`, read-only
        The number of state dimensions. *(constant = 2)*
    """

    noise_diff_coeff = Property(
        float, doc="The velocity noise diffusion coefficient :math:`q`")
    time_variant = Property(float,
                            doc="The value of the time variant :math:`t` in\
                            seconds (the default is 1)",
                            default=1)

    def __init__(self, noise_diff_coeff, time_variant, *args, **kwargs):
        """Constructor method """

        # TODO: Proper input validation

        super().__init__(noise_diff_coeff, time_variant, *args, **kwargs)

        self._ndim_state = 2

    def _transfer_matrix(self, t=None):
        if(t is None):
            t = self.time_variant
        return np.array([[1, t], [0, 1]])

    def _noise_covariance(self, t=None):
        if(t is None):
            t = self.time_variant
        return np.array([[np.power(t, 3)/3, np.power(t, 2)/2],
                         [np.power(t, 2)/2, t]])*self.noise_diff_coeff

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        ndim_state: int
            The number of model state dimension
        """

        return self._ndim_state

    def eval(self, state_vector=None, noise=False, time=None):
        """ Model transition function

        Propagates a given (set of) state(s)/particle(s) ``state_vector``
        through the dynamic model for time ``time``, with the application of
        random process noise ``noise``.

        In mathematical terms, this can be written as:

        .. math::

            x_t = f(x_{t-1},w_t,t),\ \ w_t \sim p(w_t)

        where :math:`x_t =` ``state_vector``, :math:`w_t =` ``noise`` and
        :math:`t` = ``time``.

        Parameters
        ----------
        state_vector : :class:`numpy.ndarray` of shape\
        (:py:attr:`~ndim_state`,Np)
            A (set of) prior state vector(s) :math:`x_{t-1}` from time
            :math:`t-1` (the default is None)

            - If None, the function will return the function transition \
              matrix :math:`F_t`. In this case, any value passed for ``noise``\
              will be ignored.

        noise : :class:`numpy.ndarray` of shape (:py:attr:`~ndim_state`,Np) or\
        boolean
            ``noise`` can be of two types:

            - A *numpy array* containing a (set of) externally generated \
              random process noise sample(s) :math:`w_t`; or

            - A *boolean*, which is set to True or False depending on\
              whether noise should be applied or not. If set to True, the\
              model's own noise distribution will be used to generate the\
              noise samples :math:`w_t`

            (the default in False, in which case process noise will be ignored)

        time : :class:`int`
            A time variant :math:`t` (the default is None, in which case it
            will be set equal to :py:attr:`~time_variant`)

        Returns
        -------
        state_vector_new : :class:`numpy.ndarray` of shape\
        (:py:attr:`~ndim_state`,Np)
            The (set of) predicted state vector(s) :math:`x_t`
        """

        # TODO: Proper input validation

        if time is None:
            time = self.time_variant

        if state_vector is None:
            state_vector = np.eye(self.ndim_state)
            noise = 0
        else:
            if issubclass(type(noise), bool) and noise:
                noise = self.random(Np=state_vector.shape[1], time=time)
            elif(issubclass(type(noise), bool) and not noise):
                noise = 0

        state_vector_new = self._transfer_matrix(time)@state_vector + noise
        return state_vector_new

    def covar(self, time=None):
        """Returns the transition model noise covariance matrix.

        Parameters
        ----------
        time : :class:`int`
            A time variant :math:`t` (the default is None, in which case it
            will be set equal to :py:attr:`~time_variant`)

        Returns
        -------
        covar: :class:`numpy.ndarray` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The process noise covariance.
        """

        # TODO: Proper input validation

        if(time is None):
            time = self.time_variant

        covar = self._noise_covariance(time)

        return covar

    def random(self, num_samples=1, time=None):
        """ Model noise/sample generation function

        Generates noisy samples from the transition model.

        In mathematical terms, this can be written as:

        .. math::

            w_t \sim \mathcal(N)(0,Q_t)

        where :math:`w_t =` ``noise``.

        Parameters
        ----------
        num_samples: :class:`int`, optional
            The number of samples to be generated (the default is 1)
        time : :class:`int`, optional
            A time variant :math:`t` (the default is None, in which case it
            will be set equal to :py:attr:`~time_variant`)

        Returns
        -------
        noise : :class:`numpy.ndarray` of shape\
        (:py:attr:`~ndim_state`,``num_samples``)
            A set of Np samples, generated from the model's noise distribution.
        """

        # TODO: Proper input validation

        if time is None:
            time = self.time_variant

        noise = multivariate_normal.rvs(
            np.array([0, 0]), self._noise_covariance(time), num_samples).T

        return noise

    def pdf(self, state_vector_trans, state_vector_prior, time=None):
        """ Model pdf/likelihood evaluation function

        Evaluates the pdf/likelihood of the (set of) transformed state
        vector(s)``state_vector_trans``, given the (set of) prior state
        vector(s) ``state_vector_prior``.

        In mathematical terms, this can be written as:

        .. math::

            p = p(x_t | x_{t-1}) = \mathcal{N}(x_t; x_{t-1}, Q_t)

        where :math:`x_t` = ``state_vector_trans``
        , :math:`x_{t-1}` = ``state_vector_prior`` and
        :math:`Q_t` = :py:attr:`~covar`.

        Parameters
        ----------
        state_vector_trans : :class:`numpy.ndarray` of shape\
        (:py:attr:`~ndim_state`, :math:`N_{p1}`)
            A (set of) predicted state vector(s) :math:`x_{t|t-1}`
        state_vector_prior : :class:`numpy.ndarray` of shape \
        (:py:attr:`~ndim_state`, :math:`N_{p2}`)
            A (set of) prior state vector(s) :math:`x_{t-1}` from time
            :math:`t-1`
        time : :class:`int`, optional
            A time variant :math:`t` (the default is None, in which case it
            will be set equal to :py:attr:`~time_variant`)

        Returns
        -------
        p: :class:`numpy.ndarray` of shape (:math:`N_{p1}`, :math:`N_{p2}`)
            A matrix of probabilities/likelihoods, where each element
            (:math:`i`, :math:`j`) corresponds to the likelihood of
            ``x_t[:,i]`` given ``x_tm1[:,j]``.
        """

        # TODO: 1) Optimise performance
        #       2) Proper input validation
        Np1 = state_vector_trans.shape[1]
        Np2 = state_vector_prior.shape[1]
        p = np.zeros((Np1, Np2))

        if time is None:
            time = self.time_variant

        for i in range(0, Np2):
            p[:, i] = multivariate_normal.pdf(
                state_vector_trans.T,
                mean=self.eval(state_vector_prior[:, i]).T,
                cov=self._noise_covariance(time)
            ).T
        return p
