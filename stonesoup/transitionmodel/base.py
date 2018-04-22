# -*- coding: utf-8 -*-
from ..base import Base, Property
from abc import abstractmethod, abstractproperty
import logging

# import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal


class TransitionModel(Base):
    """Transition Model base class

    Paramaters
    ----------
    ndim_state: int
        The number of state dimensions

        - Constant for each model
    """

    @abstractproperty
    def ndim_state(self):
        """ Number of state dimensions"""
        pass

    @abstractmethod
    def eval(self):
        """ Model transition function """
        pass

    @abstractmethod
    def random(self):
        """ Model noise/sample generation function """
        pass

    @abstractmethod
    def pdf(self):
        """ Model pdf/likelihood evaluation function """
        pass


class ConstantVelocity1D(TransitionModel):
    r"""This is a class implementation of a time-varying 1D Linear-Gaussian
    Constant Velocity Dynamic Model.

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
    noise_diff_coeff: scalar
        The velocity noise diffusion coefficient :math:`q`
    ndim_state : scalar, constant
        The number of state dimensions. *(constant = 2)*
    time_variant: scalar, optional
        The value of the time variant :math:`t` in seconds (the default is 1)

    """

    noise_diff_coeff = Property(float, doc="noise diffusion coefficient")
    time_variant = Property(float, default=1)

    def __init__(self, noise_diff_coeff, time_variant, *args, **kwargs):
        """Constructor method """

        # TODO: Proper input validation

        # Create logger
        self._logger = logging.getLogger(__name__)
        self._logger = logging.LoggerAdapter(
            self._logger, {'classname': self.__class__.__name__})

        super().__init__(noise_diff_coeff, time_variant, *args, **kwargs)

        # self.noise_diff_coeff = noise_diff_coeff
        # self.time_variant = time_variant

        # Definition of F(t) and Q(t)
        self._F = lambda t=self.time_variant: sp.array(
            [[1, t], [0, 1]])
        self._Q = lambda t=self.time_variant: sp.array(
            [[sp.power(t, 3)/3, sp.power(t, 2)/2],
             [sp.power(t, 2)/2, t]])*self.noise_diff_coeff

        self._ndim_state = 2

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        ndim_state: int
            The number of model state dimension
        """

        return self._ndim_state

    def eval(self, x_tm1=None, noise=False, time=None):
        r""" Model transition function

        Propagates a given (set of) state(s)/particle(s) ``x_tm1`` through
        the dynamic model for time ``time``, with the application of random
        process noise ``noise``.

        In mathematical terms, this can be written as:

        .. math::

            x_t = f(x_{t-1},w_t,t), w_t \sim p(w_t)

        where :math:`x_t =` ``x_tm1``, :math:`w_t =` ``noise`` and
        :math:`t` = ``time``.

        Parameters
        ----------
        x_tm1 : 1/2-D numpy.ndarray of shape (Ns,Np)
            A (set of) prior state vector(s) :math:`x_{t-1}` from time
            :math:`t-1` (the default is None)

            - If None, the function will return the function transition \
              matrix :math:`F_t`. In this case, any value passed for ``noise``\
              will be ignored.

        noise : 1/2-D numpy.ndarray of shape (Ns,Np) or boolean
            ``noise`` can be of two types:

            - A *numpy array* containing a (set of) externally generated \
              random process noise sample(s) :math:`w_t`; or

            - A *boolean*, which is set to True or False depending on\
              whether noise should be applied or not. If set to True, the\
              model's own noise distribution will be used to generate the\
              noise samples :math:`w_t`

            (the default in False, in which case process noise will be ignored)

        time : scalar
            A time variant :math:`t` (the default is None, in which case it
            will be set equal to :py:attr:`time_variant`)

        Returns
        -------
        x_t : 1/2-D numpy.ndarray of shape (Ns,Np)
            The (set of) predicted state vector(s) :math:`x_t`
        """

        # TODO: Proper input validation

        if time is None:
            time = self.time_variant

        if x_tm1 is None:
            self._logger.debug("x_tm1 is None: Returning F")
            x_tm1 = sp.eye(self.ndim_state)
            noise = 0
        else:
            if issubclass(type(noise), bool) and noise:
                self._logger.debug(
                    "noise is True: Applying internally generated noise")
                noise = self.random(Np=x_tm1.shape[1], time=time)
            elif(issubclass(type(noise), bool) and not noise):
                self._logger.debug("noise is False: Ignoring noise")
                noise = 0
        self._logger.debug("self._F: {}".format(self._F(time),))

        x_t = self._F(time)@x_tm1 + noise
        return x_t

    def covar(self, time=None):
        """Returns the transition model noise covariance matrix.

        Parameters
        ----------
        time : scalar
            A time variant :math:`t` (the default is None, in which case it
            will be set equal to :py:attr:`time_variant`)

        Returns
        -------
        covar: numpy.ndarray of shape (Ns,Ns)
            The state covariance, in the form of a rank 2 array.
        """

        # TODO: Proper input validation

        if(time is None):
            time = self.time_variant

        covar = self._Q(time)

        return covar

    def random(self, Np=1, time=None):
        r""" Model noise/sample generation function

        Generates noise samples from the transition model.

        In mathematical terms, this can be written as:

        .. math::

            w_t \sim \mathcal(N)(0,Q_t)

        where :math:`w_t =` ``noise``.

        Parameters
        ----------
        Np: scalar, optional
            The number of samples to be generated (the default is 1)
        time : scalar, optional
            A time variant :math:`t` (the default is None, in which case it
            will be set equal to :py:attr:`time_variant`)

        Returns
        -------
        noise : 2-D array of shape (Ns,Np)
            A set of Np samples, generated from the model's noise distribution.
        """

        # TODO: Proper input validation

        if time is None:
            time = self.time_variant

        noise = multivariate_normal.rvs(
            sp.array([0, 0]), self._Q(time), Np).T

        return noise

    def pdf(self, x_t, x_tm1, time=None):
        r""" Model pdf/likelihood evaluation function

        Evaluates the pdf/likelihood of the (set of) predicted state vector(s)
        ``x_t``, given the (set of) prior state vector(s) ``x_tm1``.

        In mathematical terms, this can be written as:

        .. math::

            p = p(x_t | x_{t-1})

        where :math:`x_t` = ``x_t`` and :math:`x_{t-1}` = ``x_tm1``.

        Parameters
        ----------
        x_t : 1/2-D numpy.ndarray of shape (:math:`N_s`, :math:`N_{p1}`)
            A (set of) predicted state vector(s) :math:`x_{t|t-1}`
        x_tm1 : 1/2-D numpy.ndarray of shape (:math:`N_s`, :math:`N_{p2}`)
            A (set of) prior state vector(s) :math:`x_{t-1}` from time
            :math:`t-1`
        time : scalar, optional
            A time variant :math:`t` (the default is None, in which case it
            will be set equal to :py:attr:`time_variant`)

        Returns
        -------
        p: 1/2-D numpy.ndarray of shape (:math:`N_{p1}`, :math:`N_{p2}`)
            A matrix of probabilities/likelihoods, where each element
            (:math:`i`, :math:`j`) corresponds to the likelihood of
            ``x_t[:][i]`` given ``x_tm1[:][j]``.
        """

        # TODO: 1) Optimise performance
        #       2) Proper input validation
        Np1 = x_t.shape[1]
        Np2 = x_tm1.shape[1]
        p = sp.zeros((Np1, Np2))

        if time is None:
            time = self.time_variant

        for i in range(0, Np2):
            p[:, i] = multivariate_normal.pdf(
                x_t.T, mean=self.eval(x_tm1[:, i]).T, cov=self._Q(time)).T
        return p
