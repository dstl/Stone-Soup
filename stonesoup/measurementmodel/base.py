# -*- coding: utf-8 -*-
import scipy as sp
from scipy.stats import multivariate_normal

from ..base import Base, Property
from abc import abstractmethod, abstractproperty


class MeasurementModel(Base):
    """Measurement Model base class"""

    @abstractproperty
    def ndim_state(self):
        """ Number of state dimensions"""
        pass

    @abstractproperty
    def ndim_meas(self):
        """ Number of measurement dimensions"""
        pass

    @abstractproperty
    def mapping(self):
        """ Mapping between measurement and state dims """
        pass

    @abstractmethod
    def eval(self):
        """ Measurement model function """
        pass

    @abstractmethod
    def random(self):
        """ Measurement noise/sample generation function """
        pass

    @abstractmethod
    def pdf(self):
        """ Measurement likelihood evaluation function """
        pass


class LinearGaussian1D(MeasurementModel):
    r"""This is a class implementation of a time-invariant 1D
    Linear-Gaussian Measurement Model.

    The model is described by the following equations:

    .. math::

      y_t = H_k*x_t + v_k,\ \ \ \   v(k)\sim \mathcal{N}(0,R)

    where H_k is a 1xNs matrix and v_k is Gaussian distributed.

    Parameters
        ----------
        ndim_state : int
            The number of dimensions of the state Ns, to which the model
            maps to.
        ndim_meas : int
            The number of measurement dimensions Nm (constant = 1)
        mapping : int
            Index of state dimension to which the measurement maps
        noise_var : float
            The measurement noise variance
    """

    ndim_state = Property(int, doc="number of state dimensions")
    mapping = Property(sp.ndarray, doc="state-to-measurement dimension mappin")
    noise_var = Property(float, doc="Measurement noise variance")

    def __init__(self, ndim_state, mapping, noise_var, *args, **kwargs):
        """Constructor method

        Parameters
        ----------
        ndim_state : int
            The number of dimensions of the state, to which the model maps to.
        mapping : int
            Index of state dimension to which the measurement maps
        noise_var : float
            The measurement noise variance
        """

        self._H = sp.zeros((1, ndim_state))
        self._H[0, mapping] = 1
        self._R = noise_var

        super().__init__(ndim_state, mapping, noise_var, *args, **kwargs)

        # TODO: Proper input validation

    @property
    def ndim_meas(self):
        return self._ndim_meas

    def eval(self, x_t=None, noise=False):
        r""" Model transition function

        Propagates a given (set of) state(s)/particle(s) ``x_tm1`` through
        the dynamic model for time ``time``, with the application of random
        process noise ``noise``.

        In mathematical terms, this can be written as:

        .. math::

            y_t = H*x_t + v_t, v_t \sim p(v_t)

        where :math:`v_t =` ``noise``.

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
              random process noise sample(s) :math:`v_t`; or

            - A *boolean*, which is set to True or False depending on\
              whether noise should be applied or not. If set to True, the\
              model's own noise distribution will be used to generate the\
              noise samples :math:`v_t`

            (the default in False, in which case process noise will be ignored)

        time : scalar
            A time variant :math:`t` (the default is None, in which case it
            will be set equal to :py:attr:`time_variant`)

        Returns
        -------
        x_t : 1/2-D numpy.ndarray of shape (Ns,Np)
            The (set of) predicted state vector(s) :math:`x_{t|t-1}`
        """

        if x_t is None:
            x_t = sp.eye(self.ndim_state)
            noise = 0
        else:
            if issubclass(type(noise), bool) and noise:
                noise = self.random(Np=x_t.shape[1])
            elif(issubclass(type(noise), bool) and not noise):
                noise = 0

        y_t = self._H@x_t + noise

        return y_t

    def covar(self):
        """Returns the measurement model noise covariance matrix.

        Returns
        -------
        covar: numpy.ndarray of shape (Ns,Ns)
            The state covariance, in the form of a rank 2 array.
        """

        # TODO: Proper input validation

        return self._R

    def random(self, Np=1, time=None):
        r""" Model noise/sample generation function

        Generates noise samples from the transition model.

        In mathematical terms, this can be written as:

        .. math::

            v_t \sim \mathcal(N)(0,R_t)

        where :math:`v_t =` ``noise``.

        Parameters
        ----------
        Np: scalar, optional
            The number of samples to be generated (the default is 1)

        Returns
        -------
        noise : 2-D array of shape (Ns,Np)
            A set of Np samples, generated from the model's noise
            distribution.
        """

        # TODO: Proper input validation

        noise = multivariate_normal.rvs(
            sp.array([0, 0]), self._R, Np).T

        return noise

    def pdf(self, y_t, x_t, time=None):
        r""" Measurement pdf/likelihood evaluation function

        Evaluates the pdf/likelihood of the (set of) measurement vector(s)
        ``y_t``, given the (set of) state vector(s) ``x_t``.

        In mathematical terms, this can be written as:

        .. math::

            p = p(y_t | x_t)

        Parameters
        ----------
        y_t : 1/2-D numpy.ndarray of shape (:math:`N_m`, :math:`N_{p1}`)
            A (set of) measurement vector(s) :math:`y_t`
        x_t : 1/2-D numpy.ndarray of shape (:math:`N_s`, :math:`N_{p2}`)
            A (set of) state vector(s) :math:`x_t`

        Returns
        -------
        p: 1/2-D numpy.ndarray of shape (:math:`N_{p1}`, :math:`N_{p2}`)
            A matrix of probabilities/likelihoods, where each element
            (:math:`i`, :math:`j`) corresponds to the likelihood of
            ``y_t[:][i]`` given ``x_t[:][j]``.
        """

        # TODO: 1) Optimise performance
        #       2) Proper input validation
        Np1 = y_t.shape[1]
        Np2 = x_t.shape[1]
        p = sp.zeros((Np1, Np2))

        for i in range(0, Np2):
            p[:, i] = multivariate_normal.pdf(
                y_t.T, mean=self.eval(x_t[:, i]).T, cov=self._R).T
        return p
