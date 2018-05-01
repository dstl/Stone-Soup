# -*- coding: utf-8 -*-
import scipy as sp
from scipy.stats import multivariate_normal
from ..types.model import Model, LinearModel, GaussianModel
from ..base import Property
from abc import abstractproperty


class MeasurementModel(Model):
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


class LinearGaussian1D(MeasurementModel, LinearModel, GaussianModel):
    """This is a class implementation of a time-invariant 1D
    Linear-Gaussian Measurement Model.

    The model is described by the following equations:

    .. math::

      y_t = H_k*x_t + v_k,\ \ \ \   v(k)\sim \mathcal{N}(0,R)

    where H_k is a 1xNs matrix and v_k is Gaussian distributed.

    Parameters
    ----------
    ndim_meas : :class:`int`
        The number of measurement dimensions Nm (constant = 1)
    """

    ndim_state = Property(int, doc="The number of dimensions of the state,"
                          " to which the model maps")
    mapping = Property(sp.ndarray, doc="Index of state dimension to which"
                       " the measurement maps")
    noise_var = Property(float, doc="The measurement noise variance")

    _transfer_matrix = None
    _noise_covariance = None
    _ndim_meas = 1

    def __init__(self, ndim_state, mapping, noise_var, *args, **kwargs):
        """Constructor method

        Parameters
        ----------
        ndim_state : :class:`int`
            The number of dimensions of the state, to which the model maps to.
        mapping : :class:`int`
            Index of state dimension to which the measurement maps
        noise_var : :class:`float`
            The measurement noise variance
        """

        self._transfer_matrix = sp.zeros((1, ndim_state))
        self._transfer_matrix[0, mapping] = 1
        self._noise_covariance = noise_var

        super().__init__(ndim_state, mapping, noise_var, *args, **kwargs)

        # TODO: Proper input validation

    @property
    def ndim_meas(self):
        return self._ndim_meas

    def eval(self, state_vec=None, noise=False, time=None):
        """ Model transition function

        Projects a given (set of) state(s)/particle(s) ``state_vec``
        through the measurement model, with the application of random
        process noise ``noise``.

        In mathematical terms, this can be written as:

        .. math::

            y_t = H*x_t + v_t, v_t \sim p(v_t)

        where :math:`y_t =` ``meas_vec``, :math:`x_t =` ``state_vec`` and
        :math:`v_t =` ``noise``.

        Parameters
        ----------
        state_vec : :class:`numpy.ndarray` of shape (Ns,Np)
            A (set of) prior state vector(s) :math:`x_{t-1}` from time
            :math:`t-1` (the default is None)

            - If None, the function will return the measurement \
              matrix :math:`H_t`. In this case, any value passed for ``noise``\
              will be ignored.

        noise : :class:`numpy.ndarray` of shape (Ns,Np) or boolean
            ``noise`` can be of two types:

            - A *numpy array* containing a (set of) externally generated \
              random process noise sample(s) :math:`v_t`; or

            - A *boolean*, which is set to True or False depending on\
              whether noise should be applied or not. If set to True, the\
              model's own noise distribution will be used to generate the\
              noise samples :math:`v_t`

            (the default in False, in which case process noise will be ignored)

        Returns
        -------
        meas_vec : :class:`numpy.ndarray` of shape (:py:attr:`~ndim_meas`,Np)
            The (set of) projected state vector(s)/measurement(s) :math:`y_{t}`
        """

        if state_vec is None:
            state_vec = sp.eye(self.ndim_state)
            noise = 0
        else:
            if issubclass(type(noise), bool) and noise:
                noise = self.random(Np=state_vec.shape[1])
            elif(issubclass(type(noise), bool) and not noise):
                noise = 0

        meas_vec = self._transfer_matrix@state_vec + noise

        return meas_vec

    def covar(self):
        """Returns the measurement model noise covariance matrix.

        Returns
        -------
        covar: numpy.ndarray of shape\
        (:py:attr:`~ndim_meas`,:py:attr:`~ndim_meas`)
            The state covariance, in the form of a rank 2 array.
        """

        # TODO: Proper input validation

        return self._noise_covariance

    def random(self, num_samples=1, time=None):
        """ Model noise/sample generation function

        Generates noise samples from the transition model.

        In mathematical terms, this can be written as:

        .. math::

            v_t \sim \mathcal(N)(0,R_t)

        where :math:`v_t =` ``noise``.

        Parameters
        ----------
        num_samples: scalar, optional
            The number of samples to be generated (the default is 1)

        Returns
        -------
        noise : 2-D array of shape (:py:attr:`~ndim_meas`,``num_samples``)
            A set of Np samples, generated from the model's noise
            distribution.
        """

        # TODO: Proper input validation

        noise = multivariate_normal.rvs(
            sp.array([0, 0]), self._noise_covariance, num_samples).T

        return noise

    def pdf(self, meas_vec, state_vec, time=None):
        """ Measurement pdf/likelihood evaluation function

        Evaluates the pdf/likelihood of the (set of) measurement vector(s)
        ``meas_vec``, given the (set of) state vector(s) ``state_vec``.

        In mathematical terms, this can be written as:

        .. math::

            p = p(y_t | x_t)

        Parameters
        ----------
        meas_vec : :class:`numpy.ndarray` of shape\
        (:py:attr:`~ndim_meas`, :math:`N_{p1}`)
            A (set of) measurement vector(s) :math:`y_t`
        state_vec : :class:`numpy.ndarray` of shape\
        (:py:attr:`~ndim_state`, :math:`N_{p2}`)
            A (set of) state vector(s) :math:`x_t`

        Returns
        -------
        p: :class:`numpy.ndarray` of shape (:math:`N_{p1}`, :math:`N_{p2}`)
            A matrix of probabilities/likelihoods, where each element
            (:math:`i`, :math:`j`) corresponds to the likelihood of
            ``y_t[:][i]`` given ``x_t[:][j]``.
        """

        # TODO: 1) Optimise performance
        #       2) Proper input validation
        Np1 = meas_vec.shape[1]
        Np2 = state_vec.shape[1]
        p = sp.zeros((Np1, Np2))

        for i in range(0, Np2):
            p[:, i] = multivariate_normal.pdf(
                meas_vec.T, mean=self.eval(state_vec[:, i]).T,
                cov=self._noise_covariance
            ).T
        return p
