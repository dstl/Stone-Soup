# -*- coding: utf-8 -*-

import scipy as sp
from scipy.stats import multivariate_normal

from ...base import Property
from ...types.array import StateVector, CovarianceMatrix
from ..base import NonLinearModel, GaussianModel
from .base import MeasurementModel
from ...functions import cart2pol, cart2sphere, cart2angles, rotx, roty, rotz


class RangeBearingElevationGaussianToCartesian(MeasurementModel,
                                               NonLinearModel,
                                               GaussianModel):
    r"""This is a class implementation of a time-invariant measurement model, \
    where measurements are assumed to be received in the form of bearing \
    (:math:`\phi`), elevation (:math:`\theta`) and range (:math:`r`), with \
    Gaussian noise in each dimension.

    The model is described by the following equations:

    .. math::

      \vec{y}_t = h(\vec{x}_t, \vec{v}_t)

    where:

    * :math:`\vec{y}_t` is a measurement vector of the form:

    .. math::

      \vec{y}_t = \begin{bmatrix}
                \theta \\
                \phi \\
                r
            \end{bmatrix}

    * :math:`h` is a non-linear model function of the form:

    .. math::

      h(\vec{x}_t,\vec{v}_t) = \begin{bmatrix}
                asin(\mathcal{z}/\sqrt{\mathcal{x}^2 + \mathcal{y}^2 +\mathcal{z}^2}) \\
                atan2(\mathcal{y},\mathcal{x}) \\
                \sqrt{\mathcal{x}^2 + \mathcal{y}^2 + \mathcal{z}^2}
                \end{bmatrix} + \vec{v}_t

    * :math:`\vec{v}_t` is Gaussian distributed with covariance :math:`R`, i.e.:

    .. math::

      \vec{v}_t \sim \mathcal{N}(0,R)

    .. math::

      R = \begin{bmatrix}
            \sigma_{\theta}^2 & 0 & 0 \\
            0 & \sigma_{\phi}^2 & 0 \\
            0 & 0 & \sigma_{r}^2
            \end{bmatrix}

    The :py:attr:`mapping` property of the model is a 3 element vector, \
    whose first (i.e. :py:attr:`mapping[0]`), second (i.e. \
    :py:attr:`mapping[1]`) and third (i.e. :py:attr:`mapping[2`) elements \
    contain the state index of the :math:`x`, :math:`y` and :math:`z`  \
    coordinates, respectively.

    Note
    ----
    The current implementation of this class assumes a 3D Cartesian plane.

    """  # noqa:E501

    noise_covar = Property(CovarianceMatrix, doc="Noise covariance")
    translation_offset = Property(
        StateVector, default=StateVector(sp.array([[0], [0], [0]])),
        doc="A 3x1 array specifying the Cartesian origin offset in terms of :math:`x,y,z`\
            coordinates.")
    rotation_offset = Property(
        StateVector, default=StateVector(sp.array([[0], [0], [0]])),
        doc="A 3x1 array of angles (rad), specifying the clockwise rotation\
            around each Cartesian axis in the order :math:`x,y,z`.\
            The rotation angles are positive if the rotation is in the \
            counter-clockwise direction when viewed by an observer looking\
            along the respective rotation axis, towards the origin.")

    @property
    def ndim_meas(self):
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return 3

    @property
    def _rotation_matrix(self):
        """_rotation_matrix getter method

        Calculates and returns the (3D) axis rotation matrix.

        Returns
        -------
        :class:`numpy.ndarray` of shape (3, 3)
            The model (3D) rotation matrix.
        """

        theta_x = -self.rotation_offset[0, 0]
        theta_y = -self.rotation_offset[1, 0]
        theta_z = -self.rotation_offset[2, 0]

        return rotz(theta_z)@roty(theta_y)@rotx(theta_x)

    def function(self, state_vector, noise=None, **kwargs):
        r"""Model function :math:`h(\vec{x}_t,\vec{v}_t)`

        Parameters
        ----------
        state_vector: :class:`~.StateVector`
            An input state vector
        noise: :class:`numpy.ndarray`
            An externally generated random process noise sample (the default in
            `None`, in which case process noise will be generated internally)

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_state`, 1)
            The model function evaluated given the provided time interval.
        """

        if noise is None:
            noise = self.rvs()

        # Account for origin offset
        xyz = state_vector[self.mapping] - self.translation_offset

        # Rotate coordinates
        xyz_rot = self._rotation_matrix @ xyz

        # Convert to Spherical
        rho, phi, theta = cart2sphere(*xyz_rot[:, 0])

        return sp.array([[theta],
                         [phi],
                         [rho]]) + noise

    def covar(self, **kwargs):
        """Returns the measurement model noise covariance matrix.

        Returns
        -------
        :class:`~.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_meas`, :py:attr:`~ndim_meas`)
            The measurement noise covariance.
        """

        return self.noise_covar

    def rvs(self, num_samples=1, **kwargs):
        r""" Model noise/sample generation function

        Generates noise samples from the measurement model.

        In mathematical terms, this can be written as:

        .. math::

            \vec{v}_t \sim \mathcal{N}(0,R)

        Parameters
        ----------
        num_samples: scalar, optional
            The number of samples to be generated (the default is 1)

        Returns
        -------
        2-D array of shape (:py:attr:`~ndim_meas`, ``num_samples``)
            A set of Np samples, generated from the model's noise
            distribution.
        """

        noise = multivariate_normal.rvs(
            sp.zeros(self.ndim_meas), self.covar(), num_samples)

        if num_samples == 1:
            return noise.reshape((-1, 1))
        else:
            return noise.T

    def pdf(self, meas_vec, state_vec, **kwargs):
        r""" Measurement pdf/likelihood evaluation function

        Evaluates the pdf/likelihood of the (set of) measurement vector(s)
        ``meas_vec``, given the (set of) state vector(s) ``state_vec``.

        In mathematical terms, this can be written as:

        .. math::

            p(\vec{y}_t | \vec{x}_t) = \mathcal{N}(\vec{y}_t; \vec{x}_t, R)

        Parameters
        ----------
        meas_vec : :class:`~.StateVector`
            A measurement
        state_vec : :class:`~.StateVector`
            A state

        Returns
        -------
        :class:`float`
            The likelihood of ``meas``, given ``state``
        """

        likelihood = multivariate_normal.pdf(
            meas_vec.T,
            mean=(self.function(state_vec, 0)).ravel(),
            cov=self.covar()
        )
        return likelihood


class RangeBearingGaussianToCartesian(
        RangeBearingElevationGaussianToCartesian):
    r"""This is a class implementation of a time-invariant measurement model, \
    where measurements are assumed to be received in the form of bearing \
    (:math:`\phi`) and range (:math:`r`), with Gaussian noise in each dimension.

    The model is described by the following equations:

    .. math::

      \vec{y}_t = h(\vec{x}_t, \vec{v}_t)

    where:

    * :math:`\vec{y}_t` is a measurement vector of the form:

    .. math::

      \vec{y}_t = \begin{bmatrix}
                \phi \\
                r
            \end{bmatrix}

    * :math:`h` is a non-linear model function of the form:

    .. math::

      h(\vec{x}_t,\vec{v}_t) = \begin{bmatrix}
                atan2(\mathcal{y},\mathcal{x}) \\
                \sqrt{\mathcal{x}^2 + \mathcal{y}^2}
                \end{bmatrix} + \vec{v}_t

    * :math:`\vec{v}_t` is Gaussian distributed with covariance :math:`R`, i.e.:

    .. math::

      \vec{v}_t \sim \mathcal{N}(0,R)

    .. math::

      R = \begin{bmatrix}
            \sigma_{\phi}^2 & 0 \\
            0 & \sigma_{r}^2
            \end{bmatrix}

    The :py:attr:`mapping` property of the model is a 2 element vector, \
    whose first (i.e. :py:attr:`mapping[0]`) and second (i.e. \
    :py:attr:`mapping[0]`) elements contain the state index of the \
    :math:`x` and :math:`y` coordinates, respectively.

    Note
    ----
    The current implementation of this class assumes a 2D Cartesian plane.

    """  # noqa:E501

    noise_covar = Property(CovarianceMatrix, doc="Noise covariance")
    translation_offset = Property(
        StateVector, default=StateVector(sp.array([[0], [0]])),
        doc="A 2x1 array specifying the origin offset in terms of :math:`x,y`\
            coordinates.")

    @property
    def ndim_meas(self):
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return 2

    def function(self, state_vector, noise=None, **kwargs):
        r"""Model function :math:`h(\vec{x}_t,\vec{v}_t)`

        Parameters
        ----------
        state_vector: :class:`~.StateVector`
            An input state vector
        noise: :class:`numpy.ndarray`
            An externally generated random process noise sample (the default in
            `None`, in which case process noise will be generated internally)

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_meas`, 1)
            The model function evaluated given the provided time interval.
        """

        if noise is None:
            noise = self.rvs()

        # Account for origin offset
        xyz = [[state_vector[self.mapping[0], 0]
                - self.translation_offset[0, 0]],
               [state_vector[self.mapping[1], 0]
                - self.translation_offset[1, 0]],
               [0]]

        # Rotate coordinates
        xyz_rot = self._rotation_matrix @ xyz

        # Covert to polar
        rho, phi = cart2pol(*xyz_rot[:2, 0])

        return sp.array([[phi], [rho]]) + noise


class BearingElevationGaussianToCartesian(RangeBearingGaussianToCartesian):
    r"""This is a class implementation of a time-invariant measurement model, \
    where measurements are assumed to be received in the form of bearing \
    (:math:`\phi`) and elevation (:math:`\theta`) and with \
    Gaussian noise in each dimension.

    The model is described by the following equations:

    .. math::

      \vec{y}_t = h(\vec{x}_t, \vec{v}_t)

    where:

    * :math:`\vec{y}_t` is a measurement vector of the form:

    .. math::

      \vec{y}_t = \begin{bmatrix}
                \theta \\
                \phi
            \end{bmatrix}

    * :math:`h` is a non-linear model function of the form:

    .. math::

      h(\vec{x}_t,\vec{v}_t) = \begin{bmatrix}
                asin(\mathcal{z}/\sqrt{\mathcal{x}^2 + \mathcal{y}^2 +\mathcal{z}^2}) \\
                atan2(\mathcal{y},\mathcal{x}) \\
                \end{bmatrix} + \vec{v}_t

    * :math:`\vec{v}_t` is Gaussian distributed with covariance :math:`R`, i.e.:

    .. math::

      \vec{v}_t \sim \mathcal{N}(0,R)

    .. math::

      R = \begin{bmatrix}
            \sigma_{\theta}^2 & 0 \\
            0 & \sigma_{\phi}^2\\
            \end{bmatrix}

    The :py:attr:`mapping` property of the model is a 3 element vector, \
    whose first (i.e. :py:attr:`mapping[0]`), second (i.e. \
    :py:attr:`mapping[1]`) and third (i.e. :py:attr:`mapping[2]`) elements  \
    contain the state index of the :math:`x`, :math:`y` and :math:`z`  \
    coordinates, respectively.

    Note
    ----
    The current implementation of this class assumes a 3D Cartesian plane.

    """  # noqa:E501

    noise_covar = Property(CovarianceMatrix, doc="Noise covariance")
    translation_offset = Property(
        StateVector, default=StateVector(sp.array([[0], [0], [0]])),
        doc="A 3x1 array specifying the origin offset in terms of :math:`x,y,z`\
            coordinates.")

    @property
    def ndim_meas(self):
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return 2

    def function(self, state_vector, noise=None, **kwargs):
        r"""Model function :math:`h(\vec{x}_t,\vec{v}_t)`

        Parameters
        ----------
        state_vector: :class:`~.StateVector`
            An input state vector
        noise: :class:`numpy.ndarray`
            An externally generated random process noise sample (the default in
            `None`, in which case process noise will be generated internally)

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_state`, 1)
            The model function evaluated given the provided time interval.
        """

        if noise is None:
            noise = self.rvs()

        # Account for origin offset
        xyz = state_vector[self.mapping] - self.translation_offset

        # Rotate coordinates
        xyz_rot = self._rotation_matrix @ xyz

        # Convert to Angles
        phi, theta = cart2angles(*xyz_rot[:, 0])

        return sp.array([[theta],
                         [phi]]) + noise
