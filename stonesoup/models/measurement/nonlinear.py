# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import inv

from ...base import Property

from ...functions import cart2pol, pol2cart, \
    cart2sphere, sphere2cart, cart2angles, \
    rotx, roty, rotz
from ...types.array import StateVector, CovarianceMatrix
from ...types.angle import Bearing, Elevation
from ..base import NonLinearModel, GaussianModel, ReversibleModel
from .base import MeasurementModel


class NonLinearGaussianMeasurement(MeasurementModel,
                                   NonLinearModel,
                                   GaussianModel):
    r"""This class combines the MeasurementModel, NonLinearModel and \
    GaussianModel classes. It is not meant to be instantiated directly \
    but subclasses should be derived from this class.
    """
    noise_covar = Property(CovarianceMatrix, doc="Noise covariance")
    rotation_offset = Property(
        StateVector, default=StateVector(np.array([[0], [0], [0]])),
        doc="A 3x1 array of angles (rad), specifying the clockwise rotation\
            around each Cartesian axis in the order :math:`x,y,z`.\
            The rotation angles are positive if the rotation is in the \
            counter-clockwise direction when viewed by an observer looking\
            along the respective rotation axis, towards the origin.")

    def covar(self, **kwargs):
        """Returns the measurement model noise covariance matrix.

        Returns
        -------
        :class:`~.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_meas`, :py:attr:`~ndim_meas`)
            The measurement noise covariance.
        """

        return self.noise_covar

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


class CartesianToElevationBearingRange(
        NonLinearGaussianMeasurement, ReversibleModel):
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

    translation_offset = Property(
        StateVector, default=StateVector(np.array([[0], [0], [0]])),
        doc="A 3x1 array specifying the Cartesian origin offset in terms of :math:`x,y,z`\
            coordinates.")

    @property
    def ndim_meas(self):
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return 3

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

        return np.array([[Elevation(theta)],
                         [Bearing(phi)],
                         [rho]]) + noise

    def inverse_function(self, state_vector, **kwargs):

        theta, phi, rho = state_vector[:, 0]
        x, y, z = sphere2cart(rho, phi, theta)

        xyz = [[x], [y], [z]]
        inv_rotation_matrix = inv(self._rotation_matrix)
        xyz_rot = inv_rotation_matrix @ xyz
        xyz = [xyz_rot[0][0], xyz_rot[1][0], xyz_rot[2][0]]
        x, y, z = xyz + self.translation_offset[:, 0]

        res = np.zeros((self.ndim_state, 1))
        res[self.mapping, 0] = x, y, z

        return res

    def rvs(self, num_samples=1, **kwargs):
        out = super().rvs(num_samples, **kwargs)
        out = np.array([[Elevation(0.)], [Bearing(0.)], [0.]]) + out
        return out


class CartesianToBearingRange(
        NonLinearGaussianMeasurement, ReversibleModel):
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

    translation_offset = Property(
        StateVector, default=StateVector(np.array([[0], [0]])),
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

    def inverse_function(self, state_vector, **kwargs):
        if not ((self.rotation_offset[0][0] == 0)
                and (self.rotation_offset[1][0] == 0)):
            raise RuntimeError(
                "Measurement model assumes 2D space. \
                Rotation in 3D space is unsupported at this time.")

        phi, rho = state_vector[:, 0]
        x, y = pol2cart(rho, phi)

        xyz = [[x], [y], [0]]
        inv_rotation_matrix = inv(self._rotation_matrix)
        xyz_rot = inv_rotation_matrix @ xyz
        xy = [xyz_rot[0][0], xyz_rot[1][0]]
        x, y = xy + self.translation_offset[:, 0]

        res = np.zeros((self.ndim_state, 1))
        res[self.mapping, 0] = x, y

        return res

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

        return np.array([[Bearing(phi)], [rho]]) + noise

    def rvs(self, num_samples=1, **kwargs):
        out = super().rvs(num_samples, **kwargs)
        out = np.array([[Bearing(0)], [0.]]) + out
        return out


class CartesianToElevationBearing(NonLinearGaussianMeasurement):
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

    translation_offset = Property(
        StateVector, default=StateVector(np.array([[0], [0], [0]])),
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

        return np.array([[Elevation(theta)],
                         [Bearing(phi)]]) + noise

    def rvs(self, num_samples=1, **kwargs):
        out = super().rvs(num_samples, **kwargs)
        out = np.array([[Elevation(0.)], [Bearing(0.)]]) + out
        return out
