# -*- coding: utf-8 -*-
from abc import ABC
from typing import List, Union
import copy

import numpy as np
from scipy.linalg import inv, pinv, block_diag

from ...base import Property

from ...functions import cart2pol, pol2cart, \
    cart2sphere, sphere2cart, cart2angles, \
    rotx, roty, rotz
from ...types.array import StateVector, CovarianceMatrix, StateVectors
from ...types.angle import Bearing, Elevation
from ..base import LinearModel, NonLinearModel, GaussianModel, ReversibleModel
from .base import MeasurementModel


class CombinedReversibleGaussianMeasurementModel(ReversibleModel, GaussianModel, MeasurementModel):
    r"""Combine multiple models into a single model by stacking them.

    The assumption is that all models are Gaussian, and must be combination of
    :class:`~.LinearModel` and :class:`~.NonLinearModel` models. They must all
    expect the same dimension state vector (i.e. have the same
    :attr:`~.MeasurementModel.ndim_state`), using model mapping as appropriate.

    This also implements the :meth:`inverse_function`, but will raise a
    :exc:`NotImplementedError` if any model isn't either a
    :class:`~.LinearModel` or :class:`~.ReversibleModel`.
    """
    mapping = None
    model_list = Property(List[MeasurementModel], doc="List of Measurement Models.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for model in self.model_list:
            if model.ndim_state != self.ndim_state:
                raise ValueError("Models must all have the same `ndim_state`")

    @property
    def ndim_state(self) -> int:
        """Number of state dimensions"""
        return self.model_list[0].ndim_state

    @property
    def ndim_meas(self) -> int:
        return sum(model.ndim_meas for model in self.model_list)

    def function(self, state, **kwargs) -> StateVector:
        return np.vstack([model.function(state, **kwargs)
                          for model in self.model_list]).view(StateVector)

    @staticmethod
    def _linear_inverse_function(model, state, **kwargs):
        model_matrix = model.matrix(**kwargs)
        inv_model_matrix = pinv(model_matrix)

        return inv_model_matrix@state.state_vector

    def inverse_function(self, detection, **kwargs) -> StateVector:
        state = copy.copy(detection)
        ndim_count = 0
        state_vector = np.zeros((self.ndim_state, 1)).view(StateVector)
        for model in self.model_list:
            state.state_vector = detection.state_vector[ndim_count:model.ndim_meas + ndim_count, :]
            if isinstance(model, ReversibleModel):
                state_vector += model.inverse_function(state, **kwargs)
            elif isinstance(model, LinearModel):
                state_vector += self._linear_inverse_function(model, state, **kwargs)
            else:
                raise NotImplementedError(
                    "Model {!r} not reversible".format(type(model)))
            ndim_count += model.ndim_meas

        return state_vector

    def covar(self, **kwargs) -> CovarianceMatrix:
        return block_diag(
            *(model.covar(**kwargs) for model in self.model_list)
            ).view(CovarianceMatrix)

    def rvs(self, num_samples=1, **kwargs) -> Union[StateVector, StateVectors]:
        rvs_vectors = np.vstack([model.rvs(num_samples, **kwargs)
                                 for model in self.model_list])
        if num_samples == 1:
            return rvs_vectors.view(StateVector)
        else:
            return rvs_vectors.view(StateVectors)


class NonLinearGaussianMeasurement(MeasurementModel, NonLinearModel, GaussianModel, ABC):
    r"""This class combines the MeasurementModel, NonLinearModel and \
    GaussianModel classes. It is not meant to be instantiated directly \
    but subclasses should be derived from this class.
    """
    noise_covar = Property(CovarianceMatrix, doc="Noise covariance")
    rotation_offset = Property(
        StateVector, default=None,
        doc="A 3x1 array of angles (rad), specifying the clockwise rotation\
            around each Cartesian axis in the order :math:`x,y,z`.\
            The rotation angles are positive if the rotation is in the \
            counter-clockwise direction when viewed by an observer looking\
            along the respective rotation axis, towards the origin.")

    def __init__(self, *args, **kwargs):
        """
        Ensure that the rotation offset is initiated
        """
        super().__init__(*args, **kwargs)
        # Set values to defaults if not provided
        if self.rotation_offset is None:
            self.rotation_offset = StateVector([[0], [0], [0]])

    def covar(self, **kwargs) -> CovarianceMatrix:
        """Returns the measurement model noise covariance matrix.

        Returns
        -------
        :class:`~.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_meas`, :py:attr:`~ndim_meas`)
            The measurement noise covariance.
        """

        return self.noise_covar

    @property
    def _rotation_matrix(self) -> np.ndarray:
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


class CartesianToElevationBearingRange(NonLinearGaussianMeasurement, ReversibleModel):
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
        StateVector, default=None,
        doc="A 3x1 array specifying the Cartesian origin offset in terms of :math:`x,y,z`\
            coordinates.")

    def __init__(self, *args, **kwargs):
        """
        Ensure that the translation offset is initiated
        """
        super().__init__(*args, **kwargs)
        # Set values to defaults if not provided
        if self.translation_offset is None:
            self.translation_offset = StateVector([0] * self.ndim)

    @property
    def ndim_meas(self) -> int:
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return 3

    def function(self, state, noise=False, **kwargs) -> StateVector:
        r"""Model function :math:`h(\vec{x}_t,\vec{v}_t)`

        Parameters
        ----------
        state: :class:`~.State`
            An input state
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is
            `False`, in which case no noise will be added
            if 'True', the output of :meth:`~.Model.rvs` is added)

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_state`, 1)
            The model function evaluated given the provided time interval.
        """

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs()
            else:
                noise = 0

        # Account for origin offset
        xyz = state.state_vector[self.mapping] - self.translation_offset

        # Rotate coordinates
        xyz_rot = self._rotation_matrix @ xyz

        # Convert to Spherical
        rho, phi, theta = cart2sphere(*xyz_rot)

        return StateVector([[Elevation(theta)], [Bearing(phi)], [rho]]) + noise

    def inverse_function(self, detection, **kwargs) -> StateVector:

        theta, phi, rho = detection.state_vector
        xyz = StateVector(sphere2cart(rho, phi, theta))

        inv_rotation_matrix = inv(self._rotation_matrix)
        xyz = inv_rotation_matrix @ xyz

        res = np.zeros((self.ndim_state, 1)).view(StateVector)
        res[self.mapping, :] = xyz + self.translation_offset

        return res

    def rvs(self, num_samples=1, **kwargs) -> Union[StateVector, StateVectors]:
        out = super().rvs(num_samples, **kwargs)
        out = np.array([[Elevation(0.)], [Bearing(0.)], [0.]]) + out
        return out


class CartesianToBearingRange(NonLinearGaussianMeasurement, ReversibleModel):
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
        StateVector, default=None,
        doc="A 2x1 array specifying the origin offset in terms of :math:`x,y`\
            coordinates.")

    def __init__(self, *args, **kwargs):
        """
        Ensure that the translation offset is initiated
        """
        super().__init__(*args, **kwargs)
        # Set values to defaults if not provided
        if self.translation_offset is None:
            self.translation_offset = StateVector([0] * self.ndim)

    @property
    def ndim_meas(self) -> int:
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return 2

    def inverse_function(self, detection, **kwargs) -> StateVector:
        if not ((self.rotation_offset[0] == 0)
                and (self.rotation_offset[1] == 0)):
            raise RuntimeError(
                "Measurement model assumes 2D space. \
                Rotation in 3D space is unsupported at this time.")

        phi, rho = detection.state_vector[:]
        xy = StateVector(pol2cart(rho, phi))

        xyz = np.concatenate((xy, StateVector([0])), axis=0)
        inv_rotation_matrix = inv(self._rotation_matrix)
        xyz = inv_rotation_matrix @ xyz
        xy = xyz[0:2]

        res = np.zeros((self.ndim_state, 1)).view(StateVector)
        res[self.mapping, :] = xy + self.translation_offset

        return res

    def function(self, state, noise=False, **kwargs) -> StateVector:
        r"""Model function :math:`h(\vec{x}_t,\vec{v}_t)`

        Parameters
        ----------
        state: :class:`~.State`
            An input state
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is
            `False`, in which case no noise will be added
            if 'True', the output of :meth:`~.Model.rvs` is added)

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_meas`, 1)
            The model function evaluated given the provided time interval.
        """

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs()
            else:
                noise = 0

        # Account for origin offset
        xyz = [[state.state_vector[self.mapping[0], 0]
                - self.translation_offset[0, 0]],
               [state.state_vector[self.mapping[1], 0]
                - self.translation_offset[1, 0]],
               [0]]

        # Rotate coordinates
        xyz_rot = self._rotation_matrix @ xyz

        # Covert to polar
        rho, phi = cart2pol(*xyz_rot[:2, 0])

        return StateVector([[Bearing(phi)], [rho]]) + noise

    def rvs(self, num_samples=1, **kwargs) -> Union[StateVector, StateVectors]:
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
        StateVector, default=None,
        doc="A 3x1 array specifying the origin offset in terms of :math:`x,y,z`\
            coordinates.")

    def __init__(self, *args, **kwargs):
        """
        Ensure that the translation offset is initiated
        """
        super().__init__(*args, **kwargs)
        # Set values to defaults if not provided
        if self.translation_offset is None:
            self.translation_offset = StateVector([0] * self.ndim_state)

    @property
    def ndim_meas(self) -> int:
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return 2

    def function(self, state, noise=False, **kwargs) -> StateVector:
        r"""Model function :math:`h(\vec{x}_t,\vec{v}_t)`

        Parameters
        ----------
        state: :class:`~.State`
            An input state
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is
            `False`, in which case no noise will be added
            if 'True', the output of :meth:`~.Model.rvs` is added)

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_state`, 1)
            The model function evaluated given the provided time interval.
        """

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs()
            else:
                noise = 0

        # Account for origin offset
        xyz = state.state_vector[self.mapping, :] - self.translation_offset

        # Rotate coordinates
        xyz_rot = self._rotation_matrix @ xyz

        # Convert to Angles
        phi, theta = cart2angles(*xyz_rot)

        return StateVector([[Elevation(theta)], [Bearing(phi)]]) + noise

    def rvs(self, num_samples=1, **kwargs) -> Union[StateVector, StateVectors]:
        out = super().rvs(num_samples, **kwargs)
        out = np.array([[Elevation(0.)], [Bearing(0.)]]) + out
        return out


class CartesianToBearingRangeRate(NonLinearGaussianMeasurement):
    r"""This is a class implementation of a time-invariant measurement model, \
    where measurements are assumed to be received in the form of bearing \
    (:math:`\phi`), range (:math:`r`) and range-rate (:math:`\dot{r}`),
    with Gaussian noise in each dimension.

    The model is described by the following equations:

    .. math::

      \vec{y}_t = h(\vec{x}_t, \vec{v}_t)

    where:

    * :math:`\vec{y}_t` is a measurement vector of the form:

    .. math::

      \vec{y}_t = \begin{bmatrix}
                \phi \\
                r \\
                \dot{r}
            \end{bmatrix}

    * :math:`h` is a non-linear model function of the form:

    .. math::

      h(\vec{x}_t,\vec{v}_t) = \begin{bmatrix}
                atan2(\mathcal{y},\mathcal{x}) \\
                \sqrt{\mathcal{x}^2 + \mathcal{y}^2} \\
                (x\dot{x} + y\dot{y})/\sqrt{x^2 + y^2}
                \end{bmatrix} + \vec{v}_t

    * :math:`\vec{v}_t` is Gaussian distributed with covariance\
     :math:`R`, i.e.:

    .. math::

      \vec{v}_t \sim \mathcal{N}(0,R)

    .. math::

      R = \begin{bmatrix}
            \sigma_{\phi}^2 & 0 & 0\\
            0 & \sigma_{r}^2 & 0 \\
            0 & 0 & \sigma_{\dot{r}}^2
            \end{bmatrix}

    The :py:attr:`mapping` property of the model is a 3 element vector, \
    whose first (i.e. :py:attr:`mapping[0]`), second (i.e. \
    :py:attr:`mapping[1]`) and third (i.e. :py:attr:`mapping[2]`) elements \
    contain the state index of the :math:`x`, :math:`y` and :math:`z`  \
    coordinates, respectively.

    Note
    ----
    This class implementation assuming at 3D cartesian space, it therefore\
     expects a 6D state space.
    """

    translation_offset = Property(
        StateVector, default=None,
        doc="A 3x1 array specifying the origin offset in terms of :math:`x,y` coordinates.")
    velocity_mapping = Property(
        np.array, default=(1, 3, 5),
        doc="Mapping to the targets velocity within its state space")
    velocity = Property(
        StateVector, default=None,
        doc="A 3x1 array specifying the sensor velocity in terms of :math:`x,y,z` \
        coordinates.")

    def __init__(self, *args, **kwargs):
        """
        Ensure that the translation offset is initiated
        """
        super().__init__(*args, **kwargs)
        # Set values to defaults if not provided
        if self.translation_offset is None:
            self.translation_offset = StateVector([0] * 3)

        if self.velocity is None:
            self.velocity = StateVector([0] * 3)

    @property
    def ndim_meas(self) -> int:
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return 3

    def function(self, state, noise=False, **kwargs) -> StateVector:
        r"""Model function :math:`h(\vec{x}_t,\vec{v}_t)`

        Parameters
        ----------
        state: :class:`~.State`
            An input state
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is
            `False`, in which case no noise will be added
            if 'True', the output of :meth:`~.Model.rvs` is added)

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_state`, 1)
            The model function evaluated given the provided time interval.

        """

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs()
            else:
                noise = 0

        # Account for origin offset in position to enable range and angles to be determined
        xy_pos = state.state_vector[self.mapping, :] - self.translation_offset

        # Rotate coordinates based upon the sensor_velocity
        xy_rot = self._rotation_matrix @ xy_pos

        # Convert to Spherical
        rho, phi, _ = cart2sphere(*xy_rot)

        # Determine the net velocity component in the engagement
        xy_vel = state.state_vector[self.velocity_mapping, :] - self.velocity

        # Use polar to calculate range rate
        rr = np.dot(xy_pos[:, 0], xy_vel[:, 0]) / np.linalg.norm(xy_pos)

        return StateVector([[Bearing(phi)], [rho], [rr]]) + noise

    def rvs(self, num_samples=1, **kwargs) -> Union[StateVector, StateVectors]:
        out = super().rvs(num_samples, **kwargs)
        out = np.array([[Bearing(0)], [0.], [0.]]) + out
        return out


class CartesianToElevationBearingRangeRate(NonLinearGaussianMeasurement, ReversibleModel):
    r"""This is a class implementation of a time-invariant measurement model, \
    where measurements are assumed to be received in the form of elevation \
    (:math:`\theta`),  bearing (:math:`\phi`), range (:math:`r`) and
    range-rate (:math:'\dot{r}'), with Gaussian noise in each dimension.

    The model is described by the following equations:

    .. math::

      \vec{y}_t = h(\vec{x}_t, \vec{v}_t)

    where:

    * :math:`\vec{y}_t` is a measurement vector of the form:

    .. math::

      \vec{y}_t = \begin{bmatrix}
                \theta \\
                \phi \\
                r \\
                \dot{r}
            \end{bmatrix}

    * :math:`h` is a non-linear model function of the form:

    .. math::

      h(\vec{x}_t,\vec{v}_t) = \begin{bmatrix}
                asin(\mathcal{z}/\sqrt{\mathcal{x}^2 + \mathcal{y}^2 +\mathcal{z}^2}) \\
                atan2(\mathcal{y},\mathcal{x}) \\
                \sqrt{\mathcal{x}^2 + \mathcal{y}^2 + \mathcal{z}^2} \\
                (x\dot{x} + y\dot{y} + z\dot{z})/\sqrt{x^2 + y^2 + z^2}
                \end{bmatrix} + \vec{v}_t

    * :math:`\vec{v}_t` is Gaussian distributed with covariance :math:`R`, i.e.:

    .. math::

      \vec{v}_t \sim \mathcal{N}(0,R)

    .. math::

      R = \begin{bmatrix}
            \sigma_{\theta}^2 & 0 & 0 & 0\\
            0 & \sigma_{\phi}^2 & 0 & 0\\
            0 & 0 & \sigma_{r}^2 & 0\\
            0 & 0 & 0 & \sigma_{\dot{r}}^2
            \end{bmatrix}

    The :py:attr:`mapping` property of the model is a 3 element vector, \
    whose first (i.e. :py:attr:`mapping[0]`), second (i.e. \
    :py:attr:`mapping[1]`) and third (i.e. :py:attr:`mapping[2]`) elements \
    contain the state index of the :math:`x`, :math:`y` and :math:`z`  \
    coordinates, respectively.

    Note
    ----
    This class implementation assuming at 3D cartesian space, it therefore\
    expects a 6D state space.
    """

    translation_offset = Property(
        StateVector, default=None,
        doc="A 3x1 array specifying the origin offset in terms of :math:`x,y,z` coordinates.")
    velocity_mapping = Property(
        np.array, default=(1, 3, 5),
        doc="Mapping to the targets velocity within its state space")
    velocity = Property(
        StateVector, default=None,
        doc="A 3x1 array specifying the sensor velocity in terms of :math:`x,y,z` coordinates.")

    def __init__(self, *args, **kwargs):
        """
        Ensure that the translation offset is initiated
        """
        super().__init__(*args, **kwargs)
        # Set values to defaults if not provided
        if self.translation_offset is None:
            self.translation_offset = StateVector([0] * 3)

        if self.velocity is None:
            self.velocity = StateVector([0] * 3)

    @property
    def ndim_meas(self) -> int:
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return 4

    def function(self, state, noise=False, **kwargs) -> StateVector:
        r"""Model function :math:`h(\vec{x}_t,\vec{v}_t)`

        Parameters
        ----------
        state: :class:`~.StateVector`
            An input state vector for the target

        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is
            `False`, in which case no noise will be added
            if 'True', the output of :meth:`~.Model.rvs` is added)

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_state`, 1)
            The model function evaluated given the provided time interval.
        """

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs()
            else:
                noise = 0

        # Account for origin offset in position to enable range and angles to be determined
        xyz_pos = state.state_vector[self.mapping, :] - self.translation_offset

        # Rotate coordinates based upon the sensor_velocity
        xyz_rot = self._rotation_matrix @ xyz_pos

        # Convert to Spherical
        rho, phi, theta = cart2sphere(*xyz_rot)

        # Determine the net velocity component in the engagement
        xyz_vel = state.state_vector[self.velocity_mapping, :] - self.velocity

        # Use polar to calculate range rate
        rr = np.dot(xyz_pos[:, 0], xyz_vel[:, 0]) / np.linalg.norm(xyz_pos)

        return StateVector([[Elevation(phi)],
                            [Bearing(theta)],
                            [rho],
                            [rr]]) + noise

    def inverse_function(self, detection, **kwargs) -> StateVector:
        phi, theta, rho, rho_rate = detection.state_vector

        x, y, z = sphere2cart(rho, theta, phi)
        # because only rho_rate is known, only the components in
        # x,y and z of the range rate can be found.
        x_rate = np.cos(phi) * np.cos(theta) * rho_rate
        y_rate = np.cos(phi) * np.sin(theta) * rho_rate
        z_rate = np.sin(phi) * rho_rate

        inv_rotation_matrix = inv(self._rotation_matrix)

        out_vector = StateVector([[0.], [0.], [0.], [0.], [0.], [0.]])
        out_vector[self.mapping, 0] = x, y, z
        out_vector[self.velocity_mapping, 0] = x_rate, y_rate, z_rate

        out_vector[self.mapping, :] = inv_rotation_matrix @ out_vector[self.mapping, :]
        out_vector[self.velocity_mapping, :] = \
            inv_rotation_matrix @ out_vector[self.velocity_mapping, :]

        out_vector[self.mapping, :] = out_vector[self.mapping, :] + self.translation_offset

        return out_vector

    def rvs(self, num_samples=1, **kwargs) -> Union[StateVector, StateVectors]:
        out = super().rvs(num_samples, **kwargs)
        out = np.array([[Elevation(0)], [Bearing(0)], [0.], [0.]]) + out
        return out
