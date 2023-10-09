import copy
from abc import ABC
from math import sqrt
from typing import Sequence, Tuple, Union

import numpy as np
from scipy.linalg import inv, pinv, block_diag
from scipy.stats import multivariate_normal

from .base import MeasurementModel
from ..base import LinearModel, GaussianModel, ReversibleModel
from ...base import Property, clearable_cached_property
from ...functions import cart2pol, pol2cart, \
    cart2sphere, sphere2cart, cart2angles, \
    build_rotation_matrix, sphererate2cartrate, cartrate2sphererate, \
    jacobian as compute_jacobian
from ...types.angle import Bearing, Elevation
from ...types.array import StateVector, CovarianceMatrix, StateVectors
from ...types.numeric import Probability
from ...types.state import GaussianState, State


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
    model_list: Sequence[GaussianModel] = Property(doc="List of Measurement Models.")

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

    @property
    def mapping(self):
        return [x for model in self.model_list for x in model.mapping]

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


class NonLinearGaussianMeasurement(MeasurementModel, GaussianModel, ABC):
    r"""This class combines the MeasurementModel, NonLinearModel and \
    GaussianModel classes. It is not meant to be instantiated directly \
    but subclasses should be derived from this class.
    """
    noise_covar: CovarianceMatrix = Property(doc="Noise covariance")
    rotation_offset: StateVector = Property(
        default=None,
        doc="A 3x1 array of angles (rad), specifying the clockwise rotation "
            "around each Cartesian axis in the order :math:`x,y,z`. "
            "The rotation angles are positive if the rotation is in the "
            "counter-clockwise direction when viewed by an observer looking "
            "along the respective rotation axis, towards the origin.")

    def __init__(self, *args, **kwargs):
        """
        Ensure that the rotation offset is initiated
        """
        super().__init__(*args, **kwargs)
        # Set values to defaults if not provided
        if self.rotation_offset is None:
            self.rotation_offset = StateVector([[0], [0], [0]])

        if not isinstance(self.noise_covar, CovarianceMatrix):
            self.noise_covar = CovarianceMatrix(self.noise_covar)

    def covar(self, **kwargs) -> CovarianceMatrix:
        """Returns the measurement model noise covariance matrix.

        Returns
        -------
        :class:`~.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_meas`, :py:attr:`~ndim_meas`)
            The measurement noise covariance.
        """

        return self.noise_covar

    @clearable_cached_property('rotation_offset')
    def rotation_matrix(self) -> np.ndarray:
        """3D axis rotation matrix"""
        return build_rotation_matrix(self.rotation_offset)


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
    :py:attr:`mapping[1]`) and third (i.e. :py:attr:`mapping[2]`) elements \
    contain the state index of the :math:`x`, :math:`y` and :math:`z`  \
    coordinates, respectively.

    Note
    ----
    The current implementation of this class assumes a 3D Cartesian plane.

    """  # noqa:E501

    translation_offset: StateVector = Property(
        default=None,
        doc="A 3x1 array specifying the Cartesian origin offset in terms of :math:`x,y,z` "
            "coordinates.")

    def __init__(self, *args, **kwargs):
        """
        Ensure that the translation offset is initiated
        """
        super().__init__(*args, **kwargs)
        # Set values to defaults if not provided
        if self.translation_offset is None:
            self.translation_offset = StateVector([0] * 3)

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
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0

        # Account for origin offset
        xyz = state.state_vector[self.mapping, :] - self.translation_offset

        # Rotate coordinates
        xyz_rot = self.rotation_matrix @ xyz

        # Convert to Spherical
        rho, phi, theta = cart2sphere(xyz_rot[0, :], xyz_rot[1, :], xyz_rot[2, :])
        elevations = [Elevation(i) for i in theta]
        bearings = [Bearing(i) for i in phi]

        return StateVectors([elevations, bearings, rho]) + noise

    def inverse_function(self, detection, **kwargs) -> StateVector:

        theta, phi, rho = detection.state_vector
        xyz = StateVector(sphere2cart(rho, phi, theta))

        inv_rotation_matrix = inv(self.rotation_matrix)
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
    :py:attr:`mapping[1]`) elements contain the state index of the \
    :math:`x` and :math:`y` coordinates, respectively.

    Note
    ----
    The current implementation of this class assumes a 2D Cartesian plane.

    """  # noqa:E501

    translation_offset: StateVector = Property(
        default=None,
        doc="A 2x1 array specifying the origin offset in terms of :math:`x,y` coordinates.")

    def __init__(self, *args, **kwargs):
        """
        Ensure that the translation offset is initiated
        """
        super().__init__(*args, **kwargs)
        # Set values to defaults if not provided
        if self.translation_offset is None:
            self.translation_offset = StateVector([0] * len(self.mapping))

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

        x, y = pol2cart(detection.state_vector[1, :], detection.state_vector[0, :])

        xyz = np.array([x, y, np.zeros(detection.state_vector.shape[1])])
        inv_rotation_matrix = inv(self.rotation_matrix)
        xyz = inv_rotation_matrix @ xyz
        xy = xyz[0:2]

        res = np.zeros((self.ndim_state, detection.state_vector.shape[1])).view(StateVector)
        res[self.mapping[:2], :] = xy + self.translation_offset[:2, :]

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
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0

        # Account for origin offset
        xyz = np.array([state.state_vector[self.mapping[0], :] - self.translation_offset[0, 0],
                        state.state_vector[self.mapping[1], :] - self.translation_offset[1, 0],
                        [0] * state.state_vector.shape[1]
                        ])

        # Rotate coordinates
        xyz_rot = self.rotation_matrix @ xyz

        # Covert to polar
        rho, phi = cart2pol(*xyz_rot[:2, :])
        bearings = [Bearing(i) for i in phi]
        return StateVectors([bearings, rho]) + noise

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

    translation_offset: StateVector = Property(
        default=None,
        doc="A 3x1 array specifying the origin offset in terms of :math:`x,y,z` coordinates.")

    def __init__(self, *args, **kwargs):
        """
        Ensure that the translation offset is initiated
        """
        super().__init__(*args, **kwargs)
        # Set values to defaults if not provided
        if self.translation_offset is None:
            self.translation_offset = StateVector([0] * 3)

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
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0

        # Account for origin offset
        xyz = state.state_vector[self.mapping, :] - self.translation_offset

        # Rotate coordinates
        xyz_rot = self.rotation_matrix @ xyz

        # Convert to Angles
        phi, theta = cart2angles(xyz_rot[0, :], xyz_rot[1, :], xyz_rot[2, :])

        bearings = [Bearing(i) for i in phi]
        elevations = [Elevation(i) for i in theta]
        return StateVectors([elevations, bearings]) + noise

    def rvs(self, num_samples=1, **kwargs) -> Union[StateVector, StateVectors]:
        out = super().rvs(num_samples, **kwargs)
        out = np.array([[Elevation(0.)], [Bearing(0.)]]) + out
        return out


class Cartesian2DToBearing(NonLinearGaussianMeasurement):
    r"""This is a class implementation of a time-invariant measurement model, where measurements \
    are assumed to be received in the form of bearing (:math:`\phi`) with Gaussian noise.

    The model is described by the following equations:

    .. math::

      \phi_t = h(\vec{x}_t, v_t)

    * :math:`h` is a non-linear model function of the form:

    .. math::

      h(\vec{x}_t,v_t) = atan2(\mathcal{y},\mathcal{x}) + v_t

    * :math:`v_t` is Gaussian distributed with covariance :math:`R`, i.e.:

    .. math::

      v_t \sim \mathcal{N}(0,\sigma_{\phi}^2)

    The :py:attr:`mapping` property of the model is a 2 element vector, whose first \
    (i.e. :py:attr:`mapping[0]`) and second (i.e. :py:attr:`mapping[1]`) elements contain the \
    state index of the :math:`x` and :math:`y` coordinates, respectively.

    """  # noqa:E501

    translation_offset: StateVector = Property(
        default=None,
        doc="A 2x1 array specifying the origin offset in terms of :math:`x,y` coordinates.")

    def __init__(self, *args, **kwargs):
        """
        Ensure that the translation offset is initiated
        """
        super().__init__(*args, **kwargs)
        # Set values to defaults if not provided
        if self.translation_offset is None:
            self.translation_offset = StateVector([0] * 2)

    @property
    def ndim_meas(self):
        """ndim_meas getter method

            Returns
            -------
            :class:`int`
                The number of measurement dimensions
            """
        return 1

    def function(self, state, noise=False, **kwargs):
        r"""Model function :math:`h(\vec{x}_t,v_t)`

            Parameters
            ----------
            state: :class:`~.State`
                An input state
            noise: :class:`numpy.ndarray` or bool
                An externally generated random process noise sample (the default is `False`, in
                which case no noise will be added.
                If 'True', the output of :meth:`~.Model.rvs` is added)

            Returns
            -------
            :class:`numpy.ndarray` of shape (:py:attr:`~ndim_state`, 1)
                The model function evaluated given the provided time interval.
            """

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0

        # Account for origin offset
        xyz = np.array([state.state_vector[self.mapping[0], :] - self.translation_offset[0, 0],
                        state.state_vector[self.mapping[1], :] - self.translation_offset[1, 0],
                        [0] * state.state_vector.shape[1]
                        ])

        # Rotate coordinates
        xyz_rot = self.rotation_matrix @ xyz

        # Covert to polar
        _, phi = cart2pol(*xyz_rot[:2, :])
        bearings = [Bearing(i) for i in phi]

        return StateVectors([bearings]) + noise

    def rvs(self, num_samples=1, **kwargs) -> Union[StateVector, StateVectors]:
        out = super().rvs(num_samples, **kwargs)
        out = np.array([[Bearing(0.)]]) + out
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

    * :math:`\vec{v}_t` is Gaussian distributed with covariance
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
    This class implementation assuming at 3D cartesian space, it therefore \
    expects a 6D state space.
    """

    translation_offset: StateVector = Property(
        default=None,
        doc="A 3x1 array specifying the origin offset in terms of :math:`x,y` coordinates.")
    velocity_mapping: Tuple[int, int, int] = Property(
        default=(1, 3, 5),
        doc="Mapping to the targets velocity within its state space")
    velocity: StateVector = Property(
        default=None,
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
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0

        # Account for origin offset in position to enable range and angles to be determined
        xy_pos = state.state_vector[self.mapping, :] - self.translation_offset

        # Rotate coordinates based upon the sensor_velocity
        xy_rot = self.rotation_matrix @ xy_pos

        # Convert to Spherical
        rho, phi, _ = cart2sphere(xy_rot[0, :], xy_rot[1, :], xy_rot[2, :])

        # Determine the net velocity component in the engagement
        xy_vel = state.state_vector[self.velocity_mapping, :] - self.velocity

        # Use polar to calculate range rate
        rr = np.einsum('ij,ij->j', xy_pos, xy_vel) / np.linalg.norm(xy_pos, axis=0)

        # Convert to bearings
        bearings = [Bearing(i) for i in phi]

        return StateVectors([bearings, rho, rr]) + noise

    def rvs(self, num_samples=1, **kwargs) -> Union[StateVector, StateVectors]:
        out = super().rvs(num_samples, **kwargs)
        out = np.array([[Bearing(0)], [0.], [0.]]) + out
        return out


class CartesianToElevationBearingRangeRate(NonLinearGaussianMeasurement, ReversibleModel):
    r"""This is a class implementation of a time-invariant measurement model, \
    where measurements are assumed to be received in the form of elevation \
    (:math:`\theta`),  bearing (:math:`\phi`), range (:math:`r`) and
    range-rate (:math:`\dot{r}`), with Gaussian noise in each dimension.

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
    This class implementation assuming at 3D cartesian space, it therefore \
    expects a 6D state space.
    """

    translation_offset: StateVector = Property(
        default=None,
        doc="A 3x1 array specifying the origin offset in terms of :math:`x,y,z` coordinates.")
    velocity_mapping: Tuple[int, int, int] = Property(
        default=(1, 3, 5),
        doc="Mapping to the targets velocity within its state space")
    velocity: StateVector = Property(
        default=None,
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
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0

        # Account for origin offset in position to enable range and angles to be determined
        xyz_pos = state.state_vector[self.mapping, :] - self.translation_offset

        # Rotate coordinates based upon the sensor_velocity
        xyz_rot = self.rotation_matrix @ xyz_pos

        # Convert to Spherical
        rho, phi, theta = cart2sphere(xyz_rot[0, :], xyz_rot[1, :], xyz_rot[2, :])

        # Determine the net velocity component in the engagement
        xyz_vel = state.state_vector[self.velocity_mapping, :] - self.velocity

        # Use polar to calculate range rate
        rr = np.einsum('ij,ij->j', xyz_pos, xyz_vel) / np.linalg.norm(xyz_pos, axis=0)

        bearings = [Bearing(i) for i in phi]
        elevations = [Elevation(i) for i in theta]
        return StateVectors([elevations,
                             bearings,
                             rho,
                             rr]) + noise

    def inverse_function(self, detection, **kwargs) -> StateVector:
        theta, phi, rho, rho_rate = detection.state_vector

        x, y, z = sphere2cart(rho, phi, theta)
        # because only rho_rate is known, only the components in
        # x,y and z of the range rate can be found.
        x_rate = np.cos(theta) * np.cos(phi) * rho_rate
        y_rate = np.cos(theta) * np.sin(phi) * rho_rate
        z_rate = np.sin(theta) * rho_rate

        inv_rotation_matrix = inv(self.rotation_matrix)

        out_vector = StateVector([[0.], [0.], [0.], [0.], [0.], [0.]])
        out_vector[self.mapping, 0] = x, y, z
        out_vector[self.velocity_mapping, 0] = x_rate, y_rate, z_rate

        out_vector[self.mapping, :] = inv_rotation_matrix @ out_vector[self.mapping, :]
        out_vector[self.velocity_mapping, :] = \
            inv_rotation_matrix @ out_vector[self.velocity_mapping, :]

        out_vector[self.mapping, :] = out_vector[self.mapping, :] + self.translation_offset
        out_vector[self.velocity_mapping, :] = out_vector[self.velocity_mapping, :] + \
            self.velocity

        return out_vector

    def rvs(self, num_samples=1, **kwargs) -> Union[StateVector, StateVectors]:
        out = super().rvs(num_samples, **kwargs)
        out = np.array([[Elevation(0)], [Bearing(0)], [0.], [0.]]) + out
        return out

    def jacobian(self, state, **kwargs):
        """Model jacobian matrix :math:`H_{jac}`

        Parameters
        ----------
        state : :class:`~.State`
            An input state

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_meas`, \
        :py:attr:`~ndim_state`)
            The model jacobian matrix evaluated around the given state vector.
        """
        # Account for origin offset in position to enable range and angles to be determined
        xyz_pos = state.state_vector[self.mapping, :] - self.translation_offset

        # Determine the net velocity component in the engagement
        xyz_vel = state.state_vector[self.velocity_mapping, :] - self.velocity

        # Rotate into RADAR coordinate system to linearize around the correct
        # state
        xyz_pos = self.rotation_matrix @ xyz_pos
        xyz_vel = self.rotation_matrix @ xyz_vel

        jac = np.zeros((4, 6), dtype=np.float_)

        x, y, z = xyz_pos
        vx, vy, vz = xyz_vel
        x2, y2, z2 = x**2, y**2, z**2
        x2y2 = x2 + y2
        r2 = x2y2 + z2
        r = sqrt(r2)
        sqrt_x2_y2 = sqrt(x2y2)
        r32 = r2*r

        # Jacobian encodes partial derivatives of measurement vector components
        # Y = <theta, phi, r, rdot> against state vector
        # X = <x, vx, y, vy, z, vz>.

        # dtheta/dx
        sqrt_x2_y2r2 = sqrt_x2_y2*r2
        jac[0, 0] = -(x*z)/(sqrt_x2_y2r2)

        # dtheta/dy
        jac[0, 2] = -(y*z)/(sqrt_x2_y2r2)

        # dthtea/dz
        jac[0, 4] = sqrt_x2_y2/r2

        # dphi/dx
        jac[1, 0] = - y/(x2y2)

        # dphi/dy
        jac[1, 2] = x/(x2y2)

        # dphi/dz = 0

        # dr/dx and drdot/dvx
        jac[2, 0] = jac[3, 1] = x/r

        # dr/dx and drdot/dvy
        jac[2, 2] = jac[3, 3] = y/r

        # dr/dx and drdot/dvz
        jac[2, 4] = jac[3, 5] = z/r

        vx_x, vy_y, vz_z = vx*x, vy*y, vz*z

        # drdot/dx
        jac[3, 0] = (-x*(vy_y + vz_z) + vx*(y2 + z2))/r32

        # drdot/dy
        jac[3, 2] = (vy*(x2 + z2) - y*(vx_x + vz_z))/r32

        # drdot/dz
        jac[3, 4] = (vz*(x2y2) - (vx_x + vy_y)*z)/r32

        # Up to this point, the Jacobian has been with respect to the state
        # vector after rotating into the RADAR coordinate system. However, we
        # want the Jacobian with respect to world state vector, so we must post
        # multiply Jacobian by the RADAR rotation matrix.
        jac[:, self.mapping] = jac[:, self.mapping] @ self.rotation_matrix
        jac[:, self.velocity_mapping] = jac[:, self.velocity_mapping] @ self.rotation_matrix

        return jac


class CartesianRateToElevationRateBearingRateRangeRate(NonLinearGaussianMeasurement,
                                                       ReversibleModel):
    r"""This is a class implementation of a time-invariant measurement model, where states are
    assumed to be received in the form of bearing (:math:`\phi`), bearing rate
    (:math:`\dot{\phi}`) elevation (:math:`\theta`), elevation rate (:math:`\dot{\theta}`), range
    (:math:`r`) and range rate (:math:`\dot{r}`), with Gaussian noise in each dimension.

    The model is described by the following equations:

    .. math::

      \vec{y}_t = h(\vec{x}_t, \vec{v}_t)

    where:

    * :math:`\vec{y}_t` is a measurement vector of the form:

    .. math::

      \vec{y}_t = \begin{bmatrix}
                \theta \\
                \dot{\theta}\\
                \phi \\
                \dot{\phi}\\
                r\\
                \dot{r}
            \end{bmatrix}

    * :math:`h` is a non-linear model function of the form:

    .. math::

      h(\vec{x}_t,\vec{v}_t) = \begin{bmatrix}
                \arcsin\left(\frac{\mathcal{z}}{\sqrt{\mathcal{x}^2 + \mathcal{y}^2 +\mathcal{z}^2}}\right) \\
                \frac{\frac{\dot{\mathcal{z}}}{\sqrt{\mathcal{x}^2+\mathcal{y}^2+\mathcal{z}^2}}-\frac{\mathcal{z}\left(2\mathcal{x}\dot{\mathcal{x}}+2\mathcal{y}\dot{\mathcal{y}}+2\mathcal{z}\dot{\mathcal{z}}\right)}{2(\mathcal{x}^2+\mathcal{y}^2+\mathcal{z}^2)^\frac{3}{2}}}{\sqrt{1-\frac{\mathcal{z}^2}{\mathcal{x}^2+\mathcal{y}^2+\mathcal{z}^2}}}\\
                \arctan2(\mathcal{y},\mathcal{x}) \\
                \frac{\mathcal{x}\dot{\mathcal{y}}- \mathcal{y}\dot{\mathcal{x}}}{\mathcal{x}^2+\mathcal{y}^2}\\
                \sqrt{\mathcal{x}^2 + \mathcal{y}^2 + \mathcal{z}^2}\\
                \frac{\mathcal{x}\dot{\mathcal{x}}+\mathcal{y}\dot{\mathcal{y}}+\mathcal{z}\dot{\mathcal{z}}}{\sqrt{\mathcal{x}^2+\mathcal{y}^2+\mathcal{z}^2}}
                \end{bmatrix} + \vec{v}_t

    * :math:`\vec{v}_t` is Gaussian distributed with covariance :math:`R`, i.e.:

    .. math::

      \vec{v}_t \sim \mathcal{N}(0,R)

    .. math::

      R = \begin{bmatrix}
            \sigma_{\theta}^2 & 0 & 0 & 0 & 0 & 0\\
            0 & \sigma_{\dot{\theta}}^2 & 0 & 0 & 0 & 0\\
            0 & 0 & \sigma_{\phi}^2 & 0 & 0  & 0 \\
            0 & 0 & 0 & \sigma_{\dot{\phi}}^2 & 0 & 0 \\
            0 & 0 & 0 & 0& \sigma_{\mathcal{r}}^2  & 0\\
            0 & 0 & 0 & 0 & 0 & \sigma_{\dot{\mathcal{r}}}^2
            \end{bmatrix}

    The :py:attr:`mapping` property of the model is the 6 element vector, whose elements contain
    the state index of the :math:`x`, :math:`\dot{x}`, :math:`y`, :math:`\dot{y}`, :math:`z` and
    :math:`\dot{z}` coordinates, respectively.

    Note
    ----
    The current implementation of this class assumes a 3D Cartesian plane.

    """  # noqa:E501

    translation_offset: StateVector = Property(
        default=None,
        doc="A 3x1 array specifying the Cartesian origin offset in terms of :math:`x,y,z` "
            "coordinates.")

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        # Set values to defaults if not provided
        # Ensure that the translation offset is initiated as a 6-dimension vector.
        if self.translation_offset is None:
            self.translation_offset = StateVector([0] * 6)
        if len(self.translation_offset) == 3:
            self.translation_offset = \
                np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0],
                          [0, 0, 0], [0, 0, 1], [0, 0, 0]]) @ self.translation_offset
        if len(self.translation_offset) == 2:
            self.translation_offset = np.array([[1, 0], [0, 0], [0, 1],
                                                [0, 0], [0, 0], [0, 0]]) @ self.translation_offset
        self.rot_resize = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0],
                                    [0, 0, 0], [0, 0, 1], [0, 0, 0]])
        self.mapping = (0, 1, 2, 3, 4, 5)

    @property
    def ndim_meas(self) -> int:
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return 6

    @clearable_cached_property('rotation_offset')
    def rotation_matrix(self) -> np.ndarray:
        """rotation_matrix getter method

        Calculates and returns the (3D) axis rotation matrix for full 3d 6-state state vector of
        position and velocity.

        Returns
        -------
        :class:`numpy.ndarray` of shape (6, 6)
            The model (3D) rotation matrix.
        """
        RR = block_diag(build_rotation_matrix(self.rotation_offset),
                        build_rotation_matrix(self.rotation_offset))
        A = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1]])
        return A @ RR @ A.T

    def function(self, state, **kwargs) -> StateVector:
        r"""Model function :math:`h(\vec{x}_t,\vec{v}_t)`

        Parameters
        ----------
        state: :class:`~.State`
            An input state

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_state`, 1)
            The model function evaluated given the provided time interval.
        """

        # Account for origin offset
        # self.mapping is (0,1,2,3,4,5) and translation offset dimensions are 6 by 1
        xyz = state.state_vector[self.mapping, :] - self.translation_offset

        # Rotate coordinates
        xyz_rot = self.rotation_matrix @ xyz

        # Convert to Spherical
        theta, dtheta, phi, dphi, rho, drho = cartrate2sphererate(*(xyz_rot[i, :]
                                                                    for i in range(6)))
        elevations = [Elevation(i) for i in np.atleast_1d(theta)]
        bearings = [Bearing(i) for i in np.atleast_1d(phi)]
        rhos = np.atleast_1d(rho)
        drhos = np.atleast_1d(drho)
        dbearings = np.atleast_1d(dphi)
        delevations = np.atleast_1d(dtheta)

        return StateVectors([elevations, delevations,
                             bearings, dbearings,
                             rhos, drhos])

    def inverse_function(self, state, **kwargs) -> StateVector:
        """Inverse function

        Parameters
        ----------
        state: :class:`~.State`
            An input state (in Spherical polar position and velocity)

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_state`, 1)
            The inverse model function evaluated given the provided time interval.
        """
        # theta, dtheta, phi, dphi, rho, drho = state.state_vector
        xyz = StateVectors(sphererate2cartrate(
            np.atleast_1d(state.state_vector[0, :]),
            np.atleast_1d(state.state_vector[1, :]),
            np.atleast_1d(state.state_vector[2, :]),
            np.atleast_1d(state.state_vector[3, :]),
            np.atleast_1d(state.state_vector[4, :]),
            np.atleast_1d(state.state_vector[5, :])))

        xyz = inv(self.rotation_matrix) @ xyz

        res = xyz
        res[self.mapping, :] = xyz[self.mapping, :] + self.translation_offset

        return res

    def cart2ebr(self, state: State):
        """Conversion of Cartesian state to Spherical State function

        Parameters
        ----------
        state: :class:`~.State`
            An input state (Cartesian)

        Returns
        -------
        :class:`~.State` in Spherical polar.
        """
        # TODO: Check conversion with tests
        new_state = State.from_state(state)
        new_sv = self.function(state, noise=False)
        new_sv[0] = Elevation(new_sv[0])
        new_sv[2] = Bearing(new_sv[2])
        new_state.state_vector = new_sv
        if isinstance(state, GaussianState):
            J = self.jacobian(state)
            new_covar = J @ state.covar @ J.T
            new_state.covar = new_covar
        return new_state

    def ebr2cart(self, state):
        """Conversion of Cartesian state to Spherical State function

        Parameters
        ----------
        state: :class:`~.State`
            An input state (Cartesian)

        Returns
        -------
        :class:`~.State` in Spherical polar.
        """
        # TODO: Check conversion with tests
        new_state = State.from_state(state)
        new_sv = self.inverse_function(state, noise=False)
        new_state.state_vector = new_sv
        if isinstance(state, GaussianState):
            J = self.jacobian(state, self.inverse_function)
            new_covar = J @ state.covar @ J.T
            new_state.covar = new_covar
        return new_state

    def jacobian(self, state, function=None, **kwargs):
        """Model Jacobian matrix :math:`H_{jac}` using a given function. The aim of this updated
        version of the :meth:`function` allows the use of defining either the function or the
        :meth:`inverse_function` when running the Jacobian function.

        Parameters
        ----------
        state : :class:`~.State`
            An input state
        function : function handle
            A (non-linear) transition function
            Must be of the form "y = fun(x)", where y can be a scalar or \
            :class:`numpy.ndarray` of shape `(Nd, 1)` or `(Nd,)`

        Returns
        -------
        :class:`numpy.ndarray` of shape (attr:`~ndim_meas`, :attr:`~ndim_state`)
            The model Jacobian matrix evaluated around the given state vector.
        """
        if function is None:
            function = self.function

        def fun(x):
            return function(x, **kwargs)

        return compute_jacobian(fun, state)


class RangeRangeRateBinning(CartesianToElevationBearingRangeRate):
    r"""This is a class implementation of a time-invariant measurement model, \
    where measurements are assumed to be in the form of elevation \
    (:math:`\theta`),  bearing (:math:`\phi`), range (:math:`r`) and
    range-rate (:math:`\dot{r}`), with Gaussian noise in each dimension and the
    range and range-rate are binned based on the
    range resolution and range-rate resolution respectively.

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
                \textrm{asin}(\mathcal{z}/\sqrt{\mathcal{x}^2 + \mathcal{y}^2 +\mathcal{z}^2}) \\
                \textrm{atan2}(\mathcal{y},\mathcal{x}) \\
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

    The covariances for radar are determined by different factors. The angle error
    is affected by the radar beam width. Range error is affected by the SNR and pulse bandwidth.
    The error for the range rate is dependent on the dwell time.
    The range and range rate are binned to the centre of the cell using

    .. math::

        x = \textrm{floor}(x/\Delta x)*\Delta x + \frac{\Delta x}{2}

    The :py:attr:`mapping` property of the model is a 3 element vector, \
    whose first (i.e. :py:attr:`mapping[0]`), second (i.e. \
    :py:attr:`mapping[1]`) and third (i.e. :py:attr:`mapping[2]`) elements \
    contain the state index of the :math:`x`, :math:`y` and :math:`z`  \
    coordinates, respectively.

    The :py:attr:`velocity_mapping` property of the model is a 3 element vector, \
    whose first (i.e. :py:attr:`velocity_mapping[0]`), second (i.e. \
    :py:attr:`velocity_mapping[1]`) and third (i.e. :py:attr:`velocity_mapping[2]`) elements \
    contain the state index of the :math:`\dot{x}`, :math:`\dot{y}` and :math:`\dot{z}`  \
    coordinates, respectively.

    Note
    ----
    This class implementation assumes a 3D cartesian space, it therefore \
    expects a 6D state space.
    """

    range_res: float = Property(doc="Size of the range bins in m")
    range_rate_res: float = Property(doc="Size of the velocity bins in m/s")

    @property
    def ndim_meas(self):
        return 4

    def function(self, state, noise=False, **kwargs):
        r"""Model function :math:`h(\vec{x}_t,\vec{v}_t)`

        Parameters
        ----------
        state: :class:`~.StateVector`
            An input state vector for the target

        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is
            ``False``, in which case no noise will be added and no binning takes place
            if ``True``, the output of :attr:`~.Model.rvs` is added and the
            range and range rate are binned)

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_state`, 1)
            The model function evaluated given the provided time interval.

        """

        out = super().function(state, noise, **kwargs)

        if isinstance(noise, bool) or noise is None:
            if noise:
                out[2] = np.floor(out[2] / self.range_res) * self.range_res + self.range_res/2
                out[3] = np.floor(out[3] / self.range_rate_res) * \
                    self.range_rate_res + self.range_rate_res/2

        return out

    @classmethod
    def _gaussian_integral(cls, a, b, mean, cov):
        # this function is the cumulative probability ranging from a to b for a normal distribution
        return (multivariate_normal.cdf(a, mean=mean, cov=cov)
                - multivariate_normal.cdf(b, mean=mean, cov=cov))

    @classmethod
    def _binned_pdf(cls, measured_value, mean, bin_size, cov):
        # this function finds the probability density of the bin the measured_value is in
        a = np.floor(measured_value / bin_size) * bin_size + bin_size
        b = np.floor(measured_value / bin_size) * bin_size
        return cls._gaussian_integral(a, b, mean, cov)/bin_size

    def pdf(self, state1, state2, **kwargs):
        r"""Model pdf/likelihood evaluation function

        Evaluates the pdf/likelihood of ``state1``, given the state
        ``state2`` which is passed to :meth:`function()`.

        For the first 2 dimensions, this can be written as:

        .. math::

            p = p(y_t | x_t) = \mathcal{N}(y_t; x_t, Q)

        where :math:`y_t` = ``state_vector1``, :math:`x_t` = ``state_vector2``,
         :math:`Q` = :attr:`covar` and :math:`\mathcal{N}` is a normal distribution

        The probability for the binned dimensions, the last 2, can be written as:

        .. math::

            p = P(a \leq \mathcal{N} \leq b)

        In this equation a and b are the edges of the bin.

        Parameters
        ----------
        state1 : :class:`~.State`
        state2 : :class:`~.State`

        Returns
        -------
        : :class:`~.Probability`
            The likelihood of ``state1``, given ``state2``
        """

        # state1 is in measurement space
        # state2 is in state_space
        if (((state1.state_vector[2, 0]-self.range_res/2) / self.range_res).is_integer()
                and ((state1.state_vector[3, 0]-self.range_rate_res/2) /
                     self.range_rate_res).is_integer()):
            mean_vector = self.function(state2, noise=False, **kwargs)
            # pdf for the angles
            az_el_pdf = multivariate_normal.pdf(
                state1.state_vector[:2, 0],
                mean=mean_vector[:2, 0],
                cov=self.covar()[:2, :2])

            # pdf for the binned range and velocity
            range_pdf = self._binned_pdf(
                state1.state_vector[2, 0],
                mean_vector[2, 0],
                self.range_res,
                self.covar()[2, 2])
            velocity_pdf = self._binned_pdf(
                state1.state_vector[3, 0],
                mean_vector[3, 0],
                self.range_rate_res,
                self.covar()[3, 3])
            return Probability(range_pdf * velocity_pdf * az_el_pdf)
        else:
            return Probability(0)

    def logpdf(self, *args, **kwargs):
        # As pdf replaced, need to go to first non GaussianModel parent
        return super(ReversibleModel, self).logpdf(*args, **kwargs)
