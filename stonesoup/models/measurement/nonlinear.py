from abc import ABC
import copy
from typing import Sequence, Tuple, Union

from math import sqrt
import numpy as np
from scipy.linalg import inv, pinv, block_diag
from scipy.stats import multivariate_normal
from scipy.special import erf

from ...base import Property, clearable_cached_property
from ...types.numeric import Probability

from ...functions import cart2pol, pol2cart, \
    cart2sphere, sphere2cart, cart2angles, \
    build_rotation_matrix
from ...types.array import StateVector, CovarianceMatrix, StateVectors
from ...types.angle import Bearing, Elevation
from ..base import LinearModel, GaussianModel, ReversibleModel
from .base import MeasurementModel
from ...types.state import State


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
        x_rate = np.cos(phi) * np.cos(theta) * rho_rate
        y_rate = np.cos(phi) * np.sin(theta) * rho_rate
        z_rate = np.sin(phi) * rho_rate

        inv_rotation_matrix = inv(self.rotation_matrix)

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


class IsotropicPlume(GaussianModel, MeasurementModel):
    r"""This is a class implementing the isotropic plume model for
    approximating the resulting plume from gas release. Mathematical
    formulation of the algorithm can be seen in [1]_ and [2]_.
    The model assumes isotropic diffusivity and mean wind velocity,
    source strength and turbulent conditions.

    The model calculates the concentration level at a given location
    based on the provided source term. The model employs a sensing
    threshold for deciding if gas has been detected or if the reading
    is just sensor noise and turbulent conditions are accounted for
    using a missed detection probability. The source term if formed
    according to the following:

    .. math::
        \mathbf{S} = \left[\begin{array}{c}
                x \\
                y \\
                z \\
                Q \\
                u \\
                \phi \\
                \zeta_1 \\
                \zeta_2
            \end{array}\right],

    where :math:`x, y` and :math:`z` are the source position in 3D Cartesian
    space, :math:`Q` is the emission rate/strength in g/s, :math:`u` is the
    wind speed in m/s, :math:`\phi` is the wind direction in radians,
    :math:`\zeta_1` is the diffusivity of the gas in the environment and
    :math:`\zeta_2` is the lifetime of the gas.

    The concentration is calculated according to

    .. math::
        \begin{multline}
        \mathcal{M}(\vec{x}_k, \Theta_k) = \frac{Q}{4\pi\Vert\vec{x}_k-\vec{p}^s\Vert}
        \cdot\text{exp}\left[\frac{-\Vert\vec{x}_k-\vec{p}^s\Vert}{\lambda}\right]\cdot\\
        \text{exp}\left[\frac{-(x_k-x)u\cos\phi}{2\zeta_1}\right]
        \cdot\text{exp}\left[\frac{-(y_k-y)u\sin\phi}{2\zeta_1}\right],
        \end{multline}

    where :math:`\vec{x}_k` is the position of the sensor in 3D Cartesian space
    (:math:`[x_k\quad y_k\quad z_k]^\intercal`), :math:`\vec{p}^s` is the source location
    (:math:`[x\quad y\quad z]^\intercal`) and

    .. math::
        \lambda = \sqrt{\frac{\zeta_1\zeta_2}{1+\frac{(u^2\zeta_2)}{4\zeta_1}}}.

    References
    ----------
    .. [1] Vergassola, Massima & Villermaux, Emmanuel & Shraiman, Boris I. "'Infotaxis'
           as a strategy for searching without gradients", Nature, vol. 445, 406-409, 2007
    .. [2] Hutchinson, Michael & Liu, Cunjia & Chen, Wen-Hua, "Source term estimation of
           a hazardous airborne release using an unmanned aerial vehicle", Journal of Field
           Robotics, Vol. 36, 797-917, 2019
    """

    ndim_state: int = Property(
        default=8,
        doc="Number of state dimensions"
    )

    mapping: Sequence[int] = Property(
        default=tuple(range(0, 8)),
        doc="Mapping between measurement and state dims"
    )

    min_noise: float = Property(
        default=1e-4,
        doc="Minimum sensor noise"
    )

    standard_deviation_percentage: float = Property(
        default=0.5,
        doc="Standard deviation as a percentage of the concentration level"
    )

    translation_offset: StateVector = Property(
        default=None,
        doc="A 3x1 array specifying the Cartesian origin offset in terms of :math:`x,y,z` "
            "coordinates.")

    missed_detection_probability: Probability = Property(
        default=0.1,
        doc="The probability that the detection has detection has been affected by turbulence."
    )

    sensing_threshold: float = Property(
        default=0.1,
        doc="Measurement threshold. Should be set high enough to minimise false detections."
    )

    def __init__(self, *args, **kwargs):
        """
        Ensure that the translation offset is initiated
        """
        super().__init__(*args, **kwargs)
        # Set values to defaults if not provided
        if self.translation_offset is None:
            self.translation_offset = StateVector([0] * 3)

    def covar(self, **kwargs) -> CovarianceMatrix:
        raise NotImplementedError('Covariance for IsotropicPlume is dependant on the '
                                  'measurement as well as standard deviation!')

    @property
    def ndim_meas(self) -> int:
        return 1

    def function(self, state: State, noise: Union[bool, np.ndarray] = False, **kwargs) -> Union[
                 StateVector, StateVectors]:
        r"""Model function :math:`h(\vec{x}_t,\vec{v}_t)`

        Parameters
        ----------
        state: :class:`~.StateVector`
            An input source term state vector

        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is
            `False`, in which case no noise will be added
            if 'True', the output of :meth:`~.Model.rvs` is added). If `False`,
            then the model also does not consider the :attr:`sensing_threshold`
            and :attr:`missed_detection_probability`

        Returns
        -------
        : :class:`numpy.ndarray` of shape (1, 1)
            The model function evaluated with the provided source term
        """

        x, y, z, Q, u, phi, ci, cii = state.state_vector[self.mapping].view(np.ndarray)

        px, py, pz = self.translation_offset
        lambda_ = np.sqrt((ci * cii)/(1 + (u**2 * cii)/(4 * ci)))
        abs_dist = np.linalg.norm(state.state_vector[self.mapping[:3], :]
                                  - self.translation_offset, axis=0)

        # prevent divide by zero when converging on the source location
        abs_dist[abs_dist < 0.1] = 0.1

        C = Q / (4 * np.pi * ci * abs_dist) * np.exp(
            (-(px - x) * u * np.cos(phi) / (2 * ci)) + (-(py - y) * u * np.sin(phi) / (2 * ci))
            + (-1 * abs_dist / lambda_))

        C = np.atleast_2d(C)

        if noise:
            C += self.rvs(state=C.view(StateVectors),
                          num_samples=state.state_vector.shape[1],
                          **kwargs)
            # measurement thresholding
            C[C < self.sensing_threshold] = 0
            # missed detections
            flag = np.random.uniform(size=state.state_vector.shape[1]) \
                > (1 - self.missed_detection_probability)
            C[:, flag] = 0

        return C.view(StateVectors)

    def logpdf(self, state1: State, state2: State, **kwargs) -> Union[float, np.ndarray]:
        r"""Model log pdf/likelihood evaluation function

        Evaluates the log pdf/likelihood of ``state1``, given the state
        ``state2`` which is passed to :meth:`function()`.

        This function implements the likelihood functions from
        :meth:`~.pdf` that have been converted to the log space.

        Parameters
        ----------
        state1 : :class:`~.State`
        state2 : :class:`~.State`

        Returns
        -------
        :  float or :class:`~.numpy.ndarray`
            The log likelihood of ``state1``, given ``state2``
        """

        p_m = self.missed_detection_probability
        nd_sigma = self.sensing_threshold
        pred_meas = self.function(state2, **kwargs)
        if state1.state_vector[0] <= self.sensing_threshold:
            pdf = p_m + ((1-p_m) * 1/2 * (1+erf((self.sensing_threshold - pred_meas)
                                                / (nd_sigma * sqrt(2)))))
            likelihood = np.atleast_1d(np.log(pdf)).view(np.ndarray)

        else:
            d_sigma = self.standard_deviation_percentage * pred_meas + self.min_noise
            with np.errstate(divide="ignore"):
                likelihood = np.atleast_1d(np.log(1/(d_sigma*np.sqrt(2*np.pi)) *
                                                  np.exp((-(state1.state_vector-pred_meas)
                                                         ** 2)/(2*d_sigma**2)))).view(np.ndarray)

        if len(likelihood) == 1:
            likelihood = likelihood[0]

        return likelihood

    def pdf(self, state1: State, state2: State, **kwargs) -> Union[Probability, np.ndarray]:
        r"""Model pdf/likelihood evaluation function

        Evaluates the pdf/likelihood of ``state1``, given the state
        ``state2`` which is passed to :meth:`function()`.

        This function implements the following likelihood function,
        adapted from (12) in [2]_, removing the background sensor noise
        term.

        .. math::
            p(z_k|\Theta_k) = \frac{1}{\sigma_d\sqrt{2\pi}}\text{exp}
            \left(-\frac{(z_k-\hat{z}_k)^2}{2\sigma_d}\right),

        where :math:`z_k` = ``state1``, :math:`\Theta_k` = ``state2``,
        :math:`\hat{z}_k` = :meth:`~.Model.function` on ``state2``
        and :math:`\sigma_d` is the measurement standard deviation
        assuming the measurement arose from a true gas detection.
        This is given by

        .. math::
            \sigma_d = \sigma_{\text{percentage}} \cdot \hat{z} + \nu_{\text{min}},

        where :math:`\sigma_{\text{percentage}}` = :attr:`standard_deviation_percentage`
        and :math:`\nu_{\text{min}}` = :attr:`noise`. In the
        event that a measurement is below the sensor threshold or
        missed, a different liklihood function is used. This is
        given by

        .. math::
            p(z_k|\Theta_k) = (P_m) + \left((1-P_m)\cdot\frac{1}{2}\left[1+\text{erf}
            \left(\frac{z_{\text{thr}}-\hat{z}_k}{\sigma_m\sqrt{2}}\right)\right]\right),

        where :math:`P_m` = :attr:`missed_detection_probability`,
        :math:`\sigma_m` is the missed detection standard deviation
        which is implemented as equal to :attr:`sensing_threshold`
        and :math:`\text{erf}()` is the error function.

        Parameters
        ----------
        state1 : :class:`~.State`
        state2 : :class:`~.State`

        Returns
        -------
        : :class:`~.Probability`
            The likelihood of ``state1``, given ``state2``
        """
        return Probability.from_log_ufunc(self.logpdf(state1, state2, **kwargs))

    def rvs(self, state: Union[StateVector, StateVectors], num_samples: int = 1,
            random_state=None, **kwargs) -> Union[StateVector, StateVectors]:
        r"""Model noise/sample generation function

        Generates noise samples from the model. For this noise, the magnitude
        of sensor noise depends on the measurement. Thus, the noise term is given by

        .. math::
             \nu_k = \mathcal{N}\left(0,(\sigma_{\text{percentage}}\cdot z_k)^2\right).

        Parameters
        ----------
        state: :class:`~.StateVector` or :class:`~.StateVectors`
            The measured state (concentration for this model) used
            to scale the noise term.
        num_samples: scalar, optional
            The number of samples to be generated (the default is 1).

        Returns
        -------
        noise : 2-D array of shape (:attr:`ndim`, ``num_samples``)
            A set of Np samples, generated from the model's noise
            distribution.
        """

        random_state = random_state if random_state is not None else self.random_state

        generator = np.random.RandomState(random_state)
        noise = generator.normal(np.zeros(self.ndim_meas),
                                 np.ravel(state*self.standard_deviation_percentage),
                                 num_samples)

        noise = np.atleast_2d(noise)

        if num_samples == 1:
            return noise.view(StateVector)
        else:
            return noise.view(StateVectors)
