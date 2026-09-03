import numpy as np
from stonesoup.base import Property
from stonesoup.models.measurement.nonlinear import NonLinearGaussianMeasurement
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.array import StateVector, StateVectors, CovarianceMatrix
from stonesoup.sensor.sensor import SimpleSensor
from stonesoup.types.groundtruth import GroundTruthState
from stonesoup.types.detection import Detection
from stonesoup.models.base import ReversibleModel
from stonesoup.models.measurement.nonlinear import (
    CartesianToBearingRange,
    CartesianToElevationBearingRange,
)
from stonesoup.sensor.radar.radar import (
    RadarRotatingBearingRange,
    RadarRotatingElevationBearingRange,
)
from stonesoup.sensor.action.dwell_action import DwellActionsGenerator
from stonesoup.sensor.action.tilt_action import TiltActionsGenerator
from stonesoup.sensormanager.action import ActionableProperty
from stonesoup.types.angle import Angle
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.base import clearable_cached_property
from stonesoup.functions import build_rotation_matrix
from abc import abstractmethod

# from stonesoup.models.measurement.bias import BiasModelWrapper
from stonesoup.models.measurement import MeasurementModel
import copy
from stonesoup.types.state import State
from scipy.spatial.transform import Rotation
from stonesoup.functions import jacobian as compute_jac


class BiasModelWrapper(MeasurementModel):
    """Abstract wrapper that removes bias values from an existing MeasurementModel."""

    measurement_model: MeasurementModel = Property(
        doc="Unbiased model being wrapped that bias will be applied to"
    )
    state_mapping: list[int] = Property(
        doc="Mapping to state vector elements relevant to wrapped model"
    )
    bias_mapping: list[int] = Property(
        doc="Mapping to state vector elements where bias is"
    )

    @property
    def mapping(self):
        return list(self.measurement_model.mapping) + list(self.bias_mapping)

    @property
    def ndim_meas(self):
        return self.measurement_model.ndim_meas

    @abstractmethod
    def function(self, state, noise=False, **kwargs):
        raise NotImplementedError()

    def covar(self, *args, **kwargs):
        return self.measurement_model.covar(*args, **kwargs)

    def jacobian(self, state, **kwargs):
        return compute_jac(self.function, state, **kwargs)

    def pdf(self, *args, **kwargs):
        raise NotImplementedError()

    def rvs(self, *args, **kwargs):
        raise NotImplementedError()


class TranslationHeading2DBiasModelWrapper(BiasModelWrapper):
    """Removes an x, y translation and heading bias."""

    bias_mapping: list[int] = Property(
        default=(-3, -2, -1), doc="Mapping to x, y, heading elements in state vector"
    )

    def function(self, state, noise=False, **kwargs):
        state_vectors = []
        for state_vector in state.state_vector.view(StateVectors):
            bias_offset_elements = state_vector[self.bias_mapping, :]
            translation_bias_offset = bias_offset_elements[:2]
            heading_bias_offset = bias_offset_elements[2]
            bias_model = copy.copy(self.measurement_model)
            bias_model.translation_offset = (
                bias_model.translation_offset + translation_bias_offset
            )
            rotation = Rotation.from_euler(
                "xyz", bias_model.rotation_offset.flatten(), degrees=False
            ) * Rotation.from_euler("z", float(heading_bias_offset), degrees=False)
            bias_model.rotation_offset = rotation.as_euler(
                "xyz", degrees=False
            ).reshape(-1, 1)
            state_vectors.append(
                bias_model.function(
                    State(state_vector[self.state_mapping, :]), noise=noise, **kwargs
                )
            )
        if len(state_vectors) == 1:
            return state_vectors[0]
        else:
            return StateVectors(state_vectors)


class OffsetNonLinearGaussian(NonLinearGaussianMeasurement, ReversibleModel):
    translation_offset: StateVector = Property(
        default=None,
        doc="A 3x1 array specifying the Cartesian origin offset in terms of :math:`x,y,z` "
        "coordinates.",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.mapping) not in [2, 3]:
            raise ValueError("Measurement space must be 2 or 3 dimensions")

    @property
    def ndim_meas(self) -> int:
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return len(self.mapping)

    def function(self, state, noise=False, **kwargs):
        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0
        # project into 3d
        target_state_vector = np.concatenate(
            (
                state.state_vector[self.mapping, :],
                np.zeros((3, state.state_vector.shape[1])),
            )
        )[:3]
        translation_offset = np.concatenate(
            (self.translation_offset, np.zeros((3, 1)))
        )[:3]
        relative_state_vector = target_state_vector - translation_offset

        rotated_measurement_vector = self.rotation_matrix @ relative_state_vector
        return StateVectors(rotated_measurement_vector[: len(self.mapping), :]) + noise

    def inverse_function(self, detection, **kwargs):
        state_vector = np.concatenate(
            (
                detection.state_vector,
                np.zeros((3, detection.state_vector.shape[1])),
            )
        )[:3]
        xyz = self.rotation_matrix.T @ state_vector

        res = np.zeros((self.ndim_state, 1)).view(StateVector)
        res[self.mapping, :] = xyz[: self.ndim_meas, :] + self.translation_offset
        return res


class Lidar(SimpleSensor):
    """A  sensor that generates measurements of targets, using a
    :class:`~.OffsetNonLinearGaussian` model, relative to its position.

    Note
    ----
    This implementation of this class assumes a 3D Cartesian space.

    """

    ndim_state: int = Property(
        default=3,
        doc="Number of state dimensions. This is utilised by (and follows in format) "
        "the underlying :class:`~.OffsetNonLinearGaussian` model",
    )
    noise_covar: CovarianceMatrix = Property(
        doc="The sensor noise covariance matrix. This is utilised by "
        "(and follows in format) the underlying "
        ":class:`~.OffsetNonLinearGaussian` model"
    )
    position_mapping: tuple[int] = Property(
        doc="Mapping between the target's state space and the sensor's "
        "measurement capability"
    )
    max_range: float = Property(
        default=np.inf, doc="The maximum detection range of the radar (in meters)"
    )

    @property
    def measurement_model(self):
        return OffsetNonLinearGaussian(
            ndim_state=self.ndim_state,
            mapping=self.position_mapping,
            noise_covar=self.noise_covar,
            translation_offset=self.position,
            rotation_offset=self.orientation,
        )

    def is_detectable(self, state: GroundTruthState, measurement_model=None) -> bool:
        return True

    def is_clutter_detectable(self, state: Detection) -> bool:
        return True


class Lidar2D(Lidar):
    max_range: float = Property(
        default=np.inf, doc="The maximum detection range of the radar (in meters)"
    )
    dwell_centre: StateVector = ActionableProperty(
        doc="A `state_vector` property that describes the rotation angle of the centre of the "
        "sensor's current FOV (i.e. the dwell centre) relative to the positive x-axis of the "
        "sensor frame/orientation. The angle is positive if the rotation is in the "
        "counter-clockwise direction when viewed by an observer looking down the z-axis of "
        "the sensor frame, towards the origin. Angle units are in radians",
        generator_cls=DwellActionsGenerator,
        generator_kwargs_mapping={"rpm": "rpm", "resolution": "resolution"},
    )
    rpm: float = Property(doc="The number of antenna rotations per minute (RPM)")
    resolution: Angle = Property(
        default=Angle(np.radians(1)),
        doc="Resolution of the dwell_centre. Used by the :class:`~.DwellActionsGenerator` "
        "during sensor management.",
    )
    fov_angle: float = Property(
        doc="The radar horizontal field of view (FOV) angle (in radians)."
    )

    @property
    def measurement_model(self):
        rotation_offset = StateVector(
            [
                [self.orientation[0, 0]],
                [self.orientation[1, 0]],
                [self.orientation[2, 0] + self.dwell_centre[0, 0]],
            ]
        )

        return OffsetNonLinearGaussian(
            ndim_state=self.ndim_state,
            mapping=self.position_mapping,
            noise_covar=self.noise_covar,
            translation_offset=self.position,
            rotation_offset=rotation_offset,
        )

    @property
    def field_of_view_model(self, measurement_model=None):
        if measurement_model is None:
            fov_model = CartesianToBearingRange(
                ndim_state=self.measurement_model.ndim_state,
                mapping=self.measurement_model.mapping,
                noise_covar=np.diag([0, 0]),  # unused. but must define it.
                translation_offset=self.measurement_model.translation_offset,
                rotation_offset=self.measurement_model.rotation_offset,
            )
        else:
            fov_model = CartesianToBearingRange(
                ndim_state=measurement_model.ndim_state,
                mapping=measurement_model.mapping,
                noise_covar=measurement_model.noise_covar,
                translation_offset=measurement_model.translation_offset,
                rotation_offset=measurement_model.rotation_offset,
            )
        return fov_model

    def is_detectable(self, state: GroundTruthState, measurement_model=None) -> bool:
        measurement_vector = self.field_of_view_model.function(state, noise=False)

        fov_min = -self.fov_angle / 2
        fov_max = +self.fov_angle / 2

        bearing_t = measurement_vector[0, 0]
        true_range = measurement_vector[1, 0]

        return fov_min <= bearing_t <= fov_max and true_range <= self.max_range

    def is_clutter_detectable(self, state: Detection) -> bool:
        measurement_vector = state.state_vector

        fov_min = -self.fov_angle / 2
        fov_max = +self.fov_angle / 2
        bearing_t = measurement_vector[0, 0]
        true_range = measurement_vector[1, 0]

        return fov_min <= bearing_t <= fov_max and true_range <= self.max_range


class Lidar3D(Lidar):
    max_range: float = Property(
        default=np.inf, doc="The maximum detection range of the radar (in meters)"
    )
    dwell_centre: StateVector = ActionableProperty(
        doc="A `state_vector` property that describes the rotation angle of the centre of the "
        "sensor's current FOV (i.e. the dwell centre) relative to the positive x-axis of the "
        "sensor frame/orientation. The angle is positive if the rotation is in the "
        "counter-clockwise direction when viewed by an observer looking down the z-axis of "
        "the sensor frame, towards the origin. Angle units are in radians",
        generator_cls=DwellActionsGenerator,
        generator_kwargs_mapping={"rpm": "rpm", "resolution": "resolution"},
    )
    tilt_centre: StateVector = ActionableProperty(
        doc="A `state_vector` property that describes the tilting angle of the centre of the "
        "sensor's current FOV (i.e. the tilt centre) relative to the x-y plane of the sensor "
        "frame/orientation. The angle is positive if the tilt is towards the positive z "
        "direction when viewed by an observer looking down the x-axis of the sensor frame, "
        "towards the origin. Angle units are in radians",
        generator_cls=TiltActionsGenerator,
        generator_kwargs_mapping={"rpm": "rpm", "resolution": "resolution"},
    )
    rpm: float = Property(doc="The number of antenna rotations per minute (RPM)")
    resolution: Angle = Property(
        default=Angle(np.radians(1)),
        doc="Resolution of the dwell_centre. Used by the :class:`~.DwellActionsGenerator` "
        "during sensor management.",
    )
    fov_angle: float = Property(
        doc="The radar horizontal field of view (FOV) angle (in radians)."
    )
    vertical_extent: float = Property(
        doc="The radar vertical field of view (FOV) angle (in radians)."
    )

    @property
    def measurement_model(self):
        rotation_offset = StateVector(
            [
                [self.orientation[0, 0]],
                [self.orientation[1, 0] + self.tilt_centre[0, 0]],
                [self.orientation[2, 0] + self.dwell_centre[0, 0]],
            ]
        )

        return OffsetNonLinearGaussian(
            ndim_state=self.ndim_state,
            mapping=self.position_mapping,
            noise_covar=self.noise_covar,
            translation_offset=self.position,
            rotation_offset=rotation_offset,
        )

    @property
    def field_of_view_model(self, measurement_model=None):
        if measurement_model is None:
            fov_model = CartesianToElevationBearingRange(
                ndim_state=self.measurement_model.ndim_state,
                mapping=self.measurement_model.mapping,
                noise_covar=np.diag([0, 0, 0]),  # unused. but must define it.
                translation_offset=self.measurement_model.translation_offset,
                rotation_offset=self.measurement_model.rotation_offset,
            )
        else:
            fov_model = CartesianToElevationBearingRange(
                ndim_state=measurement_model.ndim_state,
                mapping=measurement_model.mapping,
                noise_covar=measurement_model.noise_covar,
                translation_offset=measurement_model.translation_offset,
                rotation_offset=measurement_model.rotation_offset,
            )
        return fov_model

    def is_detectable(self, state: GroundTruthState, measurement_model=None) -> bool:
        measurement_vector = self.field_of_view_model.function(state, noise=False)
        ver_min = -self.vertical_extent / 2
        ver_max = +self.vertical_extent / 2

        fov_min = -self.fov_angle / 2
        fov_max = +self.fov_angle / 2

        elevation_t = measurement_vector[0, 0]
        bearing_t = measurement_vector[1, 0]
        true_range = measurement_vector[2, 0]
        return (
            ver_min <= elevation_t <= ver_max
            and fov_min <= bearing_t <= fov_max
            and true_range <= self.max_range
        )

    def is_clutter_detectable(self, state: Detection) -> bool:
        measurement_vector = state.state_vector

        # Check if state falls within sensor's FOV
        ver_min = -self.vertical_extent / 2
        ver_max = +self.vertical_extent / 2

        fov_min = -self.fov_angle / 2
        fov_max = +self.fov_angle / 2

        elevation_t = measurement_vector[0, 0]
        bearing_t = measurement_vector[1, 0]
        true_range = measurement_vector[2, 0]

        return (
            ver_min <= elevation_t <= ver_max
            and fov_min <= bearing_t <= fov_max
            and true_range <= self.max_range
        )
