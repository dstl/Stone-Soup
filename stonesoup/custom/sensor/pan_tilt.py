import datetime
from copy import copy
from typing import Sequence, Iterator, Set, Union, Optional, List
from itertools import product

import numpy as np

from stonesoup.base import Property
from stonesoup.custom.sensor.action.pan_tilt import PanTiltActionsGenerator, \
    PanTiltUAVActionsGenerator
from stonesoup.functions import cart2sphere
from stonesoup.models.clutter import ClutterModel
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.sensor.action import Action, RealNumberActionGenerator
from stonesoup.sensor.actionable import ActionableProperty
from stonesoup.sensor.passive import PassiveElevationBearing
from stonesoup.sensor.sensor import Sensor
from stonesoup.types.angle import Angle, Bearing, Elevation
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.detection import TrueDetection
from stonesoup.types.groundtruth import GroundTruthState
from stonesoup.functions import build_rotation_matrix


class PanTiltCamera(PassiveElevationBearing):
    """A camera that can pan and tilt."""

    pan_tilt: StateVector = ActionableProperty(
        doc="A StateVector containing the sensor pan and tilt angles. Defaults to a zero vector",
        default=None,
        generator_cls=PanTiltActionsGenerator)
    fov_angle: float = Property(
        doc="The field of view (FOV) angle (in radians).")
    clutter_model: ClutterModel = Property(
        default=None,
        doc="An optional clutter generator that adds a set of simulated "
            ":class:`Clutter` objects to the measurements at each time step. "
            "The clutter is simulated according to the provided distribution.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def measure(self, ground_truths: Set[GroundTruthState], noise: Union[np.ndarray, bool] = True,
                **kwargs) -> Set[TrueDetection]:
        detections = set()
        measurement_model = self.measurement_model

        for truth in ground_truths:
            # Transform state to measurement space and generate
            # random noise
            measurement_vector = measurement_model.function(truth, noise=noise, **kwargs)

            # Check if state falls within sensor's FOV
            fov_min = -self.fov_angle / 2
            fov_max = +self.fov_angle / 2
            bearing_t = measurement_vector[1, 0]
            elevation_t = measurement_vector[0, 0]

            # Do not measure if state not in FOV
            if (not fov_min <= bearing_t <= fov_max) or (not fov_min <= elevation_t <= fov_max):
                continue

            detection = TrueDetection(measurement_vector,
                                      measurement_model=measurement_model,
                                      timestamp=truth.timestamp,
                                      groundtruth_path=truth)
            detections.add(detection)

        # Generate clutter at this time step
        if self.clutter_model is not None:
            self.clutter_model.measurement_model = measurement_model
            clutter = self.clutter_model.function(ground_truths)
            detections |= clutter

        return detections


class PanTiltUAVCamera(Sensor):
    """A camera that can pan and tilt."""
    ndim_state: int = Property(
        doc="Number of state dimensions. This is utilised by (and follows in\
                format) the underlying :class:`~.CartesianToElevationBearing`\
                model")
    mapping: np.ndarray = Property(
        doc="Mapping between the targets state space and the sensors\
                measurement capability")
    noise_covar: CovarianceMatrix = Property(
        doc="The sensor noise covariance matrix. This is utilised by\
                (and follow in format) the underlying \
                :class:`~.CartesianToElevationBearing` model")
    fov_angle: Union[float, List[float]] = Property(
        doc="The field of view (FOV) angle (in radians).")
    pan_tilt: StateVector = ActionableProperty(
        doc="A StateVector containing the sensor pan and tilt angles. Defaults to a zero vector",
        default=None,
        generator_cls=PanTiltUAVActionsGenerator)
    clutter_model: ClutterModel = Property(
        default=None,
        doc="An optional clutter generator that adds a set of simulated "
            ":class:`Clutter` objects to the measurements at each time step. "
            "The clutter is simulated according to the provided distribution.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.fov_angle, float):
            self.fov_angle = [self.fov_angle, self.fov_angle]
        if self.pan_tilt is None:
            self.pan_tilt = StateVector([Angle(0), Angle(0)])

    @property
    def measurement_model(self):
        return LinearGaussian(
            ndim_state=self.ndim_state,
            mapping=self.mapping,
            noise_covar=self.noise_covar)

    def measure(self, ground_truths: Set[GroundTruthState], noise: Union[np.ndarray, bool] = True,
                **kwargs) -> Set[TrueDetection]:

        detections = set()
        measurement_model = self.measurement_model

        for truth in ground_truths:
            # Transform state to measurement space and generate random noise
            measurement_vector = measurement_model.function(truth, noise=noise, **kwargs)

            # Normalise measurement vector relative to sensor position
            norm_measurement_vector = measurement_vector.astype(float) - self.position.astype(float)

            # Rotate measurement vector relative to sensor orientation
            rotation_matrix = build_rotation_matrix(self.orientation)
            norm_rotated_measurement_vector = rotation_matrix @ norm_measurement_vector

            # Convert to spherical coordinates
            _, bearing_t, elevation_t = cart2sphere(*norm_rotated_measurement_vector)

            # Check if state falls within sensor's FOV
            fov_min = -np.array(self.fov_angle) / 2
            fov_max = +np.array(self.fov_angle) / 2

            # Do not measure if state not in FOV
            if (not fov_min[0] <= bearing_t <= fov_max[0]) \
                    or (not fov_min[1] <= elevation_t <= fov_max[1]):
                continue

            detection = TrueDetection(measurement_vector,
                                      measurement_model=measurement_model,
                                      timestamp=truth.timestamp,
                                      groundtruth_path=truth)
            detections.add(detection)

        # Generate clutter at this time step
        if self.clutter_model is not None:
            self.clutter_model.measurement_model = measurement_model
            clutter = self.clutter_model.function(ground_truths)
            detections |= clutter

        return detections

    @property
    def orientation(self) -> Optional[StateVector]:
        """A 3x1 StateVector of angles (rad), specifying the sensor orientation in terms of the
        counter-clockwise rotation around each Cartesian axis in the order :math:`x,y,z`.
        The rotation angles are positive if the rotation is in the counter-clockwise
        direction when viewed by an observer looking along the respective rotation axis,
        towards the origin.

        .. note::
            This property delegates the actual calculation of orientation to the Sensor's
            :attr:`movement_controller`

            It is settable if, and only if, the sensor holds its own internal movement_controller.
            """
        if self.movement_controller is None:
            return None
        return self.movement_controller.orientation + self.rotation_offset \
               + StateVector([0, self.pan_tilt[1], self.pan_tilt[0]])