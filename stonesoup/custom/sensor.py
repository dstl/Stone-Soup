import datetime
from copy import copy
from typing import Sequence, Iterator, Set, Union
from itertools import product

import numpy as np

from stonesoup.base import Property
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


class ChangePanTiltAction(Action):
    """The action of changing the dwell centre of sensors where `dwell_centre` is an
    :class:`~.ActionableProperty`"""

    rotation_end_time: datetime.datetime = Property(readonly=True,
                                                    doc="End time of rotation.")
    increasing_angle: Sequence[bool] = Property(
        default=None, readonly=True,
        doc="Indicated the direction of change in the dwell centre angle. The first element "
            "relates to bearing, the second to elevation.")

    def act(self, current_time, timestamp, init_value):
        """Assumes that duration keeps within the action end time

        Parameters
        ----------
        current_time: datetime.datetime
            Current time
        timestamp: datetime.datetime
            Modification of attribute ends at this time stamp
        init_value: Any
            Current value of the dwell centre

        Returns
        -------
        Any
            The new value of the dwell centre"""

        if timestamp >= self.end_time:
            return self.target_value  # target direction
        else:
            return init_value  # same direction


class PanTiltActionsGenerator(RealNumberActionGenerator):
    """Generates possible actions for changing the dwell centre of a sensor in a given
    time period."""

    owner: object = Property(doc="Object with `timestamp`, `rpm` (revolutions per minute) and "
                                 "dwell-centre attributes")
    resolution: Angle = Property(default=np.radians(1), doc="Resolution of action space")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = Angle(np.radians(1e-6))

    @property
    def default_action(self):
        return ChangePanTiltAction(rotation_end_time=self.end_time,
                                   generator=self,
                                   end_time=self.end_time,
                                   target_value=self.current_value,
                                   increasing_angle=None)

    def __call__(self, resolution=None, epsilon=None):
        """
        Parameters
        ----------
        resolution : Angle
            Resolution of yielded action target values
        epsilon: float
            Tolerance of equality check in iteration
        """
        if resolution is not None:
            self.resolution = resolution
        if epsilon is not None:
            self.epsilon = epsilon

    @property
    def initial_value(self):
        return self.current_value

    @property
    def duration(self):
        return self.end_time - self.start_time

    @property
    def rps(self):
        return self.owner.rpm / 60

    @property
    def angle_delta(self):
        return StateVector([Angle(0),
                            Angle(self.duration.total_seconds() * self.rps[0] * 2 * np.pi),
                            Angle(self.duration.total_seconds() * self.rps[1] * 2 * np.pi)])

    @property
    def min(self):
        min = self.initial_value.astype(float) - self.angle_delta
        min[1] = np.maximum(Angle(self.initial_value[1]) - self.angle_delta[1],
                            Angle(-np.pi / 2))
        return min

    @property
    def max(self):
        max = self.initial_value.astype(float) + self.angle_delta
        max[1] = np.minimum(Angle(self.initial_value[1]) + self.angle_delta[1],
                            Angle(np.pi / 2))
        return max

    def __contains__(self, item):

        if self.angle_delta[2] >= np.pi or self.angle_delta[1] >= np.pi / 2:
            # Left turn and right turn are > 180, so all angles hit
            return True

        if isinstance(item, ChangePanTiltAction):
            item = item.target_value

        return self.min[1] <= item[1] <= self.max[1] and self.min[2] <= item[2] <= self.max[2]

    def _end_time_direction_pan(self, angle):
        """Given a target bearing, should the dwell centre rotate so as to increase its angle
        value, or decrease? And how long until it reaches the target."""

        angle = Angle(angle)

        if self.initial_value[2] - self.epsilon \
                <= angle \
                <= self.initial_value[2] + self.epsilon:
            return self.start_time, None  # no rotation, target bearing achieved

        angle_delta = np.abs(angle - self.initial_value[2])

        return (
            self.start_time + datetime.timedelta(seconds=angle_delta / (self.rps[1] * 2 * np.pi)),
            angle > self.initial_value[2]
        )

    def _end_time_direction_tilt(self, angle):
        """Given a target bearing, should the dwell centre rotate so as to increase its angle
        value, or decrease? And how long until it reaches the target."""

        angle = Angle(angle)

        if self.initial_value[1] - self.epsilon \
                <= angle \
                <= self.initial_value[1] + self.epsilon:
            return self.start_time, None  # no rotation, target bearing achieved

        angle_delta = np.abs(angle - self.initial_value[1])

        return (
            self.start_time + datetime.timedelta(seconds=angle_delta / (self.rps[1] * 2 * np.pi)),
            angle > self.initial_value[1]
        )

    def __iter__(self) -> Iterator[ChangePanTiltAction]:
        """Returns ChangePanTiltAction types, where the value is a possible value of the [0, 0]
        element of the dwell centre's state vector."""

        possible_elevations = np.arange(self.min[1], self.max[1], self.resolution)
        for elevation in possible_elevations:
            elevation_end_time, increasing_e = self._end_time_direction_tilt(elevation)
            bearing = self.min[2]
            while bearing <= self.max[2] + self.epsilon:
                bearing_end_time, increasing_b = self._end_time_direction_pan(bearing)
                yield ChangePanTiltAction(rotation_end_time=max(bearing_end_time,
                                                                elevation_end_time),
                                          generator=self,
                                          end_time=self.end_time,
                                          target_value=StateVector([Angle(0),
                                                                    Elevation(elevation),
                                                                    Bearing(bearing)]),
                                          increasing_angle=[increasing_e, increasing_b])
                bearing += self.resolution

    def action_from_value(self, value):
        raise NotImplementedError


class PanTiltUAVActionsGenerator(PanTiltActionsGenerator):

    @property
    def min(self):
        min = self.initial_value.astype(float) - self.angle_delta
        min[0] = np.maximum(Angle(self.initial_value[0]) - self.angle_delta[0],
                            Angle(-np.pi / 2))
        min[1] = np.maximum(Angle(self.initial_value[1]) - self.angle_delta[1],
                            Angle(-np.pi / 2))
        return min

    @property
    def max(self):
        max = self.initial_value.astype(float) + self.angle_delta
        max[0] = np.minimum(Angle(self.initial_value[0]) + self.angle_delta[0],
                            Angle(np.pi / 2))
        max[1] = np.minimum(Angle(self.initial_value[1]) + self.angle_delta[1],
                            Angle(np.pi / 2))
        return max

    def __contains__(self, item):

        if self.angle_delta[2] >= np.pi / 2 or self.angle_delta[1] >= np.pi / 2:
            # Left turn and right turn are > 180, so all angles hit
            return True

        if isinstance(item, ChangePanTiltAction):
            item = item.target_value

        return self.min[1] <= item[1] <= self.max[1] and self.min[2] <= item[2] <= self.max[2]

    def _end_time_direction_pan(self, angle):
        """Given a target bearing, should the dwell centre rotate so as to increase its angle
        value, or decrease? And how long until it reaches the target."""

        angle = Angle(angle)

        if self.initial_value[2] - self.epsilon \
                <= angle \
                <= self.initial_value[2] + self.epsilon:
            return self.start_time, None  # no rotation, target bearing achieved

        angle_delta = np.abs(angle - self.initial_value[2])

        return (
            self.start_time + datetime.timedelta(seconds=angle_delta / (self.rps[1] * 2 * np.pi)),
            angle > self.initial_value[2]
        )

    def _end_time_direction_tilt(self, angle):
        """Given a target bearing, should the dwell centre rotate so as to increase its angle
        value, or decrease? And how long until it reaches the target."""

        angle = Angle(angle)

        if self.initial_value[1] - self.epsilon \
                <= angle \
                <= self.initial_value[1] + self.epsilon:
            return self.start_time, None  # no rotation, target bearing achieved

        angle_delta = np.abs(angle - self.initial_value[1])

        return (
            self.start_time + datetime.timedelta(seconds=angle_delta / (self.rps[1] * 2 * np.pi)),
            angle > self.initial_value[1]
        )

    def __iter__(self) -> Iterator[ChangePanTiltAction]:
        """Returns ChangePanTiltAction types, where the value is a possible value of the [0, 0]
        element of the dwell centre's state vector."""

        possible_tilt_angles = np.arange(self.min[1], self.max[1], self.resolution)
        possible_pan_angles = np.arange(self.min[2], self.max[2], self.resolution)
        for (pan_angle, tilt_angle) in product(possible_pan_angles, possible_tilt_angles):
            pan_end_time, increasing_p = self._end_time_direction_pan(pan_angle)
            tilt_end_time, increasing_t = self._end_time_direction_tilt(tilt_angle)
            yield ChangePanTiltAction(rotation_end_time=max(pan_end_time, tilt_end_time),
                                      generator=self,
                                      end_time=self.end_time,
                                      target_value=StateVector([Angle(0),
                                                                Elevation(tilt_angle),
                                                                Bearing(pan_angle)]),
                                      increasing_angle=[increasing_t, increasing_p])

    def action_from_value(self, value):
        raise NotImplementedError


class PanTiltCamera(PassiveElevationBearing):
    """A camera that can pan and tilt."""

    rotation_offset: StateVector = ActionableProperty(
        doc="A StateVector containing the sensor rotation "
            "offsets from the platform's primary axis (defined as the "
            "direction of motion). Defaults to a zero vector with the "
            "same length as the Platform's :attr:`velocity_mapping`",
        default=None,
        generator_cls=PanTiltActionsGenerator)
    rpm: float = Property(
        doc="The number of rotations per minute (RPM)")
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
    pan_tilt: StateVector = ActionableProperty(
        doc="A StateVector containing the sensor rotation "
            "offsets from the platform's primary axis (defined as the "
            "direction of motion). Defaults to a zero vector with the "
            "same length as the Platform's :attr:`velocity_mapping`",
        default=None,
        generator_cls=PanTiltUAVActionsGenerator)
    rpm: float = Property(
        doc="The number of rotations per minute (RPM)")
    fov_angle: float = Property(
        doc="The field of view (FOV) angle (in radians).")
    clutter_model: ClutterModel = Property(
        default=None,
        doc="An optional clutter generator that adds a set of simulated "
            ":class:`Clutter` objects to the measurements at each time step. "
            "The clutter is simulated according to the provided distribution.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            # Transform state to measurement space and generate
            # random noise
            measurement_vector = measurement_model.function(truth, noise=noise, **kwargs)

            # Normalise measurement vector relative to sensor position
            norm_measurement_vector = measurement_vector.astype(float) - self.position.astype(float)

            # Rotate measurement vector relative to sensor orientation
            rotation_matrix = build_rotation_matrix(self.orientation)
            norm_rotated_measurement_vector = rotation_matrix @ norm_measurement_vector

            # Convert to spherical coordinates
            _, bearing_t, elevation_t = cart2sphere(*norm_rotated_measurement_vector)

            # Check if state falls within sensor's FOV
            fov_min = -self.fov_angle / 2
            fov_max = +self.fov_angle / 2
            bearing_t = bearing_t
            elevation_t = elevation_t

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