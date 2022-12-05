import datetime
from typing import Union, List, Set

import numpy as np

from stonesoup.base import Property
from stonesoup.custom.sensor.action.location import LocationActionGenerator
from stonesoup.models.clutter import ClutterModel
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.sensor.action import ActionGenerator
from stonesoup.sensor.actionable import ActionableProperty
from stonesoup.sensor.sensor import Sensor
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.types.detection import TrueDetection
from stonesoup.types.groundtruth import GroundTruthState


class MovableUAVCamera(Sensor):
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
    fov_radius: Union[float, List[float]] = Property(
        doc="The detection field of view radius of the sensor")
    clutter_model: ClutterModel = Property(
        default=None,
        doc="An optional clutter generator that adds a set of simulated "
            ":class:`Clutter` objects to the measurements at each time step. "
            "The clutter is simulated according to the provided distribution.")
    location_x: float = ActionableProperty(
        doc="The sensor x location. Defaults to zero",
        default=0,
        generator_cls=LocationActionGenerator
    )
    location_y: float = ActionableProperty(
        doc="The sensor y location. Defaults to zero",
        default=0,
        generator_cls=LocationActionGenerator
    )
    limits: dict = Property(
        doc="The sensor min max location",
        default=None
    )

    @location_x.setter
    def location_x(self, value):
        self._property_location_x = value
        if not self.movement_controller:
            return
        new_position = self.movement_controller.position.copy()
        new_position[0] = value
        self.movement_controller.position = new_position

    @location_y.setter
    def location_y(self, value):
        self._property_location_y = value
        if not self.movement_controller:
            return
        new_position = self.movement_controller.position.copy()
        new_position[1] = value
        self.movement_controller.position = new_position

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
            norm_measurement_vector = measurement_vector.astype(float) - self.position.astype(
                float)

            distance = np.linalg.norm(norm_measurement_vector[0:2])

            # Do not measure if state not in FOV
            if distance > self.fov_radius:
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

    def _default_action(self, name, property_, timestamp):
        """Returns the default action of the action generator associated with the property
        (assumes the property is an :class:`~.ActionableProperty`."""

        if self.resolutions and name in self.resolutions.keys():
            if self.limits and name in self.limits.keys():
                generator = property_.generator_cls(owner=self,
                                                    attribute=name,
                                                    start_time=self.timestamp,
                                                    end_time=timestamp,
                                                    resolution=self.resolutions[name],
                                                    limits=self.limits[name])
            else:
                generator = property_.generator_cls(owner=self,
                                                    attribute=name,
                                                    start_time=self.timestamp,
                                                    end_time=timestamp,
                                                    resolution=self.resolutions[name])
        else:
            if self.limits and name in self.limits.keys():
                generator = property_.generator_cls(owner=self,
                                                    attribute=name,
                                                    start_time=self.timestamp,
                                                    end_time=timestamp,
                                                    limits=self.limits)
            else:
                generator = property_.generator_cls(owner=self,
                                                    attribute=name,
                                                    start_time=self.timestamp,
                                                    end_time=timestamp)
        return generator.default_action

    def actions(self, timestamp: datetime.datetime, start_timestamp: datetime.datetime = None
                ) -> Set[ActionGenerator]:
        """Method to return a set of action generators available up to a provided timestamp.

        A generator is returned for each actionable property that the sensor has.

        Parameters
        ----------
        timestamp: datetime.datetime
            Time of action finish.
        start_timestamp: datetime.datetime, optional
            Time of action start.

        Returns
        -------
        : set of :class:`~.ActionGenerator`
            Set of action generators, that describe the bounds of each action space.
        """

        if not self.validate_timestamp():
            self.timestamp = timestamp

        if start_timestamp is None:
            start_timestamp = self.timestamp

        generators = set()
        for name, property_ in self._actionable_properties.items():
            if self.resolutions and name in self.resolutions.keys():
                if self.limits and name in self.limits.keys():
                    generators.add(property_.generator_cls(owner=self,
                                                           attribute=name,
                                                           start_time=start_timestamp,
                                                           end_time=timestamp,
                                                           resolution=self.resolutions[name],
                                                           limits=self.limits[name]))
                else:
                    generators.add(property_.generator_cls(owner=self,
                                                           attribute=name,
                                                           start_time=start_timestamp,
                                                           end_time=timestamp,
                                                           resolution=self.resolutions[name]))
            else:
                if self.limits and name in self.limits.keys():
                    generators.add(property_.generator_cls(owner=self,
                                                           attribute=name,
                                                           start_time=start_timestamp,
                                                           end_time=timestamp,
                                                           limits=self.limits[name]))
                else:
                    generators.add(property_.generator_cls(owner=self,
                                                           attribute=name,
                                                           start_time=start_timestamp,
                                                           end_time=timestamp))
        return generators