import datetime
from typing import Union, List, Set

import numpy as np
import geopy.distance
from shapely import Point

from stonesoup.base import Property
from stonesoup.custom.functions import geodesic_point_buffer
from stonesoup.custom.sensor.action.location import LocationActionGenerator
from stonesoup.models.clutter import ClutterModel
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.sensor.action import ActionGenerator
from stonesoup.sensor.actionable import ActionableProperty
from stonesoup.sensor.sensor import Sensor
from stonesoup.types.array import CovarianceMatrix
from stonesoup.types.detection import TrueDetection
from stonesoup.types.groundtruth import GroundTruthState
from stonesoup.types.numeric import Probability


class MovableUAVCamera(Sensor):
    """A movable UAV camera sensor."""

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
    prob_detect: Probability = Property(
        default=None,
        doc="The probability of detection of the sensor. Defaults to 1.0")
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
    fov_in_km: bool = Property(
        doc="Whether the FOV radius is in kilo-meters or degrees",
        default=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.prob_detect is None:
            self.prob_detect = Probability(1)
        self._footprint = None

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

    @property
    def footprint(self):
        if self._footprint is None:
            if self.fov_in_km:
                self._footprint = geodesic_point_buffer(*np.flip(self.position[0:2]),
                                                        self.fov_radius)
            else:
                self._footprint = Point(self.position[0:2]).buffer(self.fov_radius)
        return self._footprint

    def act(self, timestamp: datetime.datetime):
        super().act(timestamp)
        if self.fov_in_km:
            self._footprint = geodesic_point_buffer(*np.flip(self.position[0:2]), self.fov_radius)
        else:
            self._footprint = Point(self.position[0:2]).buffer(self.fov_radius)

    def measure(self, ground_truths: Set[GroundTruthState], noise: Union[np.ndarray, bool] = True,
                **kwargs) -> Set[TrueDetection]:

        detections = set()
        measurement_model = self.measurement_model

        for truth in ground_truths:
            # Transform state to measurement space and generate random noise
            measurement_vector = measurement_model.function(truth, noise=noise, **kwargs)

            if self.fov_in_km:
                # distance = geopy.distance.distance(np.flip(self.position[0:2]),
                #                                    np.flip(measurement_vector[0:2])).km
                if not self.footprint.contains(Point(measurement_vector[0:2])):
                    continue
            else:
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

            # Generate detection with probability of detection
            if np.random.rand() <= self.prob_detect:
                detections.add(detection)

        # Generate clutter at this time step
        if self.clutter_model is not None:
            self.clutter_model.measurement_model = measurement_model
            clutter = self.clutter_model.function(ground_truths, **kwargs)
            detections |= clutter

        return detections

    def _default_action(self, name, property_, timestamp):
        """Returns the default action of the action generator associated with the property
        (assumes the property is an :class:`~.ActionableProperty`)."""
        generator = self._get_generator(name, property_, timestamp, self.timestamp)
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

        generators = {self._get_generator(name, property_, timestamp, start_timestamp)
                      for name, property_ in self._actionable_properties.items()}

        return generators

    def _get_generator(self, name, prop, timestamp, start_timestamp):
        """Returns the action generator associated with the """
        kwargs = {'owner': self, 'attribute': name, 'start_time': start_timestamp,
                  'end_time': timestamp}
        if self.resolutions and name in self.resolutions.keys():
            kwargs['resolution'] = self.resolutions[name]
        if self.limits and name in self.limits.keys():
            kwargs['limits'] = self.limits[name]
        generator = prop.generator_cls(**kwargs)
        return generator
