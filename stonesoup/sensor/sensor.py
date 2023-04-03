from abc import abstractmethod, ABC
from typing import Set, Union, Sequence

import numpy as np

from .actionable import Actionable
from .base import PlatformMountable
from ..base import Property
from ..models.clutter.clutter import ClutterModel
from ..types.detection import TrueDetection, Detection
from ..types.groundtruth import GroundTruthState


class Sensor(PlatformMountable, Actionable):
    """Sensor Base class for general use.

    Most properties and methods are inherited from :class:`~.PlatformMountable`.

    Notes
    -----
    * Sensors must have a measure function.
    * Attributes that are modifiable via actioning the sensor should be
      :class:`~.ActionableProperty` types.
    * The sensor has a timestamp property that should be updated via its
      :meth:`~Actionable.act` method.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.timestamp = None

    def validate_timestamp(self):

        if self.timestamp:
            return True

        try:
            self.timestamp = self.movement_controller.state.timestamp
        except AttributeError:
            return False
        if self.timestamp is None:
            return False
        return True

    @abstractmethod
    def measure(self, ground_truths: Set[GroundTruthState], noise: Union[np.ndarray, bool] = True,
                **kwargs) -> Set[TrueDetection]:
        """Generate a measurement for a given state

        Parameters
        ----------
        ground_truths : Set[:class:`~.GroundTruthState`]
            A set of :class:`~.GroundTruthState`
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is `True`, in which
            case :meth:`~.Model.rvs` is used; if `False`, no noise will be added)

        Returns
        -------
        Set[:class:`~.TrueDetection`]
            A set of measurements generated from the given states. The timestamps of the
            measurements are set equal to that of the corresponding states that they were
            calculated from. Each measurement stores the ground truth path that it was produced
            from.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def measurement_model(self):
        """Measurement model of the sensor, describing general sensor model properties"""
        raise NotImplementedError


class SimpleSensor(Sensor, ABC):

    clutter_model: ClutterModel = Property(
        default=None,
        doc="An optional clutter generator that adds a set of simulated "
            ":class:`Clutter` objects to the measurements at each time step. "
            "The clutter is simulated according to the provided distribution.")

    def measure(self, ground_truths: Set[GroundTruthState], noise: Union[np.ndarray, bool] = True,
                **kwargs) -> Set[TrueDetection]:

        measurement_model = self.measurement_model

        detectable_ground_truths = [truth for truth in ground_truths
                                    if self.is_detectable(truth)]

        if noise is True:
            if len(detectable_ground_truths) > 1:
                noise_vectors_iter = iter(measurement_model.rvs(len(detectable_ground_truths),
                                                                **kwargs))
            else:
                noise_vectors_iter = iter([measurement_model.rvs(**kwargs)])

        detections = set()
        for truth in detectable_ground_truths:
            measurement_vector = measurement_model.function(truth, noise=False, **kwargs)

            if noise is True:
                measurement_noise = next(noise_vectors_iter)
            else:
                measurement_noise = noise

            # Add in measurement noise to the measurement vector
            measurement_vector += measurement_noise

            detection = TrueDetection(measurement_vector,
                                      measurement_model=measurement_model,
                                      timestamp=truth.timestamp,
                                      groundtruth_path=truth)
            detections.add(detection)

        # Generate clutter at this time step
        if self.clutter_model is not None:
            self.clutter_model.measurement_model = measurement_model
            clutter = self.clutter_model.function(ground_truths)
            detectable_clutter = [cltr for cltr in clutter
                                  if self.is_detectable(cltr)]
            detections = set.union(detections, detectable_clutter)

        return detections

    @abstractmethod
    def is_detectable(self, state: Union[GroundTruthState, Detection]) -> bool:
        raise NotImplementedError


class SensorSuite(Sensor):
    """Sensor composition type

    Models a suite of sensors all returning detections at the same 'time'. Returns all detections
    in one go.
    Can append information of the sensors to the metadata of their corresponding detections.
    """

    sensors: Sequence[Sensor] = Property(doc="Suite of sensors to get detections from.")

    attributes_inform: Set[str] = Property(
        doc="Names of attributes to store the value of at time of detection."
    )

    def measure(self, ground_truths: Set[GroundTruthState], noise: Union[bool, np.ndarray] = True,
                **kwargs) -> Set[TrueDetection]:
        """Call each sub-sensor's measure method in turn. Key word arguments are passed to the
        measure method of each sensor.

        Append additional metadata to each sensor's set of detections. Which keys are appended is
        dictated by :attr:`attributes_inform`."""

        all_detections = set()

        for sensor in self.sensors:

            detections = sensor.measure(ground_truths, noise, **kwargs)

            attributes_dict = {attribute_name: sensor.__getattribute__(attribute_name)
                               for attribute_name in self.attributes_inform}

            for detection in detections:
                detection.metadata.update(attributes_dict)

            all_detections.update(detections)

        return all_detections

    @property
    def measurement_model(self):
        """Measurement model of the sensor, describing general sensor model properties"""
        raise NotImplementedError
