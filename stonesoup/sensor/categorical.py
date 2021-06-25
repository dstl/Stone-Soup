# -*- coding: utf-8 -*-
from typing import Set, Sequence

from ..base import Property
from ..models.measurement.categorical import CategoricalMeasurementModel
from ..sensor.sensor import Sensor
from ..types.detection import TrueDetection, TrueCategoricalDetection
from ..types.groundtruth import GroundTruthState, GroundTruthPath
from ..types.state import CategoricalState


class CategoricalSensor(Sensor):
    measurement_model: CategoricalMeasurementModel = Property(
        doc="Categorical measurement model used in generating measurements.")
    category_names: Sequence[str] = Property(default=None,
                                             doc="Measurement category names.")

    @property
    def ndim_state(self):
        return self.measurement_model.ndim_state

    @property
    def ndim_meas(self):
        return self.measurement_model.ndim_meas

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.category_names and len(self.category_names) != self.ndim_meas:
            raise ValueError(f"{len(self.category_names)} category names were given for a sensor "
                             f"which returns vectors of length {self.ndim_meas}")

    def measure(self, ground_truths: Set[GroundTruthState], **kwargs) -> Set[TrueDetection]:
        """Generate a categorical measurement for a given categorical state.

        Parameters
        ----------
        ground_truths : Set[:class:`~.CategoricalGroundTruthState`]
            A set of :class:`~.CategoricalGroundTruthState`

        Returns
        -------
        Set[:class:`~.TrueCategoricalDetection`]
            A set of measurements generated from the given states. The timestamps of the
            measurements are set equal to that of the corresponding states that they were
            calculated from. Each measurement stores the ground truth path that it was produced
            from.
        """

        detections = set()
        for truth in ground_truths:

            wrong_type = False
            if isinstance(truth, GroundTruthPath):
                if not isinstance(truth[-1], CategoricalState):
                    wrong_type = True
            elif not isinstance(truth, CategoricalState):
                wrong_type = True

            if wrong_type:
                raise ValueError("Categorical sensor can only observe categorical states")

            measurement_vector = self.measurement_model.function(truth, **kwargs)
            detection = TrueCategoricalDetection(state_vector=measurement_vector,
                                                 timestamp=truth.timestamp,
                                                 measurement_model=self.measurement_model,
                                                 groundtruth_path=truth,
                                                 category_names=self.category_names)
            detections.add(detection)

        return detections
