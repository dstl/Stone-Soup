# -*- coding: utf-8 -*-

from typing import Sequence

from scipy.stats import multinomial

from ..base import Property
from ..models.measurement.categorical import MarkovianMeasurementModel
from ..sensor.sensor import Sensor
from ..types.array import StateVector
from ..types.detection import TrueCategoricalDetection


class HMMSensor(Sensor):
    r"""Sensor model that observes a categorical state space and returns categorical measurements.

    Measurements are categorical distributions over a finite set of categories
    :math:`Z = \{\zeta^n|n\in \mathbf{N}, n\le N\} (for some finite :math:`N`).
    """

    measurement_model: MarkovianMeasurementModel = Property(
        doc="Measurement model to generate detection vectors from"
    )
    measurement_categories: Sequence[str] = Property(doc="Sequence of measurement category names. "
                                                         "Defaults to a list of integers",
                                                     default=None)

    @property
    def ndim_state(self):
        return self.measurement_model.ndim_state

    @property
    def ndim_meas(self):
        return self.measurement_model.ndim_meas

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.measurement_categories is None:
            self.measurement_categories = list(map(str, range(self.ndim_meas)))

        if len(self.measurement_categories) != self.ndim_meas:
            raise ValueError(
                f"ndim_meas of {self.ndim_meas} does not match number of measurement categories "
                f"{len(self.measurement_categories)}"
            )

    def measure(self, ground_truths, noise: bool = True, **kwargs):
        r"""Generate a categorical measurement for a given set of true categorical state.

        Parameters
        ----------
        ground_truths: Set[:class:`~.CategoricalGroundTruthState`]
            A set of :class:`~.CategoricalGroundTruthState`.
        noise: bool
            Indicates whether measurement vectors are sampled from and the resultant measurement
            categories returned instead. These are discrete categories instead of a distribution
            over the measurement space. They are represented by N-tuples, with all components
            equal to 0, except at an index corresponding to the relevant category.
            For example :math:`e^k` indicates that the measurement category is :math:`\zeta^k`.
            If `False`, the resultant distribution is returned.

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
            timestamp = truth.timestamp
            detection_vector = self.measurement_model.function(truth, noise=noise, **kwargs)

            if noise:
                # Sample from resultant distribution
                rv = multinomial(n=1, p=detection_vector.flatten())
                detection_vector = StateVector(rv.rvs(size=1, random_state=None))

            detection = TrueCategoricalDetection(state_vector=detection_vector,
                                                 timestamp=timestamp,
                                                 categories=self.measurement_categories,
                                                 measurement_model=self.measurement_model,
                                                 groundtruth_path=truth)
            detections.add(detection)

        return detections
