# -*- coding: utf-8 -*-

import numpy as np

from ..base import Property
from ..types import Hypothesis
from .numeric import Probability
from .prediction import MeasurementPrediction, Prediction
from .detection import Detection, MissedDetection


class MultipleMeasurementHypothesis(Hypothesis):
    """Multiple Measurement Hypothesis base type

    """

    prediction = Property(
        Prediction,
        doc="Predicted track state")
    measurement_prediction = Property(
        MeasurementPrediction,
        default=None,
        doc="Optional track prediction in measurement space")
    weighted_measurements = Property(
        list,
        default=list(),
        doc="Weighted measurements used for hypothesis and updating")
    selected_measurement = Property(
        Detection,
        default=None,
        doc="The measurement that was selected to associate with a track.")

    def add_weighted_detections(self, measurements, weights, normalize=False):

        # verify that 'measurements' and 'weights' are the same size and the
        # correct data types
        if any(not (isinstance(measurement, Detection))
               for measurement in measurements):
            raise Exception('measurements must all be of type Detection!')
        if any(not (isinstance(weight, float) or isinstance(weight, int))
               for weight in weights):
            raise Exception('weights must all be of type float or int!')
        if len(measurements) != len(weights):
            raise Exception('There must be the same number of weights '
                            'and measurements!')

        # normalize the weights to sum up to 1 if indicated
        if normalize is True:
            sum_weights = sum(weights)
            for index in range(0, len(weights)):
                weights[index] /= sum_weights

        # store weights and measurements in 'weighted_measurements'
        for index in range(0, len(measurements)):
            self.weighted_measurements.append(
                {"measurement": measurements[index],
                 "weight": weights[index]})

    def __bool__(self):
        if (self.selected_measurement is not None):
            return not isinstance(self.selected_measurement, MissedDetection)
        else:
            raise Exception('Cannot check whether a '
                            'MultipleMeasurementHypothesis.'
                            'selected_measurement is a MissedDetection before'
                            ' it has been set!')

    def set_selected_measurement(self, detection):
        if any(np.array_equal(detection.state_vector,
                              measurement["measurement"].state_vector)
               for measurement in self.weighted_measurements):
            self.selected_measurement = detection
        else:
            raise Exception('Cannot set MultipleMeasurementHypothesis.'
                            'selected_measurement to a value not contained in'
                            ' MultipleMeasurementHypothesis.'
                            'weighted_detections!')

    def get_selected_measurement(self):
        if self.selected_measurement is not None:
            return self.selected_measurement
        else:
            raise Exception('best measurement in MultipleMeasurementhypothesis'
                            ' not selected, so it cannot be returned!')

    @property
    def measurement(self):
        return self.get_selected_measurement()


class ProbabilityMultipleMeasurementHypothesis(MultipleMeasurementHypothesis):
    """Probability-scored multiple measurement hypothesis.

    """

    def add_weighted_detections(self, measurements, weights, normalize=False):
        self.weighted_measurements = list()

        # verify that 'measurements' and 'weights' are the same size and the
        # correct data types
        if any(not (isinstance(measurement, Detection))
               for measurement in measurements):
            raise Exception('measurements must all be of type Detection!')
        if any(not isinstance(weight, Probability) for weight in weights):
            raise Exception('weights must all be of type Probability!')
        if len(measurements) != len(weights):
            raise Exception('There must be the same number of weights '
                            'and measurements!')

        # normalize the weights to sum up to 1 if indicated
        if normalize is True:
            sum_weights = Probability.sum(weights)
            for index in range(0, len(weights)):
                weights[index] /= sum_weights

        # store probabilities and measurements in 'weighted_measurements'
        for index in range(0, len(measurements)):
            self.weighted_measurements.append(
                {"measurement": measurements[index],
                 "weight": weights[index]})
