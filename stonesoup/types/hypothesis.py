# -*- coding: utf-8 -*-

from abc import abstractmethod
from collections import UserDict

from .base import Type
from ..base import Property
from .detection import Detection
from .prediction import MeasurementPrediction, Prediction
from ..types.numeric import Probability


class Hypothesis(Type):
    """Hypothesis base type

    """

    def __lt__(self, other):
        return NotImplemented

    def __le__(self, other):
        return NotImplemented

    def __eq__(self, other):
        return NotImplemented

    def __gt__(self, other):
        return NotImplemented

    def __ge__(self, other):
        return NotImplemented


class SingleMeasurementHypothesis(Hypothesis):
    """A hypothesis based on a single measurement.

    """
    prediction = Property(
        Prediction,
        doc="Predicted track state")
    measurement = Property(
        Detection,
        doc="Detection used for hypothesis and updating")
    measurement_prediction = Property(
        MeasurementPrediction,
        default=None,
        doc="Optional track prediction in measurement space")

    def __bool__(self):
        return self.measurement is not None


class SingleMeasurementDistanceHypothesis(SingleMeasurementHypothesis):
    """Distance scored hypothesis subclass.

    Notes
    -----
    As smaller distance is 'better', comparison logic is reversed
    i.e. smaller distance is a greater likelihood.
    """

    distance = Property(
        float,
        doc="Distance between detection and prediction")

    def __lt__(self, other):
        return self.distance > other.distance

    def __le__(self, other):
        return self.distance >= other.distance

    def __eq__(self, other):
        return self.distance == other.distance

    def __gt__(self, other):
        return self.distance < other.distance

    def __ge__(self, other):
        return self.distance <= other.distance


class SingleMeasurementProbabilityHypothesis(Hypothesis):
    """Single Measurement Probability scored hypothesis subclass.

    """

    probability = Property(
        Probability,
        doc="Probability that detection is true location of prediction")

    def __lt__(self, other):
        return self.probability < other.probability

    def __le__(self, other):
        return self.probability <= other.probability

    def __eq__(self, other):
        return self.probability == other.probability

    def __gt__(self, other):
        return self.probability > other.probability

    def __ge__(self, other):
        return self.probability >= other.probability


class SingleMeasurementJointHypothesis(Type, UserDict):
    """Joint Hypothesis base type

    """

    hypotheses = Property(
        Hypothesis,
        doc='Association hypotheses')

    def __new__(cls, hypotheses):
        if all(isinstance(hypothesis, SingleMeasurementDistanceHypothesis)
               for hypothesis in hypotheses.values()):
            return super().__new__(SingleMeasurementDistanceJointHypothesis)
        else:
            raise NotImplementedError

    def __init__(self, hypotheses):
        super().__init__(hypotheses)
        self.data = self.hypotheses

    @abstractmethod
    def __lt__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __le__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __gt__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __ge__(self, other):
        raise NotImplementedError


class SingleMeasurementDistanceJointHypothesis(
   SingleMeasurementJointHypothesis):
    """Distance scored hypothesis subclass.

    Notes
    -----
    As smaller distance is 'better', comparison logic is reversed
    i.e. smaller distance is a greater likelihood.
    """

    def __init__(self, hypotheses):
        super().__init__(hypotheses)

    @property
    def distance(self):
        return sum(hypothesis.distance for hypothesis in self.data.values())

    def __lt__(self, other):
        return self.distance > other.distance

    def __le__(self, other):
        return self.distance >= other.distance

    def __eq__(self, other):
        return self.distance == other.distance

    def __gt__(self, other):
        return self.distance < other.distance

    def __ge__(self, other):
        return self.distance <= other.distance
