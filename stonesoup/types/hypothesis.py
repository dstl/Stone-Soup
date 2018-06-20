# -*- coding: utf-8 -*-

from abc import abstractclassmethod
from collections import UserDict

from .base import Type
from ..base import Property
from .detection import Detection
from .prediction import MeasurementPrediction, Prediction


class Hypothesis(Type):
    """Hypothesis base type

    """

    prediction = Property(
        Prediction,
        doc="Predicted track state")
    measurement_prediction = Property(
        MeasurementPrediction,
        doc="Track prediction in measurement space")
    detection = Property(
        Detection,
        doc="Detection used for hypothesis and updating")

    @abstractclassmethod
    def __lt__(self, other):
        raise NotImplemented

    @abstractclassmethod
    def __le__(self, other):
        raise NotImplemented

    @abstractclassmethod
    def __eq__(self, other):
        raise NotImplemented

    @abstractclassmethod
    def __gt__(self, other):
        raise NotImplemented

    @abstractclassmethod
    def __ge__(self, other):
        raise NotImplemented


class DistanceHypothesis(Hypothesis):
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


class JointHypothesis(Type, UserDict):
    """Joint Hypothesis base type

    """

    hypotheses = Property(
        Hypothesis,
        doc='Association hypotheses')

    def __new__(cls, hypotheses):
        if all(isinstance(hypothesis, DistanceHypothesis)
               for hypothesis in hypotheses.values()):
            return super().__new__(DistanceJointHypothesis)
        else:
            raise NotImplementedError

    def __init__(self, hypotheses):
        super().__init__(hypotheses)
        self.data = self.hypotheses

    @abstractclassmethod
    def __lt__(self, other):
        raise NotImplemented

    @abstractclassmethod
    def __le__(self, other):
        raise NotImplemented

    @abstractclassmethod
    def __eq__(self, other):
        raise NotImplemented

    @abstractclassmethod
    def __gt__(self, other):
        raise NotImplemented

    @abstractclassmethod
    def __ge__(self, other):
        raise NotImplemented


class DistanceJointHypothesis(JointHypothesis):
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
