# -*- coding: utf-8 -*-

from abc import abstractmethod
from collections import UserDict
import numpy as np

from .base import Type
from ..base import Property
from .detection import Detection
from .prediction import MeasurementPrediction, Prediction
from ..types.numeric import Probability


class Hypothesis(Type):
    """Hypothesis base type

    A Hypothesis has sub-types:

    'SingleMeasurementHypothesis', which consists of a prediction for a single
    Track and a single Measurement that *might* be associated with it

    'MultipleMeasurementHypothesis', which consists of a prediction for a
    single Track and multiple Measurements of which one *might* be associated
    with it
    """


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


class SingleMeasurementProbabilityHypothesis(SingleMeasurementHypothesis):
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


class JointHypothesis(Type, UserDict):
    """Joint Hypothesis base type

    A Joint Hypothesis consists of multiple Hypothesese, each with a single
    Track and a single Prediction.  A Joint Hypothesis can be a
    'ProbabilityJointHypothesis' or a 'DistanceJointHypothesis', with a
    probability or distance that is a function of the Hypothesis
    probabilities.  Multiple Joint Hypotheses can be compared to see which is
    most likely to be the "correct" hypothesis.

    Note: In reality, the property 'hypotheses' is a dictionary where the
    entries have the form 'Track: Hypothesis'.  However, we cannot define
    it this way because then Hypothesis imports Track, and Track imports
    Update, and Update imports Hypothesis, which is a circular import.
    """

    hypotheses = Property(
        Hypothesis,
        doc='Association hypotheses')

    def __new__(cls, hypotheses):
        if all(isinstance(hypothesis, SingleMeasurementDistanceHypothesis)
               for hypothesis in hypotheses.values()):
            return super().__new__(DistanceJointHypothesis)
        elif all(isinstance(hypothesis, SingleMeasurementProbabilityHypothesis)
                 for hypothesis in hypotheses.values()):
            return super().__new__(ProbabilityJointHypothesis)
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


class ProbabilityJointHypothesis(JointHypothesis):
    """Probability-scored Joint Hypothesis subclass.

    """

    probability = Property(
        Probability,
        default=None,
        doc='Probability of the Joint Hypothesis')

    def __init__(self, hypotheses, *args, **kwargs):
        super().__init__(hypotheses, *args, **kwargs)
        self.probability = Probability(np.prod(
            [hypothesis.probability for hypothesis in hypotheses.values()]))

    def normalize(self):
        sum_probability = sum(hypothesis.probability
                              for hypothesis in self.hypotheses.values())
        for hypothesis in self.hypotheses.values():
            hypothesis.probability /= sum_probability

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


class DistanceJointHypothesis(
   JointHypothesis):
    """Distance scored Joint Hypothesis subclass.

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
