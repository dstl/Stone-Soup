# -*- coding: utf-8 -*-

from abc import abstractmethod
from collections import UserDict
from typing import Sequence

import numpy as np

from .base import Type
from .detection import Detection, MissedDetection, CompositeDetection
from .prediction import MeasurementPrediction, Prediction, CompositePrediction, \
    CompositeMeasurementPrediction
from ..base import Property
from ..types.numeric import Probability


class Hypothesis(Type):
    """Hypothesis base type

    A Hypothesis has sub-types:

    'SingleHypothesis', which consists of a prediction for a single
    Track and a single Measurement that *might* be associated with it

    'MultipleHypothesis', which consists of a prediction for a
    single Track and multiple Measurements of which one *might* be associated
    with it
    """


class ProbabilityHypothesis(Hypothesis):
    probability: Probability = Property(
        doc="Probability that detection is true location of prediction")

    def __lt__(self, other):
        return self.probability < other.probability

    def __le__(self, other):
        return self.probability <= other.probability

    def __eq__(self, other):
        return isinstance(other, ProbabilityHypothesis) and self.probability == other.probability

    def __gt__(self, other):
        return self.probability > other.probability

    def __ge__(self, other):
        return self.probability >= other.probability

    @property
    def weight(self):
        return self.probability


class SingleHypothesis(Hypothesis):
    """A hypothesis based on a single measurement.

    """
    prediction: Prediction = Property(doc="Predicted track state")
    measurement: Detection = Property(doc="Detection used for hypothesis and updating")
    measurement_prediction: MeasurementPrediction = Property(
        default=None, doc="Optional track prediction in measurement space")

    def __bool__(self):
        return (not isinstance(self.measurement, MissedDetection)) and \
               (self.measurement is not None)


class SingleDistanceHypothesis(SingleHypothesis):
    """Distance scored hypothesis subclass.

    Notes
    -----
    As smaller distance is 'better', comparison logic is reversed
    i.e. smaller distance is a greater likelihood.
    """

    distance: float = Property(doc="Distance between detection and prediction")

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

    @property
    def weight(self):
        try:
            return 1 / self.distance
        except ZeroDivisionError:
            return float('inf')


class SingleProbabilityHypothesis(ProbabilityHypothesis, SingleHypothesis):
    """Single Measurement Probability scored hypothesis subclass."""


class JointHypothesis(Type, UserDict):
    """Joint Hypothesis base type

    A Joint Hypothesis consists of multiple Hypotheses, each with a single
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

    hypotheses: Hypothesis = Property(doc='Association hypotheses')

    def __new__(cls, hypotheses):
        if all(isinstance(hypothesis, SingleDistanceHypothesis)
               for hypothesis in hypotheses.values()):
            return super().__new__(DistanceJointHypothesis)
        elif all(isinstance(hypothesis, SingleProbabilityHypothesis)
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


class ProbabilityJointHypothesis(ProbabilityHypothesis, JointHypothesis):
    """Probability-scored Joint Hypothesis subclass."""

    probability: Probability = Property(
        default=None,
        doc="Probability that detection is true location of prediction. Defaults to `None`, "
            "whereby the probability is calculated as being the product of the constituent "
            "multiple-hypotheses' probabilities.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probability = Probability(np.prod(
            [hypothesis.probability for hypothesis in self.hypotheses.values()]))

    def normalise(self):
        sum_probability = Probability.sum(
            hypothesis.probability for hypothesis in self.hypotheses.values())
        for hypothesis in self.hypotheses.values():
            hypothesis.probability /= sum_probability


class DistanceJointHypothesis(JointHypothesis):
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


class CompositeHypothesis(SingleHypothesis):
    """Composite hypothesis type

    A composition of :class:`~.SingleHypothesis`.
    """

    sub_hypotheses: Sequence[SingleHypothesis] = Property(
        doc="Sequence of sub-hypotheses comprising the composite hypothesis. Must not be empty.")
    prediction: CompositePrediction = Property(
        doc="Predicted track state")
    measurement: CompositeDetection = Property(
        doc="Detection used for hypothesis and updating")
    measurement_prediction: CompositeMeasurementPrediction = Property(
        default=None, doc="Optional track prediction in measurement space")

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if len(self.sub_hypotheses) == 0:
            raise ValueError("Cannot create an empty composite hypothesis")

    def __bool__(self):
        # Is null hypothesis if all sub-hypotheses are null
        return any(sub_hypothesis.__bool__() for sub_hypothesis in self.sub_hypotheses)

    def __contains__(self, item):

        # Prediction matching `item`
        if isinstance(item, Prediction):
            for sub_hypothesis in self.sub_hypotheses:
                if sub_hypothesis.prediction is item:
                    return True
            return False

        # Detection matching `item`
        if isinstance(item, Detection):
            for sub_hypothesis in self.sub_hypotheses:
                if sub_hypothesis.measurement is item:
                    return True
            return False

        # Hypothesis matching `item`
        if isinstance(item, SingleHypothesis):
            return self.sub_hypotheses.__contains__(item)

    def __getitem__(self, index):

        # retrieve sub-hypothesis by index
        if isinstance(index, int):
            return self.sub_hypotheses.__getitem__(index)

        if isinstance(index, slice):
            return self.__class__(prediction=self.prediction.__getitem__(index),
                                  measurement=self.measurement.__getitem__(index),
                                  sub_hypotheses=self.sub_hypotheses.__getitem__(index))

        # retrieve sub-hypothesis by prediction
        if isinstance(index, Prediction):
            for sub_hypothesis in self.sub_hypotheses:
                if sub_hypothesis.prediction is index:
                    return sub_hypothesis
            return None

        # retrieve sub-hypothesis by measurement
        if isinstance(index, Detection):
            for sub_hypothesis in self.sub_hypotheses:
                if sub_hypothesis.measurement is index:
                    return sub_hypothesis
            return None

    def __iter__(self):
        return self.sub_hypotheses.__iter__()

    def __len__(self):
        return self.sub_hypotheses.__len__()


Hypothesis.register(CompositeHypothesis)  # noqa: E305


class CompositeProbabilityHypothesis(CompositeHypothesis, SingleProbabilityHypothesis):
    """Composite probability hypothesis type

    A composition of :class:`~.SingleProbabilityHypothesis`.

    Calculate hypothesis probability via product of sub-hypotheses' probabilities.
    Probability is 1 if there are no sub-hypotheses.
    """

    probability: Probability = Property(
        default=None,
        doc="Probability that detection is true location of prediction. Default is `None`, "
            "whereby probability is calculated as the product of sub-hypotheses' probabilities")
    sub_hypotheses: Sequence[SingleProbabilityHypothesis] = Property(
        default=None,
        doc="Sequence of probability-scored sub-hypotheses comprising the composite hypothesis."
    )

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if any(not isinstance(sub_hypothesis, SingleProbabilityHypothesis)
               for sub_hypothesis in self.sub_hypotheses):
            raise ValueError(
                f"{type(self).__name__} must be composed of SingleProbabilityHypothesis types")

        if self.probability is None:
            # Probability is product of sub-hypothesis probabilities
            self.probability = Probability(1)
            for sub_hypothesis in self.sub_hypotheses:
                if sub_hypothesis or not self:
                    # If `self` is null-hypothesis, consider all sub-hypotheses' probabilities
                    # If `self` is not null, consider only non-null sub-hypotheses' probabilities
                    self.probability *= sub_hypothesis.probability


ProbabilityHypothesis.register(CompositeProbabilityHypothesis)  # noqa: E305
