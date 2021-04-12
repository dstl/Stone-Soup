# -*- coding: utf-8 -*-

from abc import abstractmethod
from collections import UserDict
from typing import Sequence

import numpy as np

from .base import Type
from ..base import Property
from .detection import Detection, MissedDetection, CompositeMissedDetection, CompositeDetection
from .prediction import MeasurementPrediction, Prediction, CompositePrediction, \
    CompositeMeasurementPrediction
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


class SingleProbabilityHypothesis(SingleHypothesis):
    """Single Measurement Probability scored hypothesis subclass.

    """

    probability: Probability = Property(
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

    @property
    def weight(self):
        return self.probability


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


class ProbabilityJointHypothesis(JointHypothesis):
    """Probability-scored Joint Hypothesis subclass.

    """

    probability: Probability = Property(default=None, doc='Probability of the Joint Hypothesis')

    def __init__(self, hypotheses, *args, **kwargs):
        super().__init__(hypotheses, *args, **kwargs)
        self.probability = Probability(np.prod(
            [hypothesis.probability for hypothesis in hypotheses.values()]))

    def normalise(self):
        sum_probability = Probability.sum(
            hypothesis.probability for hypothesis in self.hypotheses.values())
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


class CompositeHypothesis(SingleHypothesis):
    """Composite hypothesis class.

    Contains a sequence of sub-hypotheses.
    To be used in conjunction with composite detection types.
    """

    prediction: CompositePrediction = Property(doc="Predicted composite track state")
    measurement: CompositeDetection = Property(doc="Composite detection used for hypothesis and "
                                                   "updating")
    measurement_prediction: CompositeMeasurementPrediction = Property(
        default=None, doc="Optional composite track prediction in composite measurement space")
    sub_hypotheses: Sequence[SingleHypothesis] = Property(
        default=None,
        doc="Sequence of sub-hypotheses comprising the composite hypothesis"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.sub_hypotheses is None:
            self.sub_hypotheses = list()

    def __bool__(self):
        return (not isinstance(self.measurement, CompositeMissedDetection)) and \
               (self.measurement is not None)

    def append(self, item):
        self.sub_hypotheses.append(item)


class CompositeProbabilityHypothesis(SingleProbabilityHypothesis, CompositeHypothesis):
    """Probability scored composite hypothesis subclass

    Notes
    -----
    Attempts to calculate hypothesis probability via product of sub-hypotheses' probabilities if no
    probability is defined
    """

    probability: Probability = Property(
        default=None,
        doc="Probability that composite detection is true location of composite prediction."
    )
    sub_hypotheses: Sequence[SingleProbabilityHypothesis] = Property(
        default=None,
        doc="Sequence of probability-scored sub-hypotheses comprising the composite hypothesis."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if any(not isinstance(sub_hypothesis, SingleProbabilityHypothesis)
               for sub_hypothesis in self.sub_hypotheses):
            raise ValueError(f"{type(self).__name__} must be comprised of "
                             f"SingleProbabilityHypothesis types")

        # Get probability from sub-hypotheses if undefined
        if self.probability is None:
            self.probability = 1
            for hypothesis in self.sub_hypotheses:
                self.probability *= hypothesis.probability
