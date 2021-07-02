# -*- coding: utf-8 -*-

from abc import abstractmethod
from collections import UserDict
from typing import Sequence

import numpy as np

from .base import Type
from .detection import Detection, MissedDetection, CompositeMissedDetection, CompositeDetection
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
        return self.probability == other.probability

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


class CompositeHypothesis(Hypothesis):
    """Composite hypothesis class.

    Contains a sequence of sub-hypotheses.
    To be used in conjunction with composite detection types.
    """

    prediction: CompositePrediction = Property(doc="Predicted composite track state")
    measurement: CompositeDetection = Property(doc="Composite detection used for hypothesis and "
                                                   "updating")
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

    def __contains__(self, item):
        return self.sub_hypotheses.__contains__(item)

    def __getitem__(self, index):
        return self.sub_hypotheses[index]

    def __setitem__(self, index, value):
        set_item = self.sub_hypotheses.__setitem__(index, value)
        return set_item

    def __delitem__(self, index):
        del_item = self.sub_hypotheses.__delitem__(index)
        return del_item

    def __iter__(self):
        return iter(self.sub_hypotheses)

    def __len__(self):
        return len(self.sub_hypotheses)

    def insert(self, index, value):
        inserted_state = self.sub_hypotheses.insert(index, value)
        return inserted_state

    def append(self, value):
        """Add value at end of :attr:`sub_states`.

        Parameters
        ----------
        value: State
            A state object to be added at the end of :attr:`sub_states`.
        """
        self.sub_hypotheses.append(value)

    @property
    def measurement_prediction(self):
        # check if any are none
        return CompositeMeasurementPrediction(
            [sub_hypothesis.measurement_prediction for sub_hypothesis in self.sub_hypotheses]
        )


class CompositeProbabilityHypothesis(ProbabilityHypothesis, CompositeHypothesis):
    """Probability scored composite hypothesis subclass.

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
            self.probability = Probability(1)
            for sub_hypothesis in self.sub_hypotheses:
                if sub_hypothesis or not self:
                    # If `self` is null-hypothesis, consider all sub-hypotheses' probabilities
                    # If `self` is not null, consider only non-null sub-hypotheses' probabilities
                    self.probability *= sub_hypothesis.probability
