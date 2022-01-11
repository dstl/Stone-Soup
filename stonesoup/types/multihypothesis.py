# -*- coding: utf-8 -*-
from collections.abc import Sized, Iterable, Container
from typing import Sequence

from .detection import MissedDetection
from .numeric import Probability
from ..base import Property
from ..types import Type
from ..types.detection import Detection
from ..types.hypothesis import SingleHypothesis, CompositeHypothesis
from ..types.prediction import Prediction


class MultipleHypothesis(Type, Sized, Iterable, Container):
    """Multiple Hypothesis base type

    A Multiple Hypothesis is a container to store a collection of hypotheses.
    """

    single_hypotheses: Sequence[SingleHypothesis] = Property(
        default=None,
        doc="The initial list of :class:`~.SingleHypothesis`. Default `None` "
            "which initialises with empty list.")
    normalise: bool = Property(
        default=False,
        doc="Normalise probabilities of :class:`~.SingleHypothesis`. Default "
            "is `False`.")
    total_weight: float = Property(
        default=1,
        doc="When normalising, weights will sum to this. Default is 1.")

    def __init__(self, single_hypotheses=None, normalise=False, *args,
                 **kwargs):
        if single_hypotheses is None:
            single_hypotheses = []

        if any(not isinstance(hypothesis, SingleHypothesis)
               for hypothesis in single_hypotheses):
            raise ValueError("Cannot form MultipleHypothesis out of "
                             "non-SingleHypothesis inputs!")

        super().__init__(single_hypotheses, normalise, *args, **kwargs)

        # normalise the weights of 'single_hypotheses', if indicated
        if self.normalise:
            self.normalise_probabilities()

    def __len__(self):
        return self.single_hypotheses.__len__()

    def __contains__(self, index):
        # check if 'single_hypotheses' contains any SingleHypotheses with
        # Detection matching 'index'
        if isinstance(index, Detection):
            for hypothesis in self.single_hypotheses:
                if hypothesis.measurement is index:
                    return True
            return False

        # check if 'single_hypotheses' contains any SingleHypotheses with
        # Prediction matching 'index'
        if isinstance(index, Prediction):
            for hypothesis in self.single_hypotheses:
                if hypothesis.prediction is index:
                    return True
            return False

        # check if 'single_hypotheses' contains any SingleHypotheses
        # matching 'index'
        if isinstance(index, SingleHypothesis):
            return index in self.single_hypotheses

    def __iter__(self):
        for hypothesis in self.single_hypotheses:
            yield hypothesis

    def __getitem__(self, index):

        # retrieve SingleHypothesis by array index
        if isinstance(index, int):
            return self.single_hypotheses[index]

        # retrieve SingleHypothesis by measurement
        if isinstance(index, Detection):
            for hypothesis in self.single_hypotheses:
                if hypothesis.measurement is index:
                    return hypothesis
            return None

        # retrieve SingleHypothesis by prediction
        if isinstance(index, Prediction):
            for hypothesis in self.single_hypotheses:
                if hypothesis.prediction is index:
                    return hypothesis
            return None

    def normalise_probabilities(self, total_weight=None):
        if total_weight is None:
            total_weight = self.total_weight

        # verify that SingleHypotheses composing this MultipleHypothesis
        # all have Probabilities
        if any(not hasattr(hypothesis, 'probability')
               for hypothesis in self.single_hypotheses):
            raise ValueError("MultipleHypothesis not composed of Probability"
                             " hypotheses!")

        sum_weights = Probability.sum(
            hypothesis.probability for hypothesis in self.single_hypotheses)

        for hypothesis in self.single_hypotheses:
            hypothesis.probability =\
                (hypothesis.probability * total_weight)/sum_weights

    def get_missed_detection_probability(self):
        for hypothesis in self.single_hypotheses:
            if isinstance(hypothesis.measurement, MissedDetection):
                if hasattr(hypothesis, 'probability'):
                    return hypothesis.probability
        return None


class MultipleCompositeHypothesis(Type, Sized, Iterable, Container):
    """Multiple composite hypothesis type

    A Multiple Composite Hypothesis is a container to store a collection of composite hypotheses.

    Interfaces the same as MultipleHypothesis, but permits different input, hence methods are
    redefined.
    """

    single_hypotheses: Sequence[CompositeHypothesis] = Property(
        default=None,
        doc="The initial list of :class:`~.CompositeHypothesis`. Default `None` which initialises "
            "with empty list.")
    normalise: bool = Property(
        default=False,
        doc="Normalise probabilities of :class:`~.CompositeHypothesis`. Default is `False`.")
    total_weight: float = Property(
        default=1,
        doc="When normalising, weights will sum to this. Default is 1.")

    def __init__(self, single_hypotheses=None, normalise=False, *args,
                 **kwargs):
        if single_hypotheses is None:
            single_hypotheses = []

        if not all(isinstance(hypothesis, CompositeHypothesis)
                   for hypothesis in single_hypotheses):
            raise ValueError("Cannot form MultipleHypothesis out of "
                             "non-CompositeHypothesis inputs!")

        super().__init__(single_hypotheses, normalise, *args, **kwargs)

        # normalise the weights of 'single_hypotheses', if indicated
        if self.normalise:
            self.normalise_probabilities()

    def __contains__(self, index):
        # cannot check instance index is detection or prediction as composite hypotheses create
        # their own composite detections and predictions

        # check if 'single_hypotheses' contains any CompositeHypotheses matching 'index'
        # use `is` as standard list __contains__ checks for equality which may not work in cases
        # where hypotheses do not all share same attributes
        if isinstance(index, CompositeHypothesis):
            return any(index is single_hypothesis for single_hypothesis in self.single_hypotheses)

    def __getitem__(self, index):

        return self.single_hypotheses.__getitem__(index)

    def __iter__(self):
        return self.single_hypotheses.__iter__()

    def __len__(self):
        return self.single_hypotheses.__len__()

    def normalise_probabilities(self, total_weight=None):
        if total_weight is None:
            total_weight = self.total_weight

        # verify that SingleHypotheses composing this MultipleHypothesis
        # all have Probabilities
        if any(not hasattr(hypothesis, 'probability')
               for hypothesis in self.single_hypotheses):
            raise ValueError(
                "MultipleHypothesis not composed of composite hypotheses with probabilities")

        sum_weights = Probability.sum(
            hypothesis.probability for hypothesis in self.single_hypotheses)

        # this will NOT affect the probabilities of each composite hypothesis' sub-hypotheses
        for hypothesis in self.single_hypotheses:
            hypothesis.probability = \
                (hypothesis.probability * total_weight) / sum_weights

    def get_missed_detection_probability(self):
        for hypothesis in self.single_hypotheses:
            if hasattr(hypothesis, 'probability') and not hypothesis:
                return hypothesis.probability
        return None
