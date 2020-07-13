# -*- coding: utf-8 -*-
from collections.abc import Sized, Iterable, Container

from ..base import Property
from ..types import Type
from ..types.detection import Detection
from ..types.hypothesis import SingleHypothesis
from ..types.prediction import Prediction
from .numeric import Probability
from .detection import MissedDetection


class MultipleHypothesis(Type, Sized, Iterable, Container):
    """Multiple Hypothesis base type

    A Multiple Hypothesis is a container to store a collection of hypotheses.
    """

    single_hypotheses = Property(
        [SingleHypothesis], default=None,
        doc="The initial list of :class:`~.SingleHypothesis`. Default `None` "
            "which initialises with empty list.")
    normalise = Property(
        bool, default=False,
        doc="Normalise probabilities of :class:`~.SingleHypothesis`. Default "
            "is `False`.")
    total_weight = Property(
        float, default=1,
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
