# -*- coding: utf-8 -*-
from collections.abc import Sized, Iterable, Container
import numbers

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

    Optional inputs to 'init' function:
    -----------------------------------
    normalise - if True, normalise probabilities of 'single_hypotheses'
    total_weight - when normalising, all weights on SingleHypotheses should
                    add up to this
    """

    single_hypotheses = Property(
        [SingleHypothesis],
        default=None,
        doc="The initial list of SingleHypotheses. Default `None` which"
            " initialises with empty list.")

    def __init__(self, single_hypotheses=None, *args, **kwargs):
        if single_hypotheses is None:
            single_hypotheses = []

        if any(not isinstance(hypothesis, SingleHypothesis)
               for hypothesis in single_hypotheses):
            raise ValueError("Cannot form MultipleHypothesis out of "
                             "non-SingleHypothesis inputs!")

        self.single_hypotheses = single_hypotheses

        # normalise the weights of 'single_hypotheses', if indicated
        if 'normalise' in kwargs:
            if kwargs['normalise'] is True:
                weight = 1
                if 'total_weight' in kwargs:
                    if isinstance(kwargs['total_weight'], numbers.Integral):
                        weight = kwargs['total_weight']
                    else:
                        raise ValueError('Cannot normalise weights to a '
                                         'non-number value!')
                self.normalise_probabilities(weight)

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

    def normalise_probabilities(self, total_weight):
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
