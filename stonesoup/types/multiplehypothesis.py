# -*- coding: utf-8 -*-

from ..base import Property
from ..types import Hypothesis, ProbabilityHypothesis


class MultipleHypothesis(Hypothesis):
    """Multiple Hypothesis base type

    """

    hypotheses = Property(
        [Hypothesis],
        doc="Set of hypotheses")


class ProbabilityMultipleHypothesis(MultipleHypothesis, ProbabilityHypothesis):
    """Probability-scored multiple hypothesis subclass.

    The ProbabilityMultipleHypothesis is actually a ProbabilityHypothesis with
    an additional Property called 'hypotheses'  This property contains all of
    the possible hypotheses, with their associated probabilities.  The other
    Properties of 'ProbabilityMultipleHypothesis' can be unused, or they can
    contain the same attributes as the selected/highest-probability
    hypothesis.
    """

    hypotheses = Property(
        [ProbabilityHypothesis],
        doc="Set of probability hypotheses, probability sums to 1")
