# -*- coding: utf-8 -*-

import numpy as np

from ..base import Property
from ..types import Hypothesis
from .numeric import Probability
from .detection import MissedDetection
from .hypothesis import SingleMeasurementHypothesis


class MultipleMeasurementHypothesis(Hypothesis):
    """Multiple Measurement Hypothesis base type

    A Multiple Measurement Hypothesis relates to a situation where there is
    one Target with a predicted position, and multiple measurements that could
    be associated with that Target.  'single_measurement_hypotheses' contains
    the hypotheses for each of these individual measurements to be associated
    with that Target; the likelihood of each association is stored in the
    'probability' or 'distance' parameter of the hypothesis.  The
    'MissedDetection' hypothesis is also included in
    'single_measurement_hypotheses'.  If one of these hypotheses has been
    selected as the *correct* association, then it is linked to
    'selected_hypothesis'.

    Properties
    ----------
    single_measurement_hypotheses : :class:`list`
        list of Target/Measurement association hypotheses
    selected_hypothesis : :class:`SingleMeasurementHypothesis`
        the selected *correct* hypothesis
    """

    single_measurement_hypotheses = Property(
        [SingleMeasurementHypothesis],
        doc='Hypotheses of Measurements being associated with a Target.'
    )
    selected_hypothesis = Property(
        SingleMeasurementHypothesis,
        default=None,
        doc="The Hypothesis that was selected as the association with a Target.")

    @property
    def measurement(self):
        return self.single_measurement_hypotheses[0].measurement

    @property
    def prediction(self):
        return self.single_measurement_hypotheses[0].prediction

    @property
    def measurement_prediction(self):
        return self.single_measurement_hypotheses[0].measurement_prediction

    @property
    def timestamp(self):
        return self.single_measurement_hypotheses[0].measurement.timestamp

    def __bool__(self):
        if self.selected_hypothesis is not None:
            return not isinstance(self.selected_hypothesis.measurement, MissedDetection)
        else:
            raise Exception('Cannot check whether a '
                            'MultipleMeasurementHypothesis.'
                            'selected_measurement is a MissedDetection before'
                            ' it has been set!')

    def set_selected_hypothesis(self, selected_hypothesis):
        if any(np.array_equal(hypothesis.measurement.state_vector,
                              selected_hypothesis.measurement.state_vector)
               for hypothesis in self.single_measurement_hypotheses):
            self.selected_hypothesis = selected_hypothesis
        else:
            raise Exception('Cannot set MultipleMeasurementHypothesis.'
                            'selected_hypothesis to a value not contained in'
                            ' MultipleMeasurementHypothesis.'
                            'single_measurement_hypotheses!')

    def get_selected_hypothesis(self):
        if self.selected_hypothesis is not None:
            return self.selected_hypothesis
        else:
            raise Exception('best hypothesis in MultipleMeasurementHypothesis'
                            ' not selected, so it cannot be returned!')


class ProbabilityMultipleMeasurementHypothesis(MultipleMeasurementHypothesis):
    """Probability-scored multiple measurement hypothesis.

    Sub-type of MultipleMeasurementHypothesis where the hypotheses must be of
    type 'SingleMeasurementProbabilityHypothesis'. One of the hypotheses MUST
    be the MissedDetection hypothesis.  Used with Probabilistic Data
    Association (PDA).
    """

    def __init__(self, single_measurement_hypotheses, *args, **kwargs):
        super().__init__(single_measurement_hypotheses, *args, **kwargs)

    def normalize_probabilities(self):
        sum_weights = Probability.sum(hypothesis.probability for hypothesis in self.single_measurement_hypotheses)

        for hypothesis in self.single_measurement_hypotheses:
            hypothesis.probability = hypothesis.probability/sum_weights

        #return self

    def get_missed_detection_probability(self):
        for hypothesis in self.single_measurement_hypotheses:
            if isinstance(hypothesis.measurement, MissedDetection):
                return hypothesis.probability
        return None
