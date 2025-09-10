import warnings
from abc import abstractmethod

from .base import Hypothesiser
from ..base import Property
from ..types.multihypothesis import MultipleHypothesis
from ..types.numeric import Probability
from ..types.prediction import Prediction


class _MFAHypothesiser(Hypothesiser):

    @property
    @abstractmethod
    def _hypothesisers(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def _transition_matrix(self):
        raise NotImplementedError()

    @abstractmethod
    def _tag(self, detections_tuple, hypothesis, hypothesis_index):
        return detections_tuple.index(hypothesis.measurement) + 1 if hypothesis else 0

    def hypothesise(self, track, detections, timestamp, detections_tuple, **kwargs):
        """Form hypotheses for associations between Detections and a given track.

        Parameters
        ----------
        track: :class:`~.Track`
            The track object to hypothesise on
        detections : set of :class:`~.Detection`
            Retrieved measurements
        timestamp : datetime
            Time of the detections/predicted states
        detections_tuple : tuple of :class:`~.Detection`
            Original tuple of detections required for consistent indexing
        Returns
        -------
        : :class:`~.MultipleHypothesis`
            A container of :class:`~.SingleProbabilityHypothesis` objects, pertaining to individual
            component-detection hypotheses
        """

        # Check to make sure all detections are obtained from the same time
        timestamps = {detection.timestamp for detection in detections}
        if len(timestamps) > 1:
            warnings.warn("All detections should have the same timestamp")

        hypothesisers = self._hypothesisers
        n_hyps = len(hypothesisers)
        transition_matrix = self._transition_matrix

        hypotheses = list()
        component_weight_norm = Probability.sum(
                component.weight for component in track.state.components) * n_hyps
        for component in track.state.components:
            for hyp_index, hypothesiser in enumerate(hypothesisers):
                # Get hypotheses for that component for all measurements
                component_hypotheses = hypothesiser.hypothesise(
                    component, detections, timestamp, **kwargs)
                for hypothesis in component_hypotheses:
                    # Update component tag and weight
                    det_hyp_tag = self._tag(detections_tuple, hypothesis, hyp_index)
                    new_weight = Probability(component.weight * hypothesis.weight)
                    if transition_matrix and component.tag:
                        new_weight *= transition_matrix[component.tag[-1][1]][hyp_index]
                    new_weight /= component_weight_norm
                    hypothesis.prediction = \
                        Prediction.from_state(
                            hypothesis.prediction,
                            tag=[*component.tag, det_hyp_tag],  # TODO: Avoid dependency on indexes
                            weight=new_weight,
                        )
                    hypotheses.append(hypothesis)
        # Create Multiple Hypothesis and add to list
        hypotheses = MultipleHypothesis(hypotheses)

        return hypotheses


class MFAHypothesiser(_MFAHypothesiser):
    """Multi-Frame Assignment Hypothesiser based on an underlying Hypothesiser

    Generates a list of SingleHypotheses pertaining to individual component-detection hypotheses.
    Each hypothesis contains a prediction with the last tag corresponding to the detection index

    Note
    ----
    This is to be used in conjunction with the :class:`~.MFADataAssociator`

    References
    ----------
    1. Xia, Y., Granström, K., Svensson, L., García-Fernández, Á.F., and Williams, J.L.,
       2019. Multiscan Implementation of the Trajectory Poisson Multi-Bernoulli Mixture Filter.
       J. Adv. Information Fusion, 14(2), pp. 213–235.
    """

    hypothesiser: Hypothesiser = Property(
        doc="Underlying hypothesiser used to generate detection-target pairs")

    @property
    def _hypothesisers(self):
        return [self.hypothesiser]

    @property
    def _transition_matrix(self):
        return None

    def _tag(self, detections_tuple, hypothesis, hypothesis_index):
        return detections_tuple.index(hypothesis.measurement) + 1 if hypothesis else 0


class MHMFAHypothesiser(_MFAHypothesiser):
    """Multi-Hypothesier Multi-Frame Assignment Hypothesiser based on underlying Hypothesisers

    Generates a list of SingleHypotheses pertaining to individual component-detection hypotheses
    per hypothesiser.
    Each hypothesis contains a prediction with the last tag corresponding to detection index and
    hypothesiser index

    Note
    ----
    This is to be used in conjunction with the :class:`~.MFADataAssociator`
    """

    hypothesisers: list[Hypothesiser] = Property(
        doc="Underlying hypothesisers used to generate detection-target pairs. Note that these "
            "should not normalise weights")
    transition_matrix: list[list[Probability]] = Property(
        default=None,
        doc="n by n transition matrix to switch between hypothesisers. Default `None`"
    )

    @property
    def _hypothesisers(self):
        return self.hypothesisers

    @property
    def _transition_matrix(self):
        return self.transition_matrix

    def _tag(self, detections_tuple, hypothesis, hypothesis_index):
        return (
            detections_tuple.index(hypothesis.measurement) + 1 if hypothesis else 0,
            hypothesis_index
            )
