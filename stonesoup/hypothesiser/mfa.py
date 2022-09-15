# -*- coding: utf-8 -*-
from .base import Hypothesiser
from ..base import Property
from ..types.multihypothesis import MultipleHypothesis
from ..types.prediction import TaggedWeightedGaussianStatePrediction


class MFAHypothesiser(Hypothesiser):
    """Multi-Frame Assignment Hypothesiser based on an underlying Hypothesiser

    Generates a list of SingleHypotheses pertaining to individual component-detection hypotheses

    References
    ----------
    1. Xia, Y., Granström, K., Svensson, L., García-Fernández, Á.F., and Williams, J.L.,
       2019. Multiscan Implementation of the Trajectory Poisson Multi-Bernoulli Mixture Filter.
       J. Adv. Information Fusion, 14(2), pp. 213–235.
    """

    hypothesiser: Hypothesiser = Property(
        doc="Underlying hypothesiser used to generate detection-target pairs")

    def hypothesise(self, track, detections, timestamp, **kwargs):
        """Form hypotheses for associations between Detections and a given track.

        Parameters
        ----------
        track: :class:`~.Track`
            The track object to hypothesise on
        detections : set of :class:`Detection`
            Retrieved measurements
        timestamp : datetime
            Time of the detections/predicted states
        Returns
        -------
        : :class:`~.MultipleHypothesis`
            A container of :class:`~SingleProbabilityHypothesis` objects, pertaining to individual
            component-detection hypotheses
        """

        # Check to make sure all detections are obtained from the same time
        timestamps = set([detection.timestamp for detection in detections])
        if len(timestamps) > 1:
            raise ValueError("All detections must have the same timestamp")

        hypotheses = list()
        detections_list = list(detections)
        for component in track.state.components:
            # Get hypotheses for that component for all measurements
            component_hypotheses = self.hypothesiser.hypothesise(component, detections, timestamp)
            for hypothesis in component_hypotheses:
                # Update component tag and weight
                det_idx = detections_list.index(hypothesis.measurement) + 1 if hypothesis else 0
                new_weight = component.weight * hypothesis.weight
                hypothesis.prediction = \
                    TaggedWeightedGaussianStatePrediction(
                        tag=[*component.tag, det_idx],
                        weight=new_weight,
                        state_vector=hypothesis.prediction.state_vector,
                        covar=hypothesis.prediction.covar,
                        timestamp=hypothesis.prediction.timestamp
                    )
                hypotheses.append(hypothesis)
        # Create Multiple Hypothesis and add to list
        hypotheses = MultipleHypothesis(hypotheses)

        return hypotheses
