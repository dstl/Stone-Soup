# -*- coding: utf-8 -*-
from .base import Hypothesiser
from ..base import Property
from ..types.multihypothesis import MultipleHypothesis
from ..types.prediction import (TaggedWeightedGaussianStatePrediction,
                                WeightedGaussianStatePrediction)
from ..types.state import TaggedWeightedGaussianState


class GaussianMixtureHypothesiser(Hypothesiser):
    """Gaussian Mixture Prediction Hypothesiser based on an underlying Hypothesiser

    Generates a list of :class:`MultipleHypothesis`, where each
    MultipleHypothesis in the list contains SingleHypotheses
    pertaining to an individual component-detection hypothesis
    """

    hypothesiser = Property(
        Hypothesiser,
        doc="Underlying hypothesiser used to generate detection-target pairs")
    order_by_detection = Property(
        bool,
        default=False,
        doc="Flag to order the :class:`~.MultipleHypothesis` "
            "list by detection or component")

    def hypothesise(self, components, detections, timestamp):
        """Form hypotheses for associations between Detections and Gaussian
        Mixture components.

        Parameters
        ----------
        components : :class:`list`
            List of :class:`~.WeightedGaussianState` components
            representing the state of the target space
        detections : list of :class:`Detection`
            Retrieved measurements
        timestamp : datetime
            Time of the detections/predicted states

        Returns
        -------
        list of :class:`~.MultipleHypothesis`
            Each :class:`~.MultipleHypothesis` in the list contains
            a list of :class:`~SingleHypothesis` pertaining
            to the same Gaussian component unless
            order_by_detection is true, then they
            pertain to the same Detection.
        """

        # Check to make sure all detections are obtained from the same time
        timestamps = set([detection.timestamp for detection in detections])
        if len(timestamps) > 1:
            raise ValueError("All detections must have the same timestamp")

        hypotheses = list()
        for component in components:
            # Get hypotheses for that component for all measurements
            component_hypotheses = self.hypothesiser.hypothesise(component,
                                                                 detections,
                                                                 timestamp)
            for hypothesis in component_hypotheses:
                if isinstance(component, TaggedWeightedGaussianState):
                    hypothesis.prediction = \
                        TaggedWeightedGaussianStatePrediction(
                            tag=component.tag if component.tag != "birth"
                            else None,
                            weight=component.weight,
                            state_vector=hypothesis.prediction.state_vector,
                            covar=hypothesis.prediction.covar,
                            timestamp=hypothesis.prediction.timestamp
                            )
                else:
                    hypothesis.prediction = WeightedGaussianStatePrediction(
                        weight=component.weight,
                        state_vector=hypothesis.prediction.state_vector,
                        covar=hypothesis.prediction.covar,
                        timestamp=hypothesis.prediction.timestamp
                    )
            # Create Multiple Hypothesis and add to list
            if len(component_hypotheses) > 0:
                hypotheses.append(MultipleHypothesis(component_hypotheses))

        # Reorder list of MultipleHypothesis so that they are ordered
        # by detection, not component
        if self.order_by_detection:
            single_hypothesis_list = list()
            # Retrieve all single hypotheses
            for multiple_hypothesis in hypotheses:
                for single_hypothesis in multiple_hypothesis:
                    single_hypothesis_list.append(single_hypothesis)
            reordered_hypotheses = list()
            # Get miss detected components
            miss_detections_hypothesis = MultipleHypothesis(
                [x for x in single_hypothesis_list if not x])
            
            # Get Soft detections
            soft_detect_list = [x for x in single_hypothesis_list if x]
            for detection in detections:
                if hasattr(detection, 'components') == True:      
                    for sub_detection in detection.components:
                        # Create multiple hypothesis per GM component
                        detection_multiple_hypothesis = \
                            MultipleHypothesis(list([hypothesis for hypothesis in soft_detect_list
                                                if hypothesis.measurement.components == sub_detection])) 
                        # Add to new list
                        reordered_hypotheses.append(detection_multiple_hypothesis)                
                else:
                    # Create multiple hypothesis per detection
                    detection_multiple_hypothesis = \
                        MultipleHypothesis(list([hypothesis for hypothesis in single_hypothesis_list
                                            if hypothesis.measurement == detection]))
                    # Add to new list
                    reordered_hypotheses.append(detection_multiple_hypothesis)                    
            # Add miss detected hypothesis to end
            reordered_hypotheses.append(miss_detections_hypothesis)
            # Assign reordered list to original list
            hypotheses = reordered_hypotheses

        return hypotheses
