from . import Hypothesiser
from ..base import Property
from ..dataassociator.tree import DetectionKDTreeMixIn
from ..types.detection import MissedDetection
from ..types.multihypothesis import MultipleHypothesis
from ..types.prediction import Prediction
from ..types.state import TaggedWeightedGaussianState


class GaussianMixtureHypothesiser(Hypothesiser):
    """Gaussian Mixture Prediction Hypothesiser based on an underlying Hypothesiser

    Generates a list of :class:`MultipleHypothesis`, where each
    MultipleHypothesis in the list contains SingleHypotheses
    pertaining to an individual component-detection hypothesis
    """

    hypothesiser: Hypothesiser = Property(
        doc="Underlying hypothesiser used to generate detection-target pairs")
    order_by_detection: bool = Property(
        default=False,
        doc="Flag to order the :class:`~.MultipleHypothesis` "
            "list by detection or component")

    def generate_hypotheses(self, components, detections, timestamp, **kwargs):
        return {component: self.hypothesiser.hypothesise(
            component, detections, timestamp, **kwargs)
            for component in components}

    def hypothesise(self, components, detections, timestamp, **kwargs):
        """Form hypotheses for associations between Detections and Gaussian
        Mixture components.

        Parameters
        ----------
        components : list of :class:`~.WeightedGaussianState`
            Components representing the state of the target space
        detections : set of :class:`~.Detection`
            Retrieved measurements
        timestamp : datetime.datetime
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

        components_hypotheses = self.generate_hypotheses(
            components, detections, timestamp, **kwargs)

        hypotheses = list()
        for component, component_hypotheses in components_hypotheses.items():
            for hypothesis in component_hypotheses:
                if isinstance(component, TaggedWeightedGaussianState):
                    # Ensure that a birth component without a measurement retains
                    # the birth tag. This will prevent a track from being made
                    if component.tag == component.BIRTH and \
                            isinstance(hypothesis.measurement, MissedDetection):
                        tag = component.BIRTH
                    elif component.tag == component.BIRTH:
                        tag = None  # a new tag will be made
                    else:
                        tag = component.tag
                    hypothesis.prediction = \
                        Prediction.from_state(
                            component,
                            tag=tag,
                            state_vector=hypothesis.prediction.state_vector,
                            covar=hypothesis.prediction.covar,
                            timestamp=hypothesis.prediction.timestamp
                            )
                else:
                    hypothesis.prediction = Prediction.from_state(
                        component,
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
            for detection in detections:
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


class GaussianMixtureKDTreeHypothesiser(DetectionKDTreeMixIn, GaussianMixtureHypothesiser):
    pass
