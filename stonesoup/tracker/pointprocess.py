# -*- coding: utf-8 -*-
from .base import Tracker
from ..base import Property
from ..reader import DetectionReader
from ..types.state import TaggedWeightedGaussianState
from ..types.mixture import GaussianMixture
from ..types.track import Track
from ..updater import Updater
from ..hypothesiser.gaussianmixture import GaussianMixtureHypothesiser
from ..mixturereducer.gaussianmixture import GaussianMixtureReducer
from ..buffered_generator import BufferedGenerator


class PointProcessMultiTargetTracker(Tracker):
    """
    Base class for Gaussian Mixture (GM) style implementations of
    point process derived filters
    """
    detector = Property(
        DetectionReader,
        default=None,
        doc="Detector used to generate detection objects.")
    updater = Property(
        Updater,
        default=None,
        doc="Updater used to update the objects to their new state.")
    gaussian_mixture = Property(
        GaussianMixture,
        doc="""Gaussian Mixture modelling
        the intensity over the target state space.""")
    hypothesiser = Property(
        GaussianMixtureHypothesiser,
        default=None,
        doc="Association algorithm to pair predictions to detections")
    reducer = Property(
        GaussianMixtureReducer,
        default=None,
        doc="Reducer used to reduce the number of components in the mixture.")
    extraction_threshold = Property(
        float,
        default=0.9,
        doc="Threshold to extract components from the mixture.")
    birth_component = Property(
        TaggedWeightedGaussianState,
        default=None,
        doc="""The birth component. The weight should be equal to the mean of the
        expected number of births per timestep (Poission distributed)""")
    target_tracks = Property(
        dict,
        doc="""The dictonary containing unique target tracks indexed
               by the UUID of the target""")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.gaussian_mixture is None:
            self.gaussian_mixture = GaussianMixture()
        if self.target_tracks is None:
            self.target_tracks = dict()

    @property
    def tracks(self):
        tracks = set()
        for track in self.target_tracks.values():
            tracks |= track
        return tracks

    def update_tracks(self):
        """
        Updates the tracks (:class:`Track`) associated with the filter.

        Parameters
        ==========
        self : :state:`GaussianMixtureMultiTargetTracker`
            Current GM Multi Target Tracker at time :math:`k`

        Note
        ======
        Each track shares a unique tag with its associated component
        """
        for component in self.gaussian_mixture:
            tag = component.tag
            if tag != 0:
                # Sanity check for birth component
                if tag in self.target_tracks:
                    # Track found, so update it
                    track = self.target_tracks[tag]
                    track.states.append(component)
                else:
                    # No Track found, so create a new one only if we are
                    # reasonably confident its a target
                    if component.weight > \
                            self.extraction_threshold:
                        self.target_tracks[tag] = Track([component], id=tag)


    @BufferedGenerator.generator_method
    def tracks_gen(self):
        for time, detections in self.detector:
            # Add birth component
            self.birth_component.timestamp = time
            self.gaussian_mixture.append(self.birth_component)
            # Perform GM Prediction and generate hypotheses
            hypotheses = self.hypothesiser.hypothesise(
                        self.gaussian_mixture.components,
                        detections,
                        time
                        )
            # Perform GM Update
            self.gaussian_mixture = self.updater.update(hypotheses)
            # Reduce mixture - Pruning and Merging
            self.gaussian_mixture.components = \
                self.reducer.reduce(self.gaussian_mixture.components)
            # Update the tracks
            self.update_tracks()
            self.end_tracks()
            yield time, self.tracks


    def end_tracks(self):
        """
        Ends the tracks (:class:`Track`) that do not have an associated
        component within the filter.

        Parameters
        ==========
        self : :state:`GaussianMixtureMultiTargetTracker`
            Current GM Multi Target Tracker at time :math:`k`
        """
        component_tags = [component.tag for component in self.gaussian_mixture]
        for tag in self.target_tracks:
            if tag not in component_tags:
                # Track doesn't have matching component, so end
                self.target_tracks[tag].active = False

    @property
    def extracted_target_states(self):
        """
        Extract all target states from the Gaussian Mixture that
        are above an extraction threshold.
        """
        if self.gaussian_mixture:
            extracted_states = \
                [x for x in self.gaussian_mixture if
                 x.weight > self.extraction_threshold]
        else:
            extracted_states = []
        return extracted_states

    @property
    def estimated_number_of_targets(self):
        """
        The number of hypothesised targets.
        """
        if self.gaussian_mixture:
            estimated_number_of_targets = sum(component.weight for component in
                                              self.gaussian_mixture)
        else:
            estimated_number_of_targets = 0
        return estimated_number_of_targets


class GMPHDTargetTracker(PointProcessMultiTargetTracker):
    """
    A implementation of the Gaussian Mixture
    Probability Hypothesis Density (GM-PHD) multi-target filter

    References
    ----------

    .. [1] B.-N. Vo and W.-K. Ma, “The Gaussian Mixture Probability Hypothesis
    Density Filter,” Signal Processing,IEEE Transactions on, vol. 54, no. 11,
    pp. 4091–4104, 2006..
    """


class GMLCCTargetTracker(PointProcessMultiTargetTracker):
    """
    A implementation of the Gaussian Mixture
    Linear Complexity with Cumulants (GM-LCC) multi-target filter

    References
    ----------

    .. [1] D. E. Clark and F. De Melo. “A Linear-Complexity Second-Order
        Multi-Object Filter via Factorial Cumulants”. In: 2018
        21st International Conference on Information Fusion (FUSION).
        2018.
        DOI: 10.23919/ICIF.2018.8455331. ..
    """
