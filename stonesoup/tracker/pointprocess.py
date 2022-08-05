from .base import Tracker
from ..base import Property
from ..reader import DetectionReader
from ..types.state import TaggedWeightedGaussianState
from ..types.mixture import GaussianMixture
from ..types.numeric import Probability
from ..types.track import Track
from ..updater import Updater
from ..hypothesiser.gaussianmixture import GaussianMixtureHypothesiser
from ..mixturereducer.gaussianmixture import GaussianMixtureReducer


class PointProcessMultiTargetTracker(Tracker):
    """
    Base class for Gaussian Mixture (GM) style implementations of
    point process derived filters
    """
    detector: DetectionReader = Property(
        doc="Detector used to generate detection objects.")
    updater: Updater = Property(
        doc="Updater used to update the objects to their new state.")
    hypothesiser: GaussianMixtureHypothesiser = Property(
        doc="Association algorithm to pair predictions to detections")
    reducer: GaussianMixtureReducer = Property(
        doc="Reducer used to reduce the number of components in the mixture.")
    extraction_threshold: Probability = Property(
        default=0.9,
        doc="Threshold to extract components from the mixture.")
    birth_component: TaggedWeightedGaussianState = Property(
        default=None,
        doc="The birth component. The weight should be "
            "equal to the mean of the expected number of "
            "births per timestep (Poission distributed). "
            "The tag should be "
            ":attr:`TaggedWeightedGaussianState.BIRTH`")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_tracks = dict()
        self.gaussian_mixture = GaussianMixture()

    @property
    def tracks(self):
        tracks = set()
        for track in self.target_tracks.values():
            tracks.add(track)
        return tracks

    def __iter__(self):
        self.detector_iter = iter(self.detector)
        return super().__iter__()

    def update_tracks(self):
        """
        Updates the tracks (:class:`Track`) associated with the filter.

        Parameters
        ==========
        self : :class:`GaussianMixtureMultiTargetTracker`
            Current GM Multi Target Tracker at time :math:`k`

        Note
        ======
        Each track shares a unique tag with its associated component
        """
        for component in self.gaussian_mixture:
            tag = component.tag
            if tag != component.BIRTH:  # Sanity check for birth component
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

    def __next__(self):
        time, detections = next(self.detector_iter)
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
        return time, self.tracks

    def end_tracks(self):
        """
        Ends the tracks (:class:`Track`) that do not have an associated
        component within the filter.

        Parameters
        ==========
        self : :class:`GaussianMixtureMultiTargetTracker`
            Current GM Multi Target Tracker at time :math:`k`
        """
        component_tags = {component.tag for component in self.gaussian_mixture}
        # Delete the track
        for key in self.target_tracks.keys() - component_tags:
            del self.target_tracks[key]

    @property
    def extracted_target_states(self):
        """
        Extract all target states from the Gaussian Mixture that
        are above an extraction threshold.
        """
        return [component
                for component in self.gaussian_mixture
                if component.weight > self.extraction_threshold]

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
