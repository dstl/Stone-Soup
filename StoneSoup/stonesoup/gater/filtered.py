from .base import Gater
from ..base import Property


class FilteredDetectionsGater(Gater):
    """Wrapper for Hypothesisers - filters input data

    Wrapper for any type of hypothesiser - filters the 'detections' before
    they are fed into the hypothesiser.
    """

    metadata_filter: str = Property(
        doc="Metadata attribute used to filter which detections tracks are valid for association.")
    match_missing: bool = Property(
        default=True,
        doc="Match detections with missing metadata. Default 'True'.")

    def hypothesise(self, track, detections, *args, **kwargs):
        """
        Parameters
        ==========
        track : :class:`Track`
            A track that contains the target's state
        detections : list of :class:`Detection`
            Retrieved measurements

        Returns
        =======
        : :class:`MultipleHypothesis`
            A list containing the hypotheses between each prediction-detections
            pair.

        Note:   The specific subclass of :class:`SingleHypothesis` returned
                depends on the :class:`Hypothesiser` used.

        """
        track_metadata = track.metadata.get(self.metadata_filter)

        if (track_metadata is None) and self.match_missing:
            match_detections = detections
        else:
            match_metadata = [track_metadata]
            if self.match_missing:
                match_metadata.append(None)

            match_detections = {
                detection for detection in detections
                if detection.metadata.get(
                        self.metadata_filter) in match_metadata}

        return self.hypothesiser.hypothesise(
            track, match_detections, *args, **kwargs)
