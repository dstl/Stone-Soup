# -*- coding: utf-8 -*-
from .base import Hypothesiser
from ..base import Property


class FilteredDetectionsHypothesiser(Hypothesiser):
    """Wrapper for Hypothesisers - filters input data

    Wrapper for any type of hypothesiser - filters the 'detections' before
    they are fed into the hypothesiser.
    """

    hypothesiser = Property(
        Hypothesiser, doc="Hypothesiser that is being wrapped.")
    metadata_filter = Property(
        str, doc="Metadata attribute used to filter which detections "
                 "tracks are valid for association.")
    match_missing = Property(
        bool,
        default=True,
        doc="Match detections with missing metadata. Default 'True'.")

    def hypothesise(self, track, detections, *args, **kwargs):
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
