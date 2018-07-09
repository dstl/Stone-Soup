from .base import MetricGenerator
from ..base import Property
from ..types import Clutter


class SingleDetectionBasedMetrics(MetricGenerator):
    # Will break if tracks are based on anything other than a single detection
    # Insert code here
    # Detections are split between clutter and not clutter by using isinstance(detection, Clutter)

    # Output is a dictionary with keys equal to the metric names like 'confusion_matrix', 'track_completeness'...

    ''' Metrics generated at the end of a run '''
    def create_run_end_metrics(self, tracks, truth):

        metrics = {} # A dict
        metrics['number_of_targets'] = len(truth)
        metrics['number_of_tracks'] = len(tracks)
        metrics['track_to_target_ratio'] = len(tracks)/len(truth)

        track_truth_associations = self.associate_tracks_to_truth(tracks,truth)

    def associatie_tracks_to_truth(self,tracks,truth):

        # Some sort of magic association method that matches tracks to truths

        return track_truth_associations

    def update_confusion_matrix(self, ):
        # Live metric to update the confusion matrix after each update.


    ''' Metrics generated in line with the tracker'''

    def update_real_time_metrics(self,tracks,associations,detections):

        self.parse_associations(tracks,associations,detections)


    def parse_associations(self,tracks,associations,detections):
        # All very much pseudocode
        for track in tracks:
            if track not in self.tracks:
                self.tracks.append(track)
            track.detection_history.append(associations[track].detection)
            track.confusion_matrix = self.create_confusion_matrix(track)

    def create_confusion_matrix(self,track):




