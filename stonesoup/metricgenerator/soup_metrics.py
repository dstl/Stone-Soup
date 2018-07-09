from .base import MetricGenerator
from ..base import Property
from ..types import Clutter


class SingleDetectionBasedMetrics(MetricGenerator):
    # Will break if tracks are based on anything other than a single detection

    def create_metrics(self, tracks, truth, detections, associations):

        metrics = {} # A dict
        metrics['number_of_targets'] = len(truth)
        metrics['number_of_tracks'] = len(tracks)
        metrics['track_to_target_ratio']

        # Insert code here
# Detections are split between clutter and not clutter by using isinstance(detection, Clutter)

        # Output is a dictionary with keys equal to the metric names like 'confusion_matrix', 'track_completeness'...
        return metrics

    def update_confusion_matrix(self, ):
        # Live metric to update the confusion matrix after each update.


