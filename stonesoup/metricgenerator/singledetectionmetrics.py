from .base import MetricGenerator
from ..base import Property
from ..types import Clutter


class SingleDetectionBasedMetrics(MetricGenerator):
    # Will break if tracks are based on anything other than a single detection

    # All functions assume that each track is the best guess for that object. If a track is deleted it's because that
    # object has ceased to exist. If a track is tracked by multiple hypothesies then only the current best guess is ever
    # returned. As the detections are compared with true detections in real time, any change in detections used in the
    # past will not be accounted for unless as the result of merging (where the surviving track will keep its history)

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
        # Idea is to keep a record of the detections used in each track, probably by keeping a parallel list of tracks
        # with extra information being stored. There's probably some magic python way of doing this better with pointers
        # or something.

        '''Update the list of detections for each track with the new detection used'''
        for track in self.tracks:
            self.track.confusion_matrix = self.create_confusion_matrix(track,detections)

    def create_confusion_matrix(self,track):





