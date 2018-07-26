import numpy as np

from .base import MetricGenerator
from ..base import Property
from ..types import Clutter, TrueDetection, Metric


class SingleDetectionBasedMetrics(MetricGenerator):
    # Will break if tracks are based on anything other than a single detection

    # All functions assume that each track is the best guess for that object. If a track is deleted it's because that
    # object has ceased to exist. If a track is tracked by multiple hypothesies then only the current best guess is ever
    # returned. As the detections are compared with true detections in real time, any change in detections used in the
    # past will not be accounted for unless as the result of merging (where the surviving track will keep its history)

    # Detections are split between clutter and not clutter by using isinstance(detection, Clutter) or isinstance(detection, TrueDetection)

    # Output is a dictionary with keys equal to the metric names like 'confusion_matrix', 'track_completeness'...

    self.track_on_truth_duration = 3 # Length that a track needs to be near the same truth to be counted as 'tracking it'
    self.spatial_distance = 10
    self.temporal_distance = 0

    def generate_metrics(self, tracks, truth, detections):

        metrics = {} # A dict
        metrics['number_of_targets'] = len(truth)
        metrics['number_of_tracks'] = len(tracks)
        metrics['track_to_target_ratio'] = len(tracks)/len(truth)

        track_truth_associations = self.associate_tracks_to_truth(tracks,truth)

    def associate_tracks_to_truth(self,tracks,truth):

        # Some sort of magic association method that matches tracks to truths

        for track in tracks:
            for state in track.states:

                # Find the ground truth tracks that exist at the same time as this state
                potential_gtruths = [t for t in truth if state.timestamp in [a.timestamp for a in t.states]]

                '''Hardcoded fudge, make a better version in the future'''
                track_truth_distance, truth = [np.sqrt((state[0]-t.s)),t for ]


        return track_truth_associations

    def associate_truths_to_track(self, track):


    def associate_detection



class OSPAMetric(Metric):

    c = Property(float, doc='Maximum distance for possible association')
    measurement_matrix = Property(np.ndarray, doc='Measurement matrix to extract parameters to calculate distance over')

    def compute_metric(self, track_states, truth_states):

        '''

        :param track_states: list of state objects to be assigned to the truth
        :param truth_states: list of state objects for the truth points
        :return:
        '''

        cost_matrix = self.compute_cost_matrix(track_states,truth_states)

        # Solve cost matrix with Hungarian/Munkres using scipy.optimize.linear_sum_assignemnt

        #Calculate metric following Vo's paper or python code online.


    def compute_cost_matrix(self,track_states,truth_states):

        cost_matrix = np.ones([len(track_states),len(truth_states)]) * self.c

        for track_state, i_track in enumerate(track_states):
            for truth_state, i_truth in enumerate(truth_states):

                euc_distance = np.linalg.norm(self.measurement_matrix @ track_state- self. measurement_matrix @ truth_state)

                if euc_distance < self.c:
                    cost_matrix[i_track,i_truth] = euc_distance

        return cost_matrix






