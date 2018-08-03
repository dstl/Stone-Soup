import numpy as np

from .base import MetricGenerator
from ..base import Property
from ..types import Clutter, TrueDetection, Metric, SingleTimeMetric, TimePeriodMetric


class SingleDetectionBasedMetrics(MetricGenerator):
    # Will break if tracks are based on anything other than a single detection

    # Assumes tracks is a list of one track per every object tracked and can contain tracks on objects that no longer exist

    generators = Property(list, doc='List of generators to use')

    def generate_metrics(self, tracks, groundtruth_paths, detections):
        """

        :param tracks: set of Track objects created by the tracker
        :param groundtruth_paths:  set of GroundTruthPath objects
        :param detections: set of list of Detection objects
        :return: set of Metric objects (or objects that inherit from metric)
        """

        metrics = set()

        for generator in self.generators:
            metrics.add(generator.compute_metric(tracks, groundtruth_paths, detections))

        return metrics

class BasicMetrics(MetricGenerator):
    """Calculated simeple metrics like number of tracks, truth and """
    def compute_metric(self, tracks, groundtruth_paths, detections):

        metrics = []

        # Make a list of all the unique timestamps used
        # THIS IS REPEATED IN VARIOUS PLACES SO SHOULD PROBABLY BE DONE BETTER
        timestamps = []
        for track in tracks:
            for state in track.states:
                if state.timestamp not in timestamps:
                    timestamps.append(state.timestamp)
        for path in groundtruth_paths:
            for state in path:
                if state.timestamp not in timestamps:
                    timestamps.append(state.timestamp)

        # Number of tracks
        metrics.append(TimePeriodMetric(
                    title='Number of targets',
                    value=len(groundtruth_paths),
                    start_timestamp=min(timestamps),
                    end_timestamp=max(timestamps),
                    generator=self))

        metrics.append(TimePeriodMetric(
            title='Number of tracks',
            value=len(tracks),
            start_timestamp=min(timestamps),
            end_timestamp=max(timestamps),
            generator=self))

        metrics.append(TimePeriodMetric(
            title='Track-to-target ratio',
            value=len(tracks)/len(groundtruth_paths),
            start_timestamp=min(timestamps),
            end_timestamp=max(timestamps),
            generator=self))

        return metrics






class OSPAMetric(MetricGenerator):
    c = Property(float, doc='Maximum distance for possible association')
    p = Property(float, doc='norm associated to distance')
    measurement_matrix = Property(np.ndarray, doc='Measurement matrix to extract parameters to calculate distance over')

    def compute_metric(self, tracks, groundtruth_paths, detections):
        """
        Compute the OSPA metric at every timestep
        :param tracks:
        :param groundtruth_paths:
        :param detections:
        :return:
        """

        # Make a list of all the unique timestamps used
        timestamps = []
        for track in tracks:
            for state in track.states:
                if state.timestamp not in timestamps:
                    timestamps.append(state.timestamp)
        for path in groundtruth_paths:
            for state in path:
                if state.timestamp not in timestamps:
                    timestamps.append(state.timestamp)

        ospa_distances = []

        for timestamp in timestamps:
            track_points = [i[0] for i in [[s for s in x.states if s.timestamp == timestamp] for x in tracks] if
                            i != []]

            truth_points = [i[0] for i in [[s for s in x.states if s.timestamp == timestamp] for x in groundtruth_paths]
                            if i != []]

            ospa_distances.append(self.compute_OSPA_distance(track_points, truth_points, timestamp))

        return TimePeriodMetric(title='OSPA distances',
                                value=ospa_distances,
                                start_timestamp=min(timestamps),
                                end_timestamp=max(timestamps),
                                generator=self)

    def compute_OSPA_distance(self, track_states, truth_states, tstamp=None):

        '''

        :param track_states: list of state objects to be assigned to the truth
        :param truth_states: list of state objects for the truth points
        :param timestamp: timestamp at which the states occured. If none then selected from the list of ststes
        :return:
        '''

        from scipy.optimize import linear_sum_assignment

        if not tstamp:
            tstamp = track_states[0].timestamp
        for st in track_states + truth_states:
            if st.timestamp != tstamp:
                raise ValueError('All states must be from the same time to perform OSPA')

        cost_matrix = self.compute_cost_matrix(track_states, truth_states)

        # Solve cost matrix with Hungarian/Munkres using scipy.optimize.linear_sum_assignemnt
        n = max([len(track_states), len(truth_states)])  # Length of longest set of states
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # Calculate metric following Vo's paper or python code online.
        distance = ((1 / n) * cost_matrix[row_ind, col_ind].sum()) ** (1 / self.p)
        return SingleTimeMetric(title='OSPA distance', value=distance, timestamp=tstamp, generator=self)

    def compute_cost_matrix(self, track_states, truth_states):

        cost_matrix = np.ones([len(track_states), len(truth_states)]) * self.c

        for i_track, track_state, in enumerate(track_states):
            for i_truth, truth_state in enumerate(truth_states):

                euc_distance = np.linalg.norm(
                    self.measurement_matrix.matrix() @ track_state.state_vector.__array__() - self.measurement_matrix.matrix() @ truth_state.state_vector.__array__())

                if euc_distance < self.c:
                    cost_matrix[i_track, i_truth] = euc_distance

        return cost_matrix
