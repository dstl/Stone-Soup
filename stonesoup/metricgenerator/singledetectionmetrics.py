import numpy as np

from .base import MetricGenerator, MetricManager
from ..base import Property
from ..types import Clutter, TrueDetection, Metric, SingleTimeMetric, TimePeriodMetric, Detection, GroundTruthPath, Track




class BasicMetrics(MetricGenerator):
    """Calculated simeple metrics like number of tracks, truth and """
    def compute_metric(self, manager):

        metrics = []

        # Make a list of all the unique timestamps used
        timestamps = []
        for track in manager.tracks:
            for state in track.states:
                if state.timestamp not in timestamps:
                    timestamps.append(state.timestamp)
        for path in manager.groundtruth_paths:
            for state in path:
                if state.timestamp not in timestamps:
                    timestamps.append(state.timestamp)

        # Number of tracks
        metrics.append(TimePeriodMetric(
                    title='Number of targets',
                    value=len(manager.groundtruth_paths),
                    start_timestamp=min(timestamps),
                    end_timestamp=max(timestamps),
                    generator=self))

        metrics.append(TimePeriodMetric(
            title='Number of tracks',
            value=len(manager.tracks),
            start_timestamp=min(timestamps),
            end_timestamp=max(timestamps),
            generator=self))

        metrics.append(TimePeriodMetric(
            title='Track-to-target ratio',
            value=len(manager.tracks)/len(manager.groundtruth_paths),
            start_timestamp=min(timestamps),
            end_timestamp=max(timestamps),
            generator=self))

        return metrics


class OSPAMetric(MetricGenerator):

    c = Property(float, doc='Maximum distance for possible association')
    p = Property(float, doc='norm associated to distance')
    measurement_matrix_truth = Property(np.ndarray, doc='Measurement matrix for the truth states to extract parameters to calculate distance over')
    measurement_matrix_meas = Property(np.ndarray, doc='Measurement matrix for the track states to extract parameters to calculate distance over')

    def compute_metric(self, manager):

        metric = self.process_datasets(manager.tracks, manager.groundtruth_paths)
        return metric

    def process_datasets(self, dataset_1, dataset_2):

        states_1 = self.extract_states(dataset_1)
        states_2 = self.extract_states(dataset_2)
        return self.compute_over_time(states_1, states_2)

    def extract_states(self, object_with_states):
        """
        Extracts a list of states from a list of (or single) object containing states
        :param object_with_states:
        :return:
        """

        state_list = []
        for element in list(object_with_states):

            if isinstance(element, Track):
                for state in element.states:
                    state_list.append(state)

            elif isinstance(element, GroundTruthPath):
                for state in element.states:
                    state_list.append(state)

            elif isinstance(element, Detection):
                state_list.append(element)

            else:
                raise ValueError(type(element), ' has no state extraction method')

        return state_list

    def compute_over_time(self, measured_states, truth_states):
        """
        Compute the OSPA metric at every timestep from a list of measured states and truth states
        :param measured_states: List of states created by a filter
        :param truth_states: List of truth states to compare against
        :return:
        """

        # Make a list of all the unique timestamps used
        timestamps = []
        for state in measured_states + truth_states:
            if state.timestamp not in timestamps:
                timestamps.append(state.timestamp)

        ospa_distances = []

        for timestamp in timestamps:
            meas_points = [state for state in measured_states if state.timestamp == timestamp]

            truth_points = [state for state in truth_states if state.timestamp == timestamp]

            ospa_distances.append(self.compute_OSPA_distance(meas_points, truth_points, timestamp))

        # If only one timestamp is present then return a SingleTimeMetric
        if len(timestamps) == 1:
            return ospa_distances[0]
        else:
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

        if len(track_states) == 0 or len(truth_states) == 0:
            distance = 0
        else:

            cost_matrix = self.compute_cost_matrix(track_states, truth_states)

            # Solve cost matrix with Hungarian/Munkres using scipy.optimize.linear_sum_assignemnt
            n = max([len(track_states), len(truth_states)])  # Length of longest set of states

            # If there are either no tracks or no truth to compare then the distance is zero

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            # Calculate metric following Vo's paper or python code online.
            distance = ((1 / n) * cost_matrix[row_ind, col_ind].sum()) ** (1 / self.p)

        return SingleTimeMetric(title='OSPA distance', value=distance, timestamp=tstamp, generator=self)

    def compute_cost_matrix(self, track_states, truth_states):

        cost_matrix = np.ones([len(track_states), len(truth_states)]) * self.c

        for i_track, track_state, in enumerate(track_states):
            for i_truth, truth_state in enumerate(truth_states):

                euc_distance = np.linalg.norm(
                    self.measurement_matrix_meas @ track_state.state_vector.__array__() - self.measurement_matrix_truth @ truth_state.state_vector.__array__())

                if euc_distance < self.c:
                    cost_matrix[i_track, i_truth] = euc_distance

        return cost_matrix
