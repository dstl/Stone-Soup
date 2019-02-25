from itertools import chain

import numpy as np
from scipy.optimize import linear_sum_assignment

from .base import MetricGenerator
from ..base import Property
from ..models.measurement import MeasurementModel
from ..types.state import State, StateMutableSequence
from ..types.time import TimeRange
from ..types.metric import SingleTimeMetric, TimeRangeMetric


class OSPAMetric(MetricGenerator):
    """
    Computes the Optimal SubPattern Assignment (OPSA) distance [1] for two sets
    of :class:`~.Track` objects. The OSPA distance is measured between two
    point patterns.

    The OPSA metric is calculated at each time step in which a :class:`~.Track`
    object is present

    Reference:
        [1] A Consistent Metric for Performance Evaluation of Multi-Object
        Filters, D. Schuhmacher, B. Vo and B. Vo, IEEE Trans. Signal Processing
        2008
    """
    c = Property(float, doc="Maximum distance for possible association")
    p = Property(float, doc="norm associated to distance")
    measurement_model_truth = Property(
        MeasurementModel,
        doc="Measurement model which specifies which elements within the"
            "truth state are to be used to calculate distance over")
    measurement_model_track = Property(
        MeasurementModel,
        doc="Measurement model which specifies which elements within"
            "the truth state are to be used to calculate distance over")

    def compute_metric(self, manager):
        """
        Compute the metric using the data in the metric manager

        Parameters
        ----------
        manager : MetricManager
            containing the data to be used to create the metric(s)

        Returns
        -------
        : list of :class:`~.Metric`
            Containing the metric information. The value of the metric is a
            list of metrics at each timestamp

        """

        metric = self.process_datasets(manager.tracks,
                                       manager.groundtruth_paths)
        return metric

    def process_datasets(self, dataset_1, dataset_2):
        """
        Process a dataset of point patterns to provide OPSA distances over time

        Parameters
        ----------
        dataset_1: object containing :class:`~.state`
        dataset_2: object containing :class:`~.state`

        Returns
        -------
        : list of :class:`~.Metric`
            Contains the OSPA distance at each timestamp
        """

        states_1 = self.extract_states(dataset_1)
        states_2 = self.extract_states(dataset_2)
        return self.compute_over_time(states_1, states_2)

    def extract_states(self, object_with_states):
        """
        Extracts a list of :class:`~states` from a list of (or single) objects
        containing states. This method is defined to handle :class:`~track`,
        :class:`~groundtruthpath` and :class:`~detection` objects

        Parameters
        ----------
        object_with_states: object containing a list of states
            Method of state extraction depends on the type of the object

        Returns
        ----------
        : list of :class:`~.State`
        """
        state_list = StateMutableSequence()
        for element in list(object_with_states):
            if isinstance(element, StateMutableSequence):
                state_list.extend(element.states)
            elif isinstance(element, State):
                state_list.append(element)
            else:
                raise ValueError(
                    "{!r} has no state extraction method".format(element))

        return state_list

    def compute_over_time(self, measured_states, truth_states):
        """Compute the OSPA metric at every timestep from a list of measured
        states and truth states

        Parameters
        ----------
        measured_states: list of :class:`~.State`
            Created by a filter
        truth_states: list of :class:`~.State`
            Truth states to compare against

        Returns
        -------
        TimeRangeMetric
            Covering the duration that states exist for in the parameters.
            Metric.value contains a list of metrics for the OSPA distance at
            each timestamp
        """

        # Make a sorted list of all the unique timestamps used
        timestamps = sorted({
            state.timestamp
            for state in chain(measured_states, truth_states)})

        ospa_distances = []

        for timestamp in timestamps:
            meas_points = [state
                           for state in measured_states
                           if state.timestamp == timestamp]
            truth_points = [state
                            for state in truth_states
                            if state.timestamp == timestamp]
            ospa_distances.append(
                self.compute_OSPA_distance(meas_points, truth_points))

        # If only one timestamp is present then return a SingleTimeMetric
        if len(timestamps) == 1:
            return ospa_distances[0]
        else:
            return TimeRangeMetric(
                title='OSPA distances',
                value=ospa_distances,
                time_range=TimeRange(min(timestamps), max(timestamps)),
                generator=self)

    def compute_OSPA_distance(self, track_states, truth_states):
        r"""
        Computes the Optimal SubPattern Assignment (OPSA) metric for a single
        time step between two point patterns. Each point pattern consisting of
        a list of :class:`~.State` objects.

        The OSPA metric is defined as:

        .. math::
            \overline{D}_{p}^{(c)}({X},{Y}) = (frac{1}{n}(\sum))^{p}

        Parameters
        ----------
        track_states: list of :class:`~.State`
        truth_states: list of :class:`~.State`

        Returns
        -------
        SingleTimeMetric
            The OSPA distance

        """

        timestamps = {
            state.timestamp
            for state in chain(truth_states, track_states)}
        if len(timestamps) != 1:
            raise ValueError(
                'All states must be from the same time to perform OSPA')

        if not track_states or not truth_states:
            distance = 0
        else:
            cost_matrix = self.compute_cost_matrix(track_states, truth_states)

            # Solve cost matrix with Hungarian/Munkres using
            # scipy.optimize.linear_sum_assignemnt
            # Length of longest set of states
            n = max(len(track_states), len(truth_states))
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            # Calculate metric
            distance = ((1 / n) * cost_matrix[row_ind, col_ind].sum()) ** (
                        1 / self.p)

        return SingleTimeMetric(title='OSPA distance', value=distance,
                                timestamp=timestamps.pop(), generator=self)

    def compute_cost_matrix(self, track_states, truth_states):
        """
        Creates the cost matrix between two lists of states

        Parameters
        ----------
        track_states: list of :class:`State`
        truth_states: list of :class:`State`

        Returns
        ----------
        np.ndarry
            Matrix of euclidian distance between each element in each list of
            states
        """

        cost_matrix = np.ones([len(track_states), len(truth_states)]) * self.c

        for i_track, track_state, in enumerate(track_states):
            for i_truth, truth_state in enumerate(truth_states):
                euc_distance = np.linalg.norm(
                    self.measurement_model_track.function(
                        track_state.state_vector, noise=0)
                    - self.measurement_model_truth.function(
                        truth_state.state_vector, noise=0))

                if euc_distance < self.c:
                    cost_matrix[i_track, i_truth] = euc_distance

        return cost_matrix
