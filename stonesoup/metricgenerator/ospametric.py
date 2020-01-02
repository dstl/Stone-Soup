from itertools import chain

import numpy as np
from scipy.optimize import linear_sum_assignment

from .base import MetricGenerator
from ..base import Property
from ..models.measurement import MeasurementModel
from ..types.state import State, StateMutableSequence
from ..types.time import TimeRange
from ..types.metric import SingleTimeMetric, TimeRangeMetric


class GOSPAMetric(MetricGenerator):
    """
    Computes the Generalized Optimal SubPattern Assignment (GOPSA) metric
    for two sets of :class:`~.Track` objects. This implementation of GOSPA
    is based on the auction algorithm.

    The GOPSA metric is calculated at each time step in which a
    :class:`~.Track` object is present

    Reference:
        [1] A. S. Rahmathullah, A. F. García-Fernández, L. Svensson,
        Generalized optimal sub-pattern assignment metric, 2016,
        [online] Available: http://arxiv.org/abs/1601.05585.
    """
    p = Property(float, doc="1<=p<infty, exponent.")
    c = Property(float, doc="c>0, cutoff distance.")
    measurement_model_truth = Property(
        MeasurementModel,
        doc="Measurement model which specifies which elements within the"
            "truth state are to be used to calculate distance over")
    measurement_model_track = Property(
        MeasurementModel,
        doc="Measurement model which specifies which elements within"
            "the truth state are to be used to calculate distance over")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = 2

    def compute_metric(self, manager):
        """Compute the metric using the data in the metric manager

        Parameters
        ----------
        manager : :class:`~.MetricManager`
            contains the data to be used to create the metric(s)

        Returns
        -------
        metric : list :class:`~.Metric`
            Containing the metric information. The value of the metric is a
            list of metrics at each timestamp

        """

        metric = self.process_datasets(manager.tracks,
                                       manager.groundtruth_paths)
        return metric

    def process_datasets(self, dataset_1, dataset_2):
        """Process a dataset of point patterns to provide metric over time

        Parameters
        ----------
        dataset_1: object containing :class:`~.state`
        dataset_2: object containing :class:`~.state`

        Returns
        -------
        metrics: :class:`~.Metric`
            Containing the metric at each timestamp in the form of a list of
            :class:`~.Metric` objects
        """

        states_1 = self.extract_states(dataset_1)
        states_2 = self.extract_states(dataset_2)
        return self.compute_over_time(states_1, states_2)

    @staticmethod
    def extract_states(object_with_states):
        """
        Extracts a list of :class:`~states` from a list of (or single) objects
        containing states. This method is defined to handle :class:`~track`,
        :class:`~groundtruthpath` and :class:`~detection` objects

        Parameters
        ----------
        object_with_states: object containing a list of states
            Method of state extraction depends on the type of the object

        Returns
        -------
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
        """
        Compute the GOSPA metric at every timestep from a list of measured
        states and truth states.

        Parameters
        ----------

        measured_states: List of states created by a filter
        truth_states: List of truth states to compare against

        Returns
        -------
        metric: :class:`~.TimeRangeMetric` covering the duration that states
        exist for in the parameters. metric.value contains a list of metrics
        for the GOSPA metric at each timestamp
        """

        # Make a list of all the unique timestamps used
        # Make a sorted list of all the unique timestamps used
        timestamps = sorted({
            state.timestamp
            for state in chain(measured_states, truth_states)})

        gospa_metrics = []

        for timestamp in timestamps:
            meas_points = [state
                           for state in measured_states
                           if state.timestamp == timestamp]
            truth_points = [state
                            for state in truth_states
                            if state.timestamp == timestamp]
            metric, truth_to_measured_assignment = self.compute_gospa_metric(
                    meas_points, truth_points)
            gospa_metrics.append(metric)

        # If only one timestamp is present then return a SingleTimeMetric
        if len(timestamps) == 1:
            return gospa_metrics[0]
        else:
            return TimeRangeMetric(
                title='GOSPA Metrics',
                value=gospa_metrics,
                time_range=TimeRange(min(timestamps), max(timestamps)),
                generator=self)

    def compute_assignments(self, cost_matrix, max_iter):
        """Compute assignments using Auction Algorithm.

        Parameters
        ----------
        cost_matrix: Matrix (size mxn) that denotes the cost of assigning
                            mth truth state to each of the n measured states.
        max_iter: Maximum number of iterations to perform

        Returns
        ---------
        truth_to_measured: Vector of size 1xm, which has indices of the
                    measured objects or '-1' if unassigned.
        measured_to_truth: Vector of size 1xn, which has indices of the
                            truth objects or '-1' if unassigned.
        opt_cost: Scalar value of the optimal assignment
        """

        m_truth, n_measured = cost_matrix.shape
        # Index for objects that will be left un-assigned.
        unassigned_idx = -1

        opt_cost = 0.0
        measured_to_truth = -1 * np.ones([1, n_measured], dtype=np.int64)
        truth_to_measured = -1 * np.ones([1, m_truth], dtype=np.int64)

        if m_truth == 1:
            # Corner case 1: if there is only one truth state.
            opt_cost = np.max(cost_matrix)
            max_cost_idx = np.where(cost_matrix == opt_cost)[1]
            truth_to_measured[0, 0] = max_cost_idx[0]
            measured_to_truth[0, truth_to_measured[0, 0]] = 1

            return truth_to_measured, measured_to_truth, opt_cost

        if n_measured == 1:
            # Corner case 1: if there is only one measured state.
            opt_cost = np.max(cost_matrix)
            max_cost_idx = np.where(cost_matrix == opt_cost)[1]
            measured_to_truth[0, 0] = max_cost_idx[0]
            truth_to_measured[0, measured_to_truth[0, 0]] = 1

            return truth_to_measured, measured_to_truth, opt_cost

        swap_dim_flag = False
        epsil = 1. / np.max([m_truth, n_measured])

        if n_measured < m_truth:
            # The implementation only works when
            # m_truth <= n_measured
            # So swap cost matrix
            cost_matrix = cost_matrix.transpose()
            m_truth, n_measured = cost_matrix.shape
            tmp = measured_to_truth
            measured_to_truth = truth_to_measured
            truth_to_measured = tmp
            swap_dim_flag = True

        # Initial cost for each measured state
        c_measured = np.zeros([1, n_measured])
        k_iter = 0

        while not np.all(truth_to_measured != unassigned_idx):
            if k_iter > max_iter:
                # Raise max iterations reached warning.
                break
            for i in range(m_truth):
                if truth_to_measured[0, i] == unassigned_idx:
                    # Unassigned truth object 'i' bids for the best
                    # measured object j_star

                    # Value for each measured object for truth 'i'
                    tmp_mat = cost_matrix[i, :] - c_measured
                    val_i_j = np.sort(tmp_mat[0, :])[::-1]
                    j = np.argsort(tmp_mat[0, :])[::-1]

                    # Best measurement for truth 'i'
                    j_star = j[0]

                    # 1st and 2nd best value for truth 'i'
                    v_i_j_star = val_i_j[0]
                    w_i_j_star = val_i_j[1]
                    # Bid for measured j_star
                    if w_i_j_star != (-1. * np.inf):
                        c_measured[0, j_star] = c_measured[
                            0, j_star] + v_i_j_star - w_i_j_star + epsil
                    else:
                        c_measured[0, j_star] = c_measured[
                            0, j_star] + v_i_j_star + epsil

                    # If j_star is unassigned
                    if measured_to_truth[0, j_star] != unassigned_idx:

                        opt_cost = opt_cost - \
                            cost_matrix[measured_to_truth[0, j_star], j_star]
                        truth_to_measured[0, measured_to_truth[
                            0, j_star]] = unassigned_idx

                    measured_to_truth[0, j_star] = i
                    truth_to_measured[0, i] = j_star

                    # update the cost of new assignment
                    opt_cost = opt_cost + cost_matrix[i, j_star]
            k_iter += 1

        if swap_dim_flag:
            tmp = measured_to_truth
            measured_to_truth = truth_to_measured
            truth_to_measured = tmp

        return truth_to_measured, measured_to_truth, opt_cost

    def compute_cost_matrix(self, track_states, truth_states):
        """
        Creates the cost matrix between two lists of states

        Parameters
        ----------
        track_states: list of states
        truth_states: list of states

        Returns
        ----------
        cost_matrix: Matrix of euclidian distance between each element in each
        list of states
        """

        cost_matrix = np.ones([len(track_states), len(truth_states)]) * self.c

        for i_track, track_state, in enumerate(track_states):
            for i_truth, truth_state in enumerate(truth_states):

                euc_distance = np.linalg.norm(
                    self.measurement_model_track.function(
                        track_state, noise=0)
                    - self.measurement_model_truth.function(
                        truth_state, noise=0))

                if euc_distance < self.c:
                    cost_matrix[i_track, i_truth] = euc_distance

        return cost_matrix

    def compute_gospa_metric(self, measured_states, truth_states):
        """Computes GOSPA metric between measured and truth states.

        Parameters
        ----------
        measured_states: list of :class:`~.State`
            list of state objects to be assigned to the truth
        truth_states: list of :class:`~.State`
            list of state objects for the truth points

        Returns
        -------
        gospa_metric: Dictionary containing GOSPA metric for alpha = 2.
                      GOSPA metric is divided into four components:
                      1. distance, 2. localisation, 3. missed, and 4. false.
                      Note that distance = (localisation + missed + false)^1/p
        truth_to_measured_assignment: Assignment matrix.
        """
        timestamps = {
            state.timestamp
            for state in chain(truth_states, measured_states)}
        if len(timestamps) != 1:
            raise ValueError(
                'All states must be from the same time to compute GOSPA')

        gospa_metric = {'distance': 0.0,
                        'localisation': 0.0,
                        'missed': 0,
                        'false': 0}
        num_truth_states = len(truth_states)
        num_measured_states = len(measured_states)
        truth_to_measured_assignment = []
        cost_matrix = self.compute_cost_matrix(measured_states, truth_states)
        cost_matrix = cost_matrix.transpose()

        opt_cost = 0.0
        dummy_cost = (self.c ** self.p) / self.alpha

        if num_truth_states == 0:
            # When truth states are empty all measured states are false
            opt_cost = -1.0 * num_measured_states * dummy_cost
            # num_false = opt_cost
        else:
            if num_measured_states == 0:
                # When measured states are empty all truth
                # states are missed
                opt_cost = -1. * num_truth_states * dummy_cost
                if self.alpha == 2:
                    self.missed = opt_cost
            else:
                # Use auction algorithm when both truth_states
                # and measured_states are non-empty
                cost_matrix = -1. * np.power(cost_matrix, self.p)
                truth_to_measured_assignment, measured_to_truth_assignment,\
                    opt_cost_tmp =\
                    self.compute_assignments(cost_matrix,
                                             10 * num_truth_states *
                                             num_measured_states)
                # Now use assignments to compute bids
                for i in range(num_truth_states):
                    if truth_to_measured_assignment[0, i] != -1:
                        opt_cost = opt_cost +\
                            cost_matrix[i, truth_to_measured_assignment[0, i]]

                        if self.alpha == 2:
                            const_assign = \
                                truth_to_measured_assignment[0, i]
                            const_cmp = (-1 * self.c**self.p)

                            gospa_metric['localisation'] = \
                                gospa_metric['localisation'] +\
                                cost_matrix[i, const_assign] *\
                                np.double(
                                    cost_matrix[i, const_assign] > const_cmp)

                            gospa_metric['missed'] = gospa_metric['missed'] -\
                                dummy_cost * np.double(
                                    cost_matrix[i, const_assign] == const_cmp)

                            gospa_metric['false'] = gospa_metric['false'] -\
                                dummy_cost * np.double(
                                    cost_matrix[i, const_assign] == const_cmp)
                    else:
                        opt_cost = opt_cost - dummy_cost
                        if self.alpha == 2:
                            gospa_metric['missed'] = gospa_metric[
                                'missed'] - dummy_cost

                opt_cost = opt_cost - \
                    np.sum(measured_to_truth_assignment == -1) * dummy_cost
                if self.alpha == 2:
                    gospa_metric['false'] = gospa_metric['false'] - \
                        np.sum(measured_to_truth_assignment == -1) * dummy_cost
        gospa_metric['distance'] = np.power((-1. * opt_cost), 1 / self.p)
        gospa_metric['localisation'] = -1. * gospa_metric['localisation']
        gospa_metric['missed'] = -1 * gospa_metric['missed']
        gospa_metric['false'] = -1 * gospa_metric['false']

        single_time_gospa_metric = SingleTimeMetric(
                title='GOSPA Metric', value=gospa_metric,
                timestamp=timestamps.pop(), generator=self)

        return single_time_gospa_metric, truth_to_measured_assignment


class OSPAMetric(GOSPAMetric):
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

    c = Property(float, doc='Maximum distance for possible association')
    p = Property(float, doc='norm associated to distance')

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

        The function :math:`\bar{d}_{p}^{(c)}` is the OSPA metric of order
        :math:`p` with cut-off :math:`c`. The OSPA metric is defined as:

            .. math::
                \begin{equation*}
                    \bar{d}_{p}^{(c)}({X},{Y}) :=
                    \Biggl( \frac{1}{n}
                    \Bigl({min}_{\substack{
                        \pi\in\Pi_{n}}}
                            \sum_{i=1}^{m}
                                d^{(c)}(x_{i},y_{\pi(i)})^{p}+
                                c^{p}(n-m)\Bigr)
                        \Biggr)^{ \frac{1}{p} }
                \end{equation*}

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
