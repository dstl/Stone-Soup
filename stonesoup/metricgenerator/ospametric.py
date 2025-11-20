from itertools import chain, groupby, zip_longest

import numpy as np
from scipy.optimize import linear_sum_assignment

from .base import MetricGenerator
from ..base import Property
from ..measures import Measure, Euclidean
from ..types.state import State, StateMutableSequence
from ..types.time import TimeRange
from ..types.metric import SingleTimeMetric, TimeRangeMetric


class _SwitchingLoss:
    """
    Holds state assignment history and computes GOSPA switching term
    See https://www.mathworks.com/help/fusion/ref/trackgospametric-system-object.html#d126e213697
    """
    def __init__(self, loss_factor, p):
        self.truth_associations = {}
        self.switching_loss = None
        self.loss_factor = loss_factor
        self.p = p
        self.switching_penalty = 0.5

    def add_associations(self, truth_associations):
        """
        Add a new set of association and update the switching loss.

        Parameters
        ----------
        truth_associations: dict(truth_track_id: measurement_track_id)
        """
        self.switching_loss = 0

        for truth_id, meas_id in truth_associations.items():
            if truth_id not in self.truth_associations and meas_id is None:
                continue
            if truth_id not in self.truth_associations:
                self.truth_associations[truth_id] = meas_id
            elif self.truth_associations[truth_id] != meas_id:
                self.switching_loss += self.switching_penalty
                if meas_id is not None and self.truth_associations[truth_id] is not None:
                    self.switching_loss += self.switching_penalty

                self.truth_associations[truth_id] = meas_id

    def loss(self):
        """Compute loss based on last association."""
        if self.switching_loss is None:
            raise RuntimeError("Can't compute switching loss before any association are added.")
        return self.loss_factor * self.switching_loss**(1/self.p)


class GOSPAMetric(MetricGenerator):
    """
    Computes the Generalized Optimal SubPattern Assignment (GOSPA) metric
    for two sets of :class:`~.Track` objects. This implementation of GOSPA
    is based on the modified Jonker-Volgenant algorithm.

    The GOSPA metric is calculated at each time step in which a
    :class:`~.Track` object is present

    Reference:
        [1] A. S. Rahmathullah, A. F. García-Fernández, L. Svensson,
        Generalized optimal sub-pattern assignment metric, 2016,
        [online] Available: http://arxiv.org/abs/1601.05585.
    """
    p: float = Property(doc="1<=p<infty, exponent.")
    c: float = Property(doc="c>0, cutoff distance.")
    switching_penalty: float = Property(doc="Penalty term for switching.", default=0.0)
    measure: Measure = Property(
        default_factory=Euclidean,
        doc="Distance measure to use. Default :class:`~.measures.Euclidean()`")
    generator_name: str = Property(doc="Unique identifier to use when accessing generated metrics "
                                       "from MultiManager",
                                   default='gospa_generator')
    tracks_key: str = Property(doc='Key to access set of tracks added to MetricManager',
                               default='tracks')
    truths_key: str = Property(doc="Key to access set of ground truths added to MetricManager. "
                                   "Or key to access a second set of tracks for track-to-track"
                                   " metric generation",
                               default='groundtruth_paths')

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
        return self.compute_over_time(
            *self.extract_states(manager.states_sets[self.tracks_key], True),
            *self.extract_states(manager.states_sets[self.truths_key], True)
        )

    @staticmethod
    def extract_states(object_with_states, return_ids=False):
        """
        Extracts a list of states from a list of (or single) objects
        containing states. This method is defined to handle :class:`~.StateMutableSequence`
        and :class:`~.State` types.

        Parameters
        ----------
        object_with_states: object containing a list of states
            Method of state extraction depends on the type of the object
        return_ids: If we should return obj ids as well.

        Returns
        -------
        : list of :class:`~.State`
        """

        state_list = StateMutableSequence()
        ids = []
        for i, element in enumerate(list(object_with_states)):
            if isinstance(element, StateMutableSequence) and not isinstance(element, State):
                states = list(element.last_timestamp_generator())
                state_list.extend(states)
                ids.extend([i]*len(states))
            elif isinstance(element, State):
                state_list.append(element)
                ids.extend([i])
            else:
                raise ValueError(
                    "{!r} has no state extraction method".format(element))
        if return_ids:
            return state_list, ids
        return state_list

    def compute_over_time(self, measured_states, measured_state_ids, truth_states,
                          truth_state_ids):
        """
        Compute the GOSPA metric at every timestep from a list of measured
        states and truth states.

        Parameters
        ----------

        measured_states: List of states created by a filter
        measured_state_ids: ids for which state belongs in
        truth_states: List of truth states to compare against
        truth_state_ids: ids for which truth state belongs in

        Returns
        -------
        metric: :class:`~.TimeRangeMetric` covering the duration that states
        exist for in the parameters. metric.value contains a list of metrics
        for the GOSPA metric at each timestamp
        """
        all_meas_timestamps = np.fromiter(
            (state.timestamp for state in measured_states),
            dtype='O'
        )
        meas_order = np.argsort(all_meas_timestamps)
        all_meas_timestamps = all_meas_timestamps[meas_order]
        # np.array doesn't work for ParticleState
        all_meas_points = np.empty(len(measured_states), dtype="O")
        all_meas_points[:] = measured_states
        all_meas_points = all_meas_points[meas_order]
        all_meas_ids = np.array(measured_state_ids)[meas_order]

        all_truth_timestamps = np.fromiter(
            (state.timestamp for state in truth_states),
            dtype='O'
        )
        truth_order = np.argsort(all_truth_timestamps)
        all_truth_timestamps = all_truth_timestamps[truth_order]
        all_truth_points = np.array(truth_states)[truth_order]
        all_truth_ids = np.array(truth_state_ids)[truth_order]

        truth_iter = iter(groupby(range(len(all_truth_ids)), all_truth_timestamps.__getitem__))
        meas_iter = iter(groupby(range(len(all_meas_ids)), all_meas_timestamps.__getitem__))

        next_truth = next(truth_iter, None)
        next_meas = next(meas_iter, None)

        switching_metric = _SwitchingLoss(self.switching_penalty, self.p)
        gospa_metrics = []

        while next_truth is not None or next_meas is not None:
            timestamp = min(group[0] for group in [next_truth, next_meas] if group is not None)

            truth_idxs = []

            if next_truth is not None and timestamp == next_truth[0]:
                truth_idxs = np.fromiter(next_truth[1], dtype=int)
                next_truth = next(truth_iter, None)

            meas_idxs = []

            if next_meas is not None and timestamp == next_meas[0]:
                meas_idxs = np.fromiter(next_meas[1], dtype=int)
                next_meas = next(meas_iter, None)

            meas_points = all_meas_points[meas_idxs]
            meas_ids = all_meas_ids[meas_idxs]

            truth_points = all_truth_points[truth_idxs]
            truth_ids = all_truth_ids[truth_idxs]

            metric, truth_to_measured_assignment = self.compute_gospa_metric(
                meas_points,
                truth_points
            )
            truth_mapping = {
                truth_id: meas_ids[meas_id] if meas_id != -1 else None
                for truth_id, meas_id in zip(truth_ids, truth_to_measured_assignment)}

            switching_metric.add_associations(truth_mapping)
            metric.value['switching'] = switching_metric.loss()
            metric.value['distance'] = np.power(metric.value['distance']**self.alpha +
                                                metric.value['switching']**self.alpha,
                                                1.0/self.alpha)
            gospa_metrics.append(metric)

        # If only one timestamp is present then return a SingleTimeMetric
        if len(gospa_metrics) == 1:
            return gospa_metrics[0]
        else:
            start_time = np.concatenate((all_truth_timestamps[:1], all_meas_timestamps[:1])).min()
            end_time = np.concatenate((all_truth_timestamps[-1:], all_meas_timestamps[-1:])).max()

            return TimeRangeMetric(
                title='GOSPA Metrics',
                value=gospa_metrics,
                time_range=TimeRange(start=start_time, end=end_time),
                generator=self)

    def compute_assignments(self, cost_matrix):
        """Compute assignments using modified Jonker-Volgenant algorithm

        Parameters
        ----------
        cost_matrix: Matrix (size mxn) that denotes the cost of assigning
                            mth truth state to each of the n measured states.

        Returns
        -------
        truth_to_measured: np.ndarray
            Vector of size m, which has indices of the measured objects or '-1' if unassigned.
        measured_to_truth: np.ndarray
            Vector of size n, which has indices of the truth objects or '-1' if unassigned.
        opt_cost: float
            Scalar value of the optimal assignment
        """

        m_truth, n_measured = cost_matrix.shape
        # Index for objects that will be left un-assigned.
        unassigned_idx = -1

        measured_to_truth = np.full((n_measured, ), unassigned_idx)
        truth_to_measured = np.full((m_truth, ), unassigned_idx)

        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        opt_cost = cost_matrix[row_ind, col_ind].sum()
        truth_to_measured[row_ind] = col_ind
        measured_to_truth[col_ind] = row_ind

        return truth_to_measured, measured_to_truth, opt_cost

    def compute_cost_matrix(self, track_states, truth_states, complete=False):
        """Creates the cost matrix between two lists of states

        This distance measure here will return distances minimum of either
        :attr:`~.c` or the distance calculated from :attr:`~.Measure`.

        Parameters
        ----------
        track_states: list of states
        truth_states: list of states
        complete: bool
            Cost matrix will be square, with :attr:`~.c` present for where
            there is a mismatch in cardinality

        Returns
        -------
        cost_matrix: np.ndarray
            Matrix of distance between each element in each list of states
        """

        if complete:
            m = n = max((len(track_states), len(truth_states)))
        else:
            m, n = len(track_states), len(truth_states)

        # c could be int, so force to float
        cost_matrix = np.full((m, n), self.c, dtype=np.float64)

        for i_track, track_state, in zip_longest(range(m), track_states):
            for i_truth, truth_state in zip_longest(range(n), truth_states):
                if None in (track_state, truth_state):
                    continue

                distance = self.measure(track_state, truth_state)
                if distance < self.c:
                    cost_matrix[i_track, i_truth] = distance

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
        unassigned_index = -1

        if num_truth_states == 0:
            # When truth states are empty all measured states are false
            opt_cost = -1.0 * num_measured_states * dummy_cost
            if self.alpha == 2:
                gospa_metric['false'] = opt_cost
        elif num_measured_states == 0:
            # When measured states are empty all truth
            # states are missed
            opt_cost = -1. * num_truth_states * dummy_cost
            if self.alpha == 2:
                gospa_metric['missed'] = opt_cost
        else:
            # Use assignment algorithm when both truth_states
            # and measured_states are non-empty
            cost_matrix = -1. * np.power(cost_matrix, self.p)
            truth_to_measured_assignment, measured_to_truth_assignment, _ =\
                self.compute_assignments(cost_matrix)

            opt_cost -= np.sum(measured_to_truth_assignment == unassigned_index) * dummy_cost
            if self.alpha == 2:
                gospa_metric['false'] -= \
                    np.sum(measured_to_truth_assignment == unassigned_index)*dummy_cost

            # Now use assignments to compute bids
            for i in range(num_truth_states):
                if truth_to_measured_assignment[i] != unassigned_index:
                    opt_cost += cost_matrix[i, truth_to_measured_assignment[i]]

                    if self.alpha == 2:
                        const_assign = truth_to_measured_assignment[i]
                        const_cmp = (-1 * self.c**self.p)

                        gospa_metric['localisation'] += \
                            cost_matrix[i, const_assign]*(cost_matrix[i, const_assign] > const_cmp)

                        gospa_metric['missed'] -= \
                            dummy_cost*(cost_matrix[i, const_assign] == const_cmp)

                        gospa_metric['false'] -= \
                            dummy_cost*(cost_matrix[i, const_assign] == const_cmp)

                        if cost_matrix[i, const_assign] == const_cmp:
                            truth_to_measured_assignment[i] = unassigned_index

                else:
                    opt_cost = opt_cost - dummy_cost
                    if self.alpha == 2:
                        gospa_metric['missed'] -= dummy_cost

        gospa_metric['distance'] = np.power((-1. * opt_cost), 1 / self.p)
        gospa_metric['localisation'] *= -1.
        gospa_metric['missed'] *= -1.
        gospa_metric['false'] *= -1.

        single_time_gospa_metric = SingleTimeMetric(
                title='GOSPA Metric', value=gospa_metric,
                timestamp=timestamps.pop(), generator=self)

        return single_time_gospa_metric, truth_to_measured_assignment


class OSPAMetric(GOSPAMetric):
    """
    Computes the Optimal SubPattern Assignment (OSPA) distance [1] for two sets
    of :class:`~.Track` objects. The OSPA distance is measured between two
    point patterns.

    The OSPA metric is calculated at each time step in which a :class:`~.Track`
    object is present

    Reference:
        [1] A Consistent Metric for Performance Evaluation of Multi-Object
        Filters, D. Schuhmacher, B. Vo and B. Vo, IEEE Trans. Signal Processing
        2008
    """
    c: float = Property(doc='Maximum distance for possible association')
    p: float = Property(doc='Norm associated to distance')
    generator_name: str = Property(doc="Unique identifier to use when accessing generated metrics "
                                       "from MultiManager",
                                   default='ospa_generator')

    def compute_over_time(self, measured_states, meas_ids, truth_states, truth_ids):
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
        Computes the Optimal SubPattern Assignment (OSPA) metric for a single
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
        if len(timestamps) > 1:
            raise ValueError(
                'All states must be from the same time to perform OSPA')

        if not track_states and not truth_states:  # pragma: no cover
            # For completeness, but can't generate metric without timestamp.
            distance = 0
        elif self.p < np.inf:
            cost_matrix = self.compute_cost_matrix(track_states, truth_states, complete=True)
            # Solve cost matrix with Hungarian/Munkres using
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            # Length of longest set of states
            n = max(len(track_states), len(truth_states))
            # Calculate metric
            distance = ((1/n) * np.sum(cost_matrix[row_ind, col_ind]**self.p))**(1/self.p)
        else:  # self.p == np.inf
            if len(track_states) == len(truth_states):
                cost_matrix = self.compute_cost_matrix(track_states, truth_states)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                distance = np.max(cost_matrix[row_ind, col_ind])
            else:
                distance = self.c

        return SingleTimeMetric(title='OSPA distance', value=distance,
                                timestamp=timestamps.pop(), generator=self)
