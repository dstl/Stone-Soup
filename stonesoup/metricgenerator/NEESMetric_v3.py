import datetime
from itertools import chain
import numpy as np
from scipy.optimize import linear_sum_assignment
from ..types.state import State, StateMutableSequence
from ..types.time import TimeRange
from ..types.metric import SingleTimeMetric, TimeRangeMetric
from ..measures import Euclidean
from ..types.state import GaussianState



class NEESMetric:
    """

    Computes the Normalized Estimation Error Squared (NEES) metric for state estimators.
    The NEES metric is used to assess the consistency of state estimators.

    The NEES metric is defined as:

    \[ \text{NEES} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{x}_i)^T P_i^{-1} (x_i - \hat{x}_i) \]

    Where:
    - \( x_i \) is the true state vector.
    - \( \hat{x}_i \) is the estimated state vector.
    - \( P_i \) is the estimation error covariance matrix.
    - \( n \) is the number of states.


    This implementation computes NEES values over time to compute the averaged NEES (ANEES) and includes a
    threshold parameter `c` to limit the influence of large errors.
    """

    def __init__(self, confidence_threshold=3, generator_name='nees_generator', tracks_key='tracks', truths_key='groundtruth_paths',
                 measure=None):
        """
        Initialize NEESMetric class.

        Parameters
        ----------
        generator_name : str, optional
            Unique identifier to use when accessing generated metrics from MultiManager, by default 'nees_generator'
        tracks_key : str, optional
            Key to access set of tracks added to MetricManager, by default 'tracks'
        truths_key : str, optional
            Key to access set of ground truths added to MetricManager, by default 'groundtruth_paths'
        measure : Measure, optional
            Measure to compute distances between states, default is Euclidean().
        confidence_threshold: float
            Threshold (in units of standard deviation) for Mahalanobis distance to consider a valid assignment.
        """
        self.generator_name = generator_name
        self.tracks_key = tracks_key
        self.truths_key = truths_key
        if confidence_threshold < 0:
            raise ValueError("confidence_threshold must be non-negative.")
        self.confidence_threshold = confidence_threshold  # Threshold parameter
        if measure is None:
            self.measure = Euclidean()
        else:
            self.measure = measure

    def compute_metric(self, manager):
        """Compute the NEES metric using the data in the metric manager

        Parameters
        ----------
        manager : :class:`~.MetricManager`
            contains the data to be used to create the metric(s)

        Returns
        -------
        metric : :class:`~.TimeRangeMetric`
            Contains the final metrics over the time range.
        """
        return self.compute_over_time_v2(
            *self.extract_states(manager.states_sets[self.tracks_key], True),
            *self.extract_states(manager.states_sets[self.truths_key], True)
        )

    def compute_over_time_v2(self, measured_states, measured_state_ids, truth_states, truth_state_ids):
        """
        Compute the NEES metric over time from lists of measured and truth states.

        Parameters
        ----------
        measured_states: List of states from the tracker.
        measured_state_ids: IDs for the measured states.
        truth_states: List of ground truth states.
        truth_state_ids: IDs for the truth states.

        Returns
        -------
        metric: TimeRangeMetric
            Contains the final ANEES and average cardinality errors.
        """

        # Handle cases with no states
        if len(measured_states) == 0 and len(truth_states) == 0:
            final_metrics = {
                'ANEES': 0.0,
                'Average False Tracks': 0.0,
                'Average Missed Targets': 0.0
            }
            return SingleTimeMetric(
                title='NEES Metric', value=final_metrics, timestamp=None, generator=self)
        elif len(measured_states) == 0:
            raise ValueError('Cannot compute NEES: no measured states provided.')
        elif len(truth_states) == 0:
            raise ValueError('Cannot compute NEES: no truth states provided.')

        # Validate timestamps
        for state in chain(measured_states, truth_states):
            if state.timestamp is None:
                raise ValueError("All states must have valid timestamps.")

        # Get all unique timestamps
        timestamps = sorted({
            state.timestamp
            for state in chain(measured_states, truth_states)})

        # Initialize cumulative metrics
        total_nees = 0.0
        total_num_state_estimates = 0  # Total \#SE
        total_num_false_tracks = 0  # Total \#FA
        total_num_missed_targets = 0  # Total \#MT
        total_time_steps = len(timestamps)

        # Initialize per-timestamp metrics
        per_timestamp_metrics = {
            'timestamps': [],
            'NEES': [],
            'Num_State_Estimates': [],
            'False_Tracks': [],
            'Missed_Targets': []
        }

        for timestamp in timestamps:
            # Filter states at current timestamp
            meas_states_at_time = [state for state in measured_states if state.timestamp == timestamp]
            truth_states_at_time = [state for state in truth_states if state.timestamp == timestamp]

            # Compute metrics at current timestamp
            nees, num_state_estimates, num_false_tracks, num_missed_targets = \
                self.compute_nees_metric_v2(meas_states_at_time, truth_states_at_time)

            # Accumulate totals
            total_nees += nees
            total_num_state_estimates += num_state_estimates
            total_num_false_tracks += num_false_tracks
            total_num_missed_targets += num_missed_targets

            # Store per-timestamp metrics
            per_timestamp_metrics['timestamps'].append(timestamp)
            per_timestamp_metrics['NEES'].append(nees)
            per_timestamp_metrics['Num_State_Estimates'].append(num_state_estimates)
            per_timestamp_metrics['False_Tracks'].append(num_false_tracks)
            per_timestamp_metrics['Missed_Targets'].append(num_missed_targets)

        if total_num_state_estimates == 0:
            # No matched states; return zero metric
            anees_value = 0.0
        else:
            anees_value = total_nees / total_num_state_estimates

        # Compute average number of false tracks and missed targets per time step
        avg_num_false_tracks = total_num_false_tracks / total_time_steps
        avg_num_missed_targets = total_num_missed_targets / total_time_steps

        # Create a dictionary to hold the final metrics
        final_metrics = {
            'ANEES': anees_value,
            'Average False Tracks': avg_num_false_tracks,
            'Average Missed Targets': avg_num_missed_targets
        }

        # Create metadata_info containing per-timestamp metrics and total_time_steps
        metadata_info = {
            'per_timestamp_metrics': per_timestamp_metrics,
            'total_time_steps': total_time_steps
        }

        # Determine the title
        title_metric = 'NEES Metrics'

        # Handle time range when only one timestamp is present
        if len(timestamps) == 1:
            time_range = TimeRange(timestamps[0], timestamps[0] + datetime.timedelta(microseconds=1))
        else:
            time_range = TimeRange(min(timestamps), max(timestamps))

        # Return the TimeRangeMetric containing the final metrics
        return TimeRangeMetric(
            title=title_metric,
            value=final_metrics,
            metadata=metadata_info,
            time_range=time_range,
            generator=self)

    def compute_nees_metric_v2(self, measured_states, truth_states):
        """Computes NEES metric between measured and truth states, considering assignments within confidence region.

        Parameters
        ----------
        measured_states: list of :class:`~.State`
            List of state objects to be assigned to the truth.
        truth_states: list of :class:`~.State`
            List of state objects for the truth points.

        Returns
        -------
        nees: float
            Sum of NEES over assigned pairs.
        num_state_estimates: int
            Number of valid assignments (number of state estimates).
        num_false_tracks: int
            Number of tracks not assigned to any target.
        num_missed_targets: int
            Number of targets not assigned to any track.
        """

        # Handle cases with no states
        if len(measured_states) == 0 and len(truth_states) == 0:
            return 0.0, 0, 0, 0
        elif len(measured_states) == 0:
            return 0.0, 0, 0, len(truth_states)
        elif len(truth_states) == 0:
            return 0.0, 0, len(measured_states), 0

        # Compute the cost matrix with confidence region
        cost_matrix = self.compute_cost_matrix_v2(measured_states, truth_states, complete=True)

        # Solve the assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        num_truth_states = len(truth_states)
        num_measured_states = len(measured_states)
        unassigned_index = -1

        # Initialize assignment arrays
        truth_to_measured_assignment = np.full(num_truth_states, unassigned_index)
        measured_to_truth_assignment = np.full(num_measured_states, unassigned_index)

        # Update assignments
        for i in range(len(row_ind)):
            measured_idx = row_ind[i]
            truth_idx = col_ind[i]
            if cost_matrix[measured_idx, truth_idx] >= self.confidence_threshold:
                continue  # Skip invalid assignments
            measured_to_truth_assignment[measured_idx] = truth_idx
            truth_to_measured_assignment[truth_idx] = measured_idx

        nees = 0.0
        num_state_estimates = 0  # \#SE

        # Loop over assigned pairs
        for measured_idx, truth_idx in enumerate(measured_to_truth_assignment):
            if truth_idx != unassigned_index:
                meas_state = measured_states[measured_idx]
                truth_state = truth_states[truth_idx]

                # Ensure states are GaussianState instances
                if not isinstance(meas_state, GaussianState) or not isinstance(truth_state, GaussianState):
                    raise TypeError("States must be instances of GaussianState.")

                # Add dimension check
                if meas_state.state_vector.shape[0] != truth_state.state_vector.shape[0]:
                    raise ValueError("State vector dimensions must match.")

                # **Add dimension check**
                if meas_state.state_vector.shape[0] != truth_state.state_vector.shape[0]:
                    raise ValueError("State vector dimensions must match.")

                # **Add checks for NaNs and Infs in state vectors and covariance matrices**
                if np.isnan(meas_state.state_vector).any() or np.isnan(truth_state.state_vector).any():
                    raise ValueError("State vectors contain NaNs.")
                if np.isinf(meas_state.state_vector).any() or np.isinf(truth_state.state_vector).any():
                    raise ValueError("State vectors contain Infs.")
                if np.isnan(meas_state.covar).any() or np.isnan(truth_state.covar).any():
                    raise ValueError("Covariance matrix contains NaNs.")

                # Compute error
                error = truth_state.state_vector - meas_state.state_vector

                # Compute NEES
                covar = meas_state.covar

                # Check for positive definiteness
                try:
                    np.linalg.cholesky(covar)
                except np.linalg.LinAlgError:
                    raise np.linalg.LinAlgError("Covariance matrix is not positive definite.")

                # Invert covariance matrix
                inv_covar = np.linalg.inv(covar)

                nees_value = (error.T @ inv_covar @ error).item()

                # Normalize NEES by state dimension
                DoF = meas_state.state_vector.shape[0]  # Assuming square covariance matrices
                normalized_nees = nees_value / DoF

                nees += normalized_nees
                num_state_estimates += 1

        # Count false tracks and missed targets
        num_false_tracks = np.sum(measured_to_truth_assignment == unassigned_index)
        num_missed_targets = np.sum(truth_to_measured_assignment == unassigned_index)

        return nees, num_state_estimates, num_false_tracks, num_missed_targets

    def compute_cost_matrix_v2(self, track_states, truth_states, complete=False):
        """Creates the cost matrix between two lists of states, using Mahalanobis distance and confidence gating.

        Parameters
        ----------
        track_states: list of states
        truth_states: list of states
        complete: bool
            If True, cost matrix will be square, with high cost for mismatches in cardinality.
        Returns
        -------
        cost_matrix: np.ndarray
            Matrix of Mahalanobis distances between each track and truth state.
        """

        if complete:
            m = n = max(len(track_states), len(truth_states))
        else:
            m, n = len(track_states), len(truth_states)

        # Initialize cost matrix with high cost
        cost_matrix = np.full((m, n), self.confidence_threshold, dtype=np.float64)

        for i_track, track_state in enumerate(track_states):
            for i_truth, truth_state in enumerate(truth_states):

                # Ensure states are GaussianState instances
                if not isinstance(track_state, GaussianState) or not isinstance(truth_state, GaussianState):
                    raise TypeError("States must be instances of GaussianState.")
                # add checks for NaNs and Infs
                if np.isnan(track_state.state_vector).any() or np.isnan(truth_state.state_vector).any():
                    raise ValueError("State vectors contain NaNs.")
                if np.isinf(track_state.state_vector).any() or np.isinf(truth_state.state_vector).any():
                    raise ValueError("State vectors contain Infs.")
                if np.isnan(track_state.covar).any() or np.isnan(truth_state.covar).any():
                    raise ValueError("Covariance matrix contains NaNs.")

                # covariance test
                covar = track_state.covar
                try:
                    np.linalg.inv(covar)
                except np.linalg.LinAlgError:
                    raise np.linalg.LinAlgError("Covariance matrix is singular.")

                distance = self.measure(track_state, truth_state)
                if distance < self.confidence_threshold:
                    cost_matrix[i_track, i_truth] = distance
        return cost_matrix

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
            if isinstance(element, StateMutableSequence):
                states = list(element.last_timestamp_generator())
                state_list.extend(states)
                ids.extend([i] * len(states))
            elif isinstance(element, State):
                state_list.append(element)
                ids.extend([i])
            else:
                raise ValueError(
                    "{!r} has no state extraction method".format(element))
        if return_ids:
            return state_list, ids
        return state_list