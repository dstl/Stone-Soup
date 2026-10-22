import datetime
from itertools import chain
import numpy as np
from scipy.optimize import linear_sum_assignment
from stonesoup.types.state import State, StateMutableSequence, GaussianState
from stonesoup.types.time import TimeRange
from stonesoup.types.metric import SingleTimeMetric, TimeRangeMetric
from stonesoup.measures.state import KLDivergence
from ..measures import Euclidean

class KLMetric:
    """
    Computes the Kullback-Leibler (KL) divergence metric between estimated states and ground truth states.
    The KL divergence is a measure of how one probability distribution diverges from a second, expected probability
    distribution.
    """

    def __init__(self, confidence_threshold=3, generator_name='kl_generator',
                 tracks_key='tracks', truths_key='groundtruth_paths', measure=None, components=None):
        """
        Initialize KLMetric class.

        Parameters
        ----------
        generator_name : str, optional
            Unique identifier to use when accessing generated metrics from MultiManager, by default 'kl_generator'
        tracks_key : str, optional
            Key to access set of tracks added to MetricManager, by default 'tracks'
        truths_key : str, optional
            Key to access set of ground truths added to MetricManager, by default 'groundtruth_paths'
        measure : Measure, optional
            Measure to compute distances between states, default is Euclidean().
        confidence_threshold: float
            Threshold (in units of standard deviation) for Mahalanobis distance to consider a valid assignment.
        """
        # Initialization as described earlier
        self.generator_name = generator_name
        self.tracks_key = tracks_key
        self.truths_key = truths_key
        if confidence_threshold < 0:
            raise ValueError("confidence_threshold must be non-negative.")
        self.confidence_threshold = confidence_threshold
        self.components = components  # Default components set to None
        if measure is None:
            self.measure = Euclidean()
        else:
            self.measure = measure

    def compute_metric(self, manager):
        """Compute the KL metric using the data in the metric manager

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
        Compute the KL divergence metric at every timestep from a list of measured
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
        for the KL divergence metric at each timestamp
        """
        # Handle cases with no states
        if len(measured_states) == 0 and len(truth_states) == 0:
            final_metrics = {
                'Average KL Divergence': 0.0,
                'Average False Tracks': 0.0,
                'Average Missed Targets': 0.0
            }
            return SingleTimeMetric(
                title='KL Metric', value=final_metrics, timestamp=None, generator=self)
        elif len(measured_states) == 0:
            raise ValueError('Cannot compute KL: no measured states provided.')
        elif len(truth_states) == 0:
            raise ValueError('Cannot compute KL: no truth states provided.')

        # Validate timestamps
        for state in chain(measured_states, truth_states):
            if state.timestamp is None:
                raise ValueError("All states must have valid timestamps.")

        # Get all unique timestamps
        timestamps = sorted({
            state.timestamp
            for state in chain(measured_states, truth_states)})

        # Initialize cumulative metrics
        total_kl_divergence = 0.0
        total_num_state_estimates = 0  # Total \#SE
        total_num_false_tracks = 0  # Total \#FA
        total_num_missed_targets = 0  # Total \#MT
        total_time_steps = len(timestamps)

        # Initialize per-timestamp metrics
        per_timestamp_metrics = {
            'timestamps': [],
            'KL_Divergence': [],
            'Num_State_Estimates': [],
            'False_Tracks': [],
            'Missed_Targets': []
        }

        for timestamp in timestamps:
            # Filter states at current timestamp
            meas_states_at_time = [state for state in measured_states if state.timestamp == timestamp]
            truth_states_at_time = [state for state in truth_states if state.timestamp == timestamp]

            # Compute metrics at current timestamp
            kl_divergence_sum, num_state_estimates, num_false_tracks, num_missed_targets = \
                self.compute_kl_metric_v2(meas_states_at_time, truth_states_at_time)

            # Accumulate totals
            total_kl_divergence += kl_divergence_sum
            total_num_state_estimates += num_state_estimates
            total_num_false_tracks += num_false_tracks
            total_num_missed_targets += num_missed_targets

            # Store per-timestamp metrics
            per_timestamp_metrics['timestamps'].append(timestamp)
            per_timestamp_metrics['KL_Divergence'].append(kl_divergence_sum)
            per_timestamp_metrics['Num_State_Estimates'].append(num_state_estimates)
            per_timestamp_metrics['False_Tracks'].append(num_false_tracks)
            per_timestamp_metrics['Missed_Targets'].append(num_missed_targets)

        if total_num_state_estimates == 0:
            # No matched states; return zero metric
            average_kl_divergence = 0.0
        else:
            average_kl_divergence = total_kl_divergence / total_num_state_estimates

        # Compute average number of false tracks and missed targets per time step
        avg_num_false_tracks = total_num_false_tracks / total_time_steps
        avg_num_missed_targets = total_num_missed_targets / total_time_steps

        # Create a dictionary to hold the final metrics
        final_metrics = {
            'Average KL Divergence': average_kl_divergence,
            'Average False Tracks': avg_num_false_tracks,
            'Average Missed Targets': avg_num_missed_targets
        }

        # Create metadata_info containing per-timestamp metrics and total_time_steps
        metadata_info = {
            'per_timestamp_metrics': per_timestamp_metrics,
            'total_time_steps': total_time_steps
        }

        # Determine the title based on components
        if self.components == [0, 2]:
            title_metric = 'KL Metrics Position'
        elif self.components == [1, 2]:
            title_metric = 'KL Metrics Veloctiy'
        else:
            title_metric = 'KL Metrics'

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

    def compute_kl_metric_v2(self, measured_states, truth_states):
        """Computes KL divergence metric between measured and truth states.

        Parameters
        ----------
        measured_states: list of :class:`~.State`
            list of state objects to be assigned to the truth
        truth_states: list of :class:`~.State`
            list of state objects for the truth points

        Returns
        -------
        kl_metric: float
            Sum of  KL divergence score over assigned pairs.
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

        kl_divergence_sum = 0.0
        num_state_estimates = 0  # \#SE

        kl_divergence = KLDivergence()

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

                # Determine components to use
                if self.components is not None:
                    components = self.components
                else:
                    # Use all components based on the state vector size
                    components = list(range(meas_state.state_vector.shape[0]))

                # Check if components are valid for both measured and truth states
                if (max(components) >= meas_state.state_vector.shape[0] or
                        max(components) >= truth_state.state_vector.shape[0]):
                    raise ValueError("Component indices are out of bounds for the state vectors.")

                # Extract specified components
                meas_state_vector = meas_state.state_vector[components, :]
                truth_state_vector = truth_state.state_vector[components, :]
                meas_covar = meas_state.covar[np.ix_(components, components)]
                truth_covar = truth_state.covar[np.ix_(components, components)]
                # Check for positive definiteness
                try:
                    np.linalg.cholesky(meas_covar)
                    np.linalg.cholesky(truth_covar)
                except np.linalg.LinAlgError:
                    raise np.linalg.LinAlgError("Covariance matrix is not positive definite.")

                # Create new GaussianState instances with selected components
                meas_state_comp = GaussianState(meas_state_vector, meas_covar)
                truth_state_comp = GaussianState(truth_state_vector, truth_covar)

                # Compute KL divergence
                try:
                    kl_value = kl_divergence(truth_state_comp, meas_state_comp)
                except np.linalg.LinAlgError:
                    raise np.linalg.LinAlgError("Covariance matrix is singular.")

                kl_divergence_sum += kl_value
                num_state_estimates += 1

        # Count false tracks and missed targets
        num_false_tracks = np.sum(measured_to_truth_assignment == unassigned_index)
        num_missed_targets = np.sum(truth_to_measured_assignment == unassigned_index)

        return kl_divergence_sum, num_state_estimates, num_false_tracks, num_missed_targets

    def compute_cost_matrix_v2(self, track_states, truth_states, complete=False):
        """Creates the cost matrix between two lists of states."""
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

                # Compute the error vector
                covar = track_state.covar

                # Compute Mahalanobis distance
                try:
                    np.linalg.inv(covar)
                except np.linalg.LinAlgError:
                    raise np.linalg.LinAlgError("Covariance matrix is singular.")

                # Check for positive definiteness
                try:
                    np.linalg.cholesky(covar)
                except np.linalg.LinAlgError:
                    raise np.linalg.LinAlgError("Covariance matrix is not positive definite.")

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
