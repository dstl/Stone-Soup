from itertools import chain
import datetime
import numpy as np
from scipy.optimize import linear_sum_assignment
from ..types.state import State, StateMutableSequence, StateVector, CovarianceMatrix
from ..types.time import TimeRange
from ..types.metric import SingleTimeMetric, TimeRangeMetric
from ..measures import Euclidean
from stonesoup.types.update import GaussianStateUpdate

class RMSEMetric:
    """
    Computes the Root Mean Squared Error (RMSE) metric for state estimators.
    The RMSE metric is used to measure the accuracy of state estimators.

    This implementation computes separate RMSE values for position and velocity components
    and includes a standard deviation confidence threshold parameter `confidence_threshold` to limit the influence
     of large errors.

    The RMSE metric is defined as:
    \[ \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{x}_i)^2} \]

    Where:
    - \( x_i \) is the true state vector.
    - \( \hat{x}_i \) is the estimated state vector.
    - \( n \) is the number of states.
    """

    def __init__(self, confidence_threshold=3, generator_name='rmse_generator', tracks_key='tracks',
                 truths_key='groundtruth_paths', measure=None, components=None):
        """
        Initialize RMSEMetric class.

        Parameters
        ----------
        generator_name : str, optional
            Unique identifier to use when accessing generated metrics from MultiManager, by default 'rmse_generator'
        tracks_key : str, optional
            Key to access set of tracks added to MetricManager, by default 'tracks'
        truths_key : str, optional
            Key to access set of ground truths added to MetricManager, by default 'groundtruth_paths'
        measure : Measure, optional
            Measure to compute distances between states, default is Euclidean().
        components : list of int, optional
            Indices of the state vector to include in the RMSE calculation.
                    If None, all components are used.
        confidence_threshold: float
            Threshold (in units of standard deviation) for Mahalanobis distance to consider a valid assignment.
        """
        self.generator_name = generator_name
        self.tracks_key = tracks_key
        self.truths_key = truths_key
        self.measure = measure if measure is not None else Euclidean()
        self.components = components  # Components to include in RMSE
        if confidence_threshold < 0:
            raise ValueError("confidence_threshold must be non-negative.")
        self.confidence_threshold = confidence_threshold

    def compute_metric(self, manager):
        """Compute the RMSE metric using the data in the metric manager

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
        # return self.compute_over_time(
        return self.compute_over_time_v2(
            *self.extract_states(manager.states_sets[self.tracks_key], True),
            *self.extract_states(manager.states_sets[self.truths_key], True)
        )

    def compute_over_time_v2(self, measured_states, measured_state_ids, truth_states, truth_state_ids):
        """
        Compute the RMSE metric over time from lists of measured and truth states.

        Parameters
        ----------
        measured_states: List of states from the tracker.
        measured_state_ids: IDs for the measured states.
        truth_states: List of ground truth states.
        truth_state_ids: IDs for the truth states.

        Returns
        -------
        metric: TimeRangeMetric
            Contains the final RMSE, and average cardinality errors.
        """

        # Handle cases with no states
        if len(measured_states) == 0 and len(truth_states) == 0:
            return SingleTimeMetric(
                title='RMSE Metric', value=0.0, timestamp=None, generator=self)
        elif len(measured_states) == 0:
            raise ValueError('Cannot compute RMSE: no measured states provided.')
        elif len(truth_states) == 0:
            raise ValueError('Cannot compute RMSE: no truth states provided.')

        # Check for None or empty state vectors
        for state in chain(measured_states, truth_states):
            if not hasattr(state, 'state_vector'):
                raise ValueError("State does not have a state_vector attribute.")
            state_vector = state.state_vector
            if state_vector is None:
                raise ValueError("State vectors cannot be None.")
            if state_vector.size == 0:
                raise ValueError("State vectors cannot be empty.")
            if np.isnan(state_vector).any():
                raise ValueError("State vectors contain NaNs.")
            if np.isinf(state_vector).any():
                raise ValueError("State vectors contain Infs.")

        # Check for missing timestamps
        if any(state.timestamp is None for state in chain(measured_states, truth_states)):
            raise ValueError("All states must have a timestamp.")

        # Check for empty state vectors
        for state in chain(measured_states, truth_states):
            if state.state_vector.size == 0:
                raise ValueError("State vectors cannot be empty.")

        # Get all unique timestamps
        timestamps = sorted({
            state.timestamp
            for state in chain(measured_states, truth_states)
        })

        # Initialize cumulative metrics
        total_mse = 0.0
        # total_num_error_terms = 0  # Initialize total number of error terms
        total_num_state_estimates = 0  # Total \#SE
        total_num_false_tracks = 0  # Total \#FA
        total_num_missed_targets = 0  # Total \#MT
        total_time_steps = len(timestamps)

        # Initialize per-timestamp metrics
        per_timestamp_metrics = {
            'timestamps': [],
            'MSE': [],
            'Num_State_Estimates': [],
            'False_Tracks': [],
            'Missed_Targets': []
        }

        for timestamp in timestamps:
            # Filter states at current timestamp
            meas_states_at_time = [state for state in measured_states if state.timestamp == timestamp]
            truth_states_at_time = [state for state in truth_states if state.timestamp == timestamp]

            # Compute metrics at current timestamp
            mse, num_state_estimates, num_false_tracks, num_missed_targets = \
                self.compute_rmse_metric_v2(meas_states_at_time, truth_states_at_time)

            # Accumulate totals
            total_mse += mse
            total_num_state_estimates += num_state_estimates

            total_num_false_tracks += num_false_tracks
            total_num_missed_targets += num_missed_targets

            # Store per-timestamp metrics
            per_timestamp_metrics['timestamps'].append(timestamp)
            per_timestamp_metrics['MSE'].append(mse)
            per_timestamp_metrics['Num_State_Estimates'].append(num_state_estimates)
            per_timestamp_metrics['False_Tracks'].append(num_false_tracks)
            per_timestamp_metrics['Missed_Targets'].append(num_missed_targets)

        if total_num_state_estimates == 0:
            # if total_num_error_terms == 0:
            # No matched states; return zero metric
            rmse_value = 0.0
        else:
            rmse_value = np.sqrt(total_mse / total_num_state_estimates)

        # Compute average number of false tracks and missed targets per time step
        avg_num_false_tracks = total_num_false_tracks / total_time_steps
        avg_num_missed_targets = total_num_missed_targets / total_time_steps

        # Create a dictionary to hold the final metrics
        final_metrics = {
            'RMSE': rmse_value,
            'Average False Tracks': avg_num_false_tracks,
            'Average Missed Targets': avg_num_missed_targets
        }

        # Create metadata_info containing per-timestamp metrics and total_time_steps
        metadata_info = {
            'per_timestamp_metrics': per_timestamp_metrics,
            'total_time_steps': total_time_steps
        }

        # Determine the title based on components
        if self.components is not None:
            if self.components == [0, 2]:
                title_metric = 'RMSE Metrics Position'
            else:
                title_metric = 'RMSE Metrics Velocity'
        else:
            title_metric = 'RMSE Metrics'

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

    def compute_rmse_metric_v2(self, measured_states, truth_states):
        """Computes RMSE metric between measured and truth states, considering assignments within confidence region.

        Parameters
        ----------
        measured_states: list of :class:`~.State`
            List of state objects to be assigned to the truth.
        truth_states: list of :class:`~.State`
            List of state objects for the truth points.

        Returns
        -------
        mse: float
            Sum of squared errors over assigned pairs.
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

        mse = 0.0
        num_state_estimates = 0  # \#SE (Number of assigned pairs)

        # Loop over assigned pairs
        for measured_idx, truth_idx in enumerate(measured_to_truth_assignment):
            if truth_idx != unassigned_index:

                meas_state = measured_states[measured_idx]
                truth_state = truth_states[truth_idx]

                if meas_state.state_vector is None or truth_state.state_vector is None:
                    raise ValueError("State vectors cannot be None.")

                # Ensure state vectors are compatible
                if meas_state.state_vector.shape != truth_state.state_vector.shape:
                    raise ValueError("State vector dimensions must match.")

                # Add checks for NaNs and infs in state vectors
                if np.isnan(meas_state.state_vector).any() or np.isnan(truth_state.state_vector).any():
                    raise ValueError("State vectors contain NaNs.")

                if np.isinf(meas_state.state_vector).any() or np.isinf(truth_state.state_vector).any():
                    raise ValueError("State vectors contain Infs.")

                # Compute error
                error = truth_state.state_vector - meas_state.state_vector

                # Extract specified components
                if self.components is not None:
                    error = error[self.components, :]

                # Compute squared error
                squared_error = np.sum(error ** 2)
                mse += squared_error

                num_state_estimates += 1

        # Count false tracks and missed targets
        num_false_tracks = np.sum(measured_to_truth_assignment == unassigned_index)
        num_missed_targets = np.sum(truth_to_measured_assignment == unassigned_index)

        return mse, num_state_estimates, num_false_tracks, num_missed_targets

    def compute_cost_matrix_v2(self, track_states, truth_states, complete=False):
        """Creates the cost matrix between two lists of states, using a distance metric and confidence gating.

        Parameters
        ----------
        track_states: list of states
        truth_states: list of states
        complete: bool
            If True, cost matrix will be square, with high cost for mismatches in cardinality.

        Returns
        -------
        cost_matrix: np.ndarray
            Matrix of distances between each track and truth state.
        """
        if complete:
            m = n = max(len(track_states), len(truth_states))
        else:
            m, n = len(track_states), len(truth_states)

        # Initialize cost matrix with high cost
        cost_matrix = np.full((m, n), self.confidence_threshold, dtype=np.float64)

        for i_track, track_state in enumerate(track_states):
            for i_truth, truth_state in enumerate(truth_states):

                # Ensure the track and truth states are valid
                if track_state is None or truth_state is None:
                    raise ValueError("State cannot be None.")
                if np.isnan(track_state.state_vector).any() or np.isnan(truth_state.state_vector).any():
                    raise ValueError("State vectors contain NaNs.")
                if np.isinf(track_state.state_vector).any() or np.isinf(truth_state.state_vector).any():
                    raise ValueError("State vectors contain Infs.")

                # Extract the state vector, considering different object types
                if hasattr(track_state, 'mean'):
                    track_vector = track_state.mean
                elif hasattr(track_state, 'state_vector'):
                    track_vector = track_state.state_vector
                else:
                    raise AttributeError("Track state does not have 'mean' or 'state_vector' attributes")

                if hasattr(truth_state, 'mean'):
                    truth_vector = truth_state.mean
                elif hasattr(truth_state, 'state_vector'):
                    truth_vector = truth_state.state_vector
                else:
                    raise AttributeError("Truth state does not have 'mean' or 'state_vector' attributes")

                # Initialize track_state_vector and truth_state_vector
                track_state_vector = track_state
                truth_state_vector = truth_state

                # Extract specified components for distance calculation if `self.components` is defined
                if self.components is not None:
                    track_state_vector = extract_components(track_state, self.components)
                    truth_state_vector = extract_components(truth_state, self.components)

                # Ensure covariance matrix exists for Mahalanobis distance calculation
                if isinstance(self.measure, Euclidean):
                    # Use the measure to calculate the Euclidean distance between track_state and truth_state
                    distance = self.measure(track_state_vector, truth_state_vector)
                else:
                    if hasattr(track_state_vector, 'covar'):
                        # Your code for calculating distance...
                        distance = self.measure(track_state_vector, truth_state_vector)
                    else:
                        raise ValueError(
                            "Track state must have a covariance matrix for Mahalanobis distance calculation.")


                # Check if distance is within the confidence threshold
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

def extract_components(gaussian_state_update, components):
    """
    Extract specified components from a GaussianStateUpdate instance.

    Parameters
    ----------
    gaussian_state_update : GaussianStateUpdate
        An instance of GaussianStateUpdate.
    components : list of int
        The indices of the components to extract (e.g., [0, 2] or [1, 3]).

    Returns
    -------
    GaussianStateUpdate
        A new instance of GaussianStateUpdate containing only the specified components.
    """

    # Extract the state vector with specified components
    state_vector = gaussian_state_update.state_vector
    if state_vector is not None:
        selected_state_vector = StateVector(state_vector[components, :])
    else:
        selected_state_vector = None

    # Extract the covariance matrix with specified components
    covar = gaussian_state_update.covar
    if covar is not None:
        selected_covariance_matrix = CovarianceMatrix(covar[np.ix_(components, components)])
    else:
        selected_covariance_matrix = None

    # Create a new GaussianStateUpdate instance with the extracted components
    new_gaussian_state_update = GaussianStateUpdate(
        state_vector=selected_state_vector,
        covar=selected_covariance_matrix,
        hypothesis=None,
        timestamp=gaussian_state_update.timestamp
    )

    return new_gaussian_state_update