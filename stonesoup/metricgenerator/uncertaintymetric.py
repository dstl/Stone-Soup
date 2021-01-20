# -*- coding: utf-8 -*-

import numpy as np

from .base import MetricGenerator
from ..types.state import State, StateMutableSequence
from ..types.metric import SingleTimeMetric, TimeRangeMetric
from ..types.time import TimeRange


class SumofCovarianceNormsMetric(MetricGenerator):
    """
    Computes the sum of the covariance matrix norms of each state at a time step.
    The matrix norm calculated is the Frobenius norm. The metric generator will
    return this value at each time step in the track(s) as a measure of the uncertainty.
    """

    def compute_metric(self, manager):
        """Computes the metric using the data in the metric manager

        Parameters
        ----------
        manager : :class:`~.MetricManager`
            Contains the data to be used to create the metric

        Returns
        -------
        metric : list :class:`~.Metric`
            Containing the metric information. The value of the metric is a
            list of the metric at each timestamp

        """

        return self.compute_over_time(self.extract_states(manager.tracks))

    @staticmethod
    def extract_states(object_with_states):
        """
        Extracts a list of states from a list of (or single) objects
        containing states. This method is defined to handle :class:`~.StateMutableSequence`
        and :class:`~.State` types.

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

    def compute_over_time(self, track_states):
        """Compute the metric using the data in the metric manager

        Parameters
        ----------
        track_states : list of :class:`~.State`
            List of states created by a filter

        Returns
        ----------
        metric : TimeRangeMetric
            Covering the duration that states exist for in the parameters.
            Metric.value contains a list of the sums of covariance matrix norms
            at each timestamp

        """

        # Make a sorted list of all the unique timestamps used
        timestamps = sorted({state.timestamp for state in track_states})

        covnorm_sums = []

        for timestamp in timestamps:
            track_points = [state for state in track_states if state.timestamp == timestamp]
            covnorm_sums.append(self.compute_sum_covariancenorms(track_points))

        return TimeRangeMetric(
            title='Sum of Covariance Norms Metric',
            value=covnorm_sums,
            time_range=TimeRange(min(timestamps), max(timestamps)),
            generator=self)

    def compute_sum_covariancenorms(self, track_states):
        """
        Computes the sum of covariance norms metric for a single time step.

        Parameters
        ----------
        track_states: list of :class:`~.State`
            List of states created by a filter

        Returns
        -------
        metric: SingleTimeMetric
            The sum of covariance matrix norms metric at a single time step
        """

        timestamps = {state.timestamp for state in track_states}
        if len(timestamps) > 1:
            raise ValueError(
                'All states must be from the same time to compute total uncertainty')

        covnorms_sum = 0

        for state in track_states:
            covnorm = np.linalg.norm(state.covar)
            covnorms_sum += covnorm

        return SingleTimeMetric(title='Covariance Matrix Norm Sum', value=covnorms_sum,
                                timestamp=timestamps.pop(), generator=self)
