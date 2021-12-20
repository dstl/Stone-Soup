# -*- coding: utf-8 -*-
# © Copyright 2018-2021 University of Liverpool UK
# © Copyright 2021 Roke Manor Research Ltd UK
# Governed by MIT license - see LICENSE file or https://opensource.org/licenses/MIT

"""Gathers data together ready for a run of the multi-frame assignment algorithm.

The public API for this module is init_hyp_info(), which returns a HypInfo record.
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby
from operator import itemgetter
from typing import List, Dict, Tuple, Sequence
import numpy as np


@dataclass
class Hyp:
    """Hypothesis for a track, consisting of a choice of measurement assignments over time.

    Contains the information from a MultipleHypothesis but in a format more convenient for use in
    the steps of the MFA algorithm.
    """

    # Cost for this hypothesis
    cost: float
    # Track ID for the purposes
    trackID: int
    # List of measurement indices to associate with this track, or 0 for unassociated.
    measHistory: Sequence[int]
    # Measurement indices for the sliding window length (either padded at start or truncated);
    # after compacting measurement indices, the numbers will be different to measHistory
    # (the measurement indices are compacted to 0, ..., n-1, with -1 for unassociated).
    measHistorySlideWindow: np.ndarray
    # Is this a dummy hypothesis i.e. not a real track, representing a false alarm
    isDummy: bool = False

    @staticmethod
    def create(trackID: int, cost: float, measHistory: Sequence[int], slide_window: int):
        if len(measHistory) < slide_window:
            padding = (0,) * (slide_window - len(measHistory))
            measHistorySlideWindow = np.concatenate((padding, measHistory), dtype=np.int32)
        else:
            measHistorySlideWindow = np.array(measHistory[-slide_window:], dtype=np.int32)
        return Hyp(cost, trackID, measHistory, measHistorySlideWindow, False)


# Cost for assigning measurement to dummy track - represents probability density of false alarm
_DUMMY_TRACK_ASSIGNMENT_COST = 0


def _add_dummy_tracks(hyps: List[Hyp], all_measurements: List[Tuple[int, int]], slide_window: int):
    """Add dummy tracks for each non-null measurement

    Each measurement has a hypothesis that it was used and another that it wasn't so that all of
    the measurement hypotheses are used by the assignment.
    """
    zeros = np.zeros(slide_window, dtype=np.int32)
    for trackID, (time_index, measurement) in enumerate(all_measurements, start=hyps[-1].trackID+1):
        # Track without measurement assigned (allow measurement to be assigned to real track)
        hyps.append(Hyp(
            cost=0,
            trackID=trackID,
            measHistory=(),  # Not needed for dummy hypotheses
            measHistorySlideWindow=zeros,
            isDummy=True
        ))
        # Track with measurement assigned (measurement is false alarm)
        zeros_with_measurement = np.copy(zeros)
        zeros_with_measurement[time_index] = measurement
        hyps.append(Hyp(
            cost=_DUMMY_TRACK_ASSIGNMENT_COST,
            trackID=trackID,
            measHistory=(),  # Not needed for dummy hypotheses
            measHistorySlideWindow=zeros_with_measurement,
            isDummy=True
        ))


def _compact_measurement_indices(
    hyps: List[Hyp], all_measurements: List[Tuple[int, int]], slide_window: int
):
    """Compacts the measurement indices in measHistorySlideWindow of each Hyp.

    The main algorithm code assumes that measurements at each time step are 0, ... n - 1
    contiguously, with -1 for null measurement (track not assigned any measurement). But,
    initially, indices used at each time step are not guaranteed to be contiguous, and 0 is used to
    mean the null hypothesis; this function changes that.
    """
    meas_true_to_packed_index = [{0: -1} for _ in range(slide_window)]
    for time_index, measurements in groupby(all_measurements, key=itemgetter(0)):
        meas_true_to_packed_index[time_index].update({
            index_original: index_packed
            for index_packed, (t, index_original) in enumerate(measurements)
        })
    for h in hyps:
        h.measHistorySlideWindow = np.array(
            [meas_true_to_packed_index[t][m] for t, m in enumerate(h.measHistorySlideWindow)],
            dtype=np.int32
        )


@dataclass
class TimeStepIndices:
    """Contains useful mappings from track and measurement ID into master list of hypotheses.

    Note that track numbers (trackID) are consistent across time steps but measurement numbers
    are per time step.
    """
    # trackNull_index[i] is list of indices that, at this time step, assign the null hypothesis to
    # track i (i.e. these hypotheses do not associate track i with any measurement)
    # Philosophically, Dict[int, List[int]] is perhaps more correct, but the keys (the track IDs)
    # are contiguous so makes more sense to use List[List[int]].
    trackNull_index: List[List[int]]
    # measTrack_index[j, i] is list of indices that assign measurement j to track i
    measTrack_index: Dict[Tuple[int, int], List[int]]
    # measIndex[j] is a list of Hyp indices that use measurement j for any track
    # Does not include j==0 i.e. false alarm hypotheses
    measIndex: Dict[int, List[int]]
    # Number of measurements used in any hypothesis in this time step
    measurement_count: int


def _get_hyp_indices(hyps: List[Hyp], slide_window: int):
    """Computes useful indices mapping into the list of hypotheses.

    The main result is a list of HypInfo structures, which are used in the main algorithm step.
    In addition, track_to_hyp_map, which is a List[List[int]]. track_to_hyp_map[j] is a list of hyp
    indices that are for trackID j. This is used in construction of the constraints matrix.
    """
    track_count = hyps[-1].trackID + 1
    time_step_indices = [
        TimeStepIndices(
            [[] for trackID in range(track_count)], defaultdict(list), defaultdict(list), 0
        )
        for time_index in range(slide_window)
    ]
    track_to_hyp_map = [[] for trackID in range(track_count)]
    for hyp_index, hyp in enumerate(hyps):
        track_to_hyp_map[hyp.trackID].append(hyp_index)
        for time_index, measurement in enumerate(hyp.measHistorySlideWindow):
            this_time_step_indices = time_step_indices[time_index]
            if measurement == -1:
                # The track has not been assigned to any actual measurement (undetected)
                this_time_step_indices.trackNull_index[hyp.trackID].append(hyp_index)
            else:
                # An actual measurement to track assignment
                this_time_step_indices.measTrack_index[measurement, hyp.trackID].append(hyp_index)
                this_time_step_indices.measIndex[measurement].append(hyp_index)
                this_time_step_indices.measurement_count = max(
                    this_time_step_indices.measurement_count, measurement + 1
                )

    return time_step_indices, track_to_hyp_map


def _get_constraints_matrix(hyps, time_step_indices, track_to_hyp_map):
    # Each row has 1s for a selection of Hyp objects of which precisely one should be selected
    constraint_matrix_rows = []

    # Construct binary indicator matrix for constraint (1): each track should only be used once;
    # this constraint is necessary for the implementation of dual decomposition, since sliding
    # window is used.
    for hyp_index_list_for_track in track_to_hyp_map:
        this_row = np.zeros(len(hyps), dtype=np.int32)
        this_row[hyp_index_list_for_track] = 1
        constraint_matrix_rows.append(this_row)

    # Construct binary indicator matrix for constraint (2): each measurement in each scan should
    # only be used once
    for time_step_indices in time_step_indices:
        for measurement, hyp_index_list in time_step_indices.measIndex.items():
            this_row = np.zeros(len(hyps), dtype=np.int32)
            this_row[hyp_index_list] = 1
            constraint_matrix_rows.append(this_row)

    return np.array(constraint_matrix_rows, dtype=np.bool_)


@dataclass
class HypInfo:
    """Container for various information extracted from Hypothesis objects for MFA."""

    hyps: List[Hyp]
    # track_to_hyp_map[j] is list of hyp indices that are for trackID j
    # Although used as a map, the keys are contiguous integers so use a list
    track_to_hyp_map: List[List[int]]
    # TimeStepIndices structure for each time step
    time_step_indices: List[TimeStepIndices]
    # Binary indicator matrix of constraints used in the optimisation
    constraint_matrix: np.array
    # The individual costs of the Hyp objects pulled out into a single array
    all_costs: np.array


def init_hyp_info(hyps: List[Hyp], slide_window: int):
    all_measurements = sorted(set(
        (time_index, measurement)  # Time index is relative to start of sliding window
        for hyp in hyps
        for time_index, measurement in enumerate(hyp.measHistorySlideWindow)
        if measurement != 0
    ))
    _add_dummy_tracks(hyps, all_measurements, slide_window)
    _compact_measurement_indices(hyps, all_measurements, slide_window)
    time_step_indices, track_to_hyp_map = _get_hyp_indices(hyps, slide_window)
    return HypInfo(
        hyps=hyps,
        track_to_hyp_map=track_to_hyp_map,
        time_step_indices=time_step_indices,
        constraint_matrix=_get_constraints_matrix(hyps, time_step_indices, track_to_hyp_map),
        all_costs=np.array([hyp.cost for hyp in hyps], dtype=np.float64),
    )

