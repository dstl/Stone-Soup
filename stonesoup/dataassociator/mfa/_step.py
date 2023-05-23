# © Copyright 2018-2021 University of Liverpool UK
# © Copyright 2021 Roke Manor Research Ltd UK
# Governed by MIT license - see LICENSE file or https://opensource.org/licenses/MIT

"""Performs a single iteration of the multi-frame assignment algorithm.

This contains the real meat of the MFA algorithm. The dual subproblem is tackled in
_getSuboptimalSolutionForSubproblem(), while the primal problem is solved in _getPrimalSolution().
These are both called from algorithm_step(), which advances the algorithm by one step of the
overall iteration.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment
try:
    from ortools.linear_solver import pywraplp
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "Usage of 'stonesoup.dataassociator.mfa' requires that the optional "
        "package dependency 'ortools' is installed.") \
        from error

from ._init import Hyp, HypInfo, TimeStepIndices

# Terminate if gap between primal and dual costs is less than this
GAP_THRESHOLD = 0.02
# Maximum number of iterations for the algorithm
MAX_ITERATION_COUNT = 10


def _get2dCostMatrix(c_hat, time_step_indices: TimeStepIndices, track_count, measurement_count):
    """Returns the cost matrix"""

    # Compute track-measurement cost for single scan problem from hypothesis
    # cost, including null assignment costs
    # Construct track to measurement assignment matrix at scan k
    cost = np.full((track_count, measurement_count), np.inf)
    nullCost = np.full((track_count,), np.inf)
    # store index of the single target hypothesis with the minimum cost for
    # each track and measurement
    idxCost = np.full((track_count, measurement_count), -1, dtype=np.int32)
    idxNullCost = np.full((track_count,), -1, dtype=np.int32)
    for track_id, trackNull_index_for_track in enumerate(time_step_indices.trackNull_index):
        if len(trackNull_index_for_track):
            min_index = c_hat[trackNull_index_for_track].argmin()
            min_index_all_c_hat = trackNull_index_for_track[min_index]
            nullCost[track_id] = c_hat[min_index_all_c_hat]
            idxNullCost[track_id] = min_index_all_c_hat
        # find single target hypotheses in track i that use this
        # measurement if found, find the single target hypothesis with
        # the minimum cost, and record its index
        for measurement in range(measurement_count):
            measTrack_index_for_meas = time_step_indices.measTrack_index[measurement, track_id]
            if len(measTrack_index_for_meas):
                min_index = c_hat[measTrack_index_for_meas].argmin()
                min_index_all_c_hat = measTrack_index_for_meas[min_index]
                cost[track_id, measurement] = c_hat[min_index_all_c_hat]
                idxCost[track_id, measurement] = min_index_all_c_hat

    # Create cost matrix for nulls with null costs on diagonal and inf
    # elsewhere (so we can have any number of null assignments)
    nullCostMatrix = np.full((track_count, track_count), np.inf, dtype=np.float64)
    # Pick out diagonal entries (like np.diagonal, but this is a writable view)
    nullCostMatrix.ravel()[::track_count + 1] = nullCost

    fullCostMatrix = np.concatenate((cost, nullCostMatrix), axis=1)
    return fullCostMatrix, idxCost, idxNullCost


def _getSuboptimalSolutionForSubproblem(
    delta_k: np.array, all_costs: np.array, time_step_indices: TimeStepIndices, slide_window: int
):
    # Get hypothesis costs including Lagrangians
    c_hat = all_costs / slide_window + delta_k

    meas_count = time_step_indices.measurement_count
    track_count = len(time_step_indices.trackNull_index)

    fullCostMatrix, idxCost, idxNullCost = _get2dCostMatrix(
        c_hat, time_step_indices, track_count, meas_count
    )

    # Perform the actual assignment (Hungarian algorithm)
    # The column count (= meas_count + track_count) > row count (= track_count)
    # so it is guaranteed that every row (track) is assigned
    # row_ind is therefore simply np.array(range(track_count))
    # assignments is index of measurement for each track (or >meas_count if unassigned)
    row_ind, assignments = linear_sum_assignment(fullCostMatrix)

    # Assign hypothesis indicators associated with chosen assignments to true
    assignedHypotheses = np.zeros((all_costs.size,), dtype=np.bool_)
    for track_index, assigned_measurement_index in enumerate(assignments):
        if assigned_measurement_index < meas_count:
            assert idxCost[track_index, assigned_measurement_index] >= 0
            assignedHypotheses[idxCost[track_index, assigned_measurement_index]] = True
        else:
            assert idxNullCost[track_index] >= 0
            assignedHypotheses[idxNullCost[track_index]] = True

    return c_hat, assignedHypotheses


def _getPrimalSolution(u_hat_mean, Amatrix, hypothesisCosts):
    """Get a primal (feasible but not necessarily optimal) solution from the dual solution.

    decompose problem into two parts - one where the dual subproblem solutions agree and the
    remaining part to be solved using OR tools solver.
    """

    # find partial primal solution without conflicts
    idx_selectedHyps = u_hat_mean == 1

    idx_unselectedHyps = np.logical_not(idx_selectedHyps)

    # Tracks and measurements not used by the partial solution (ordered by
    # tracks first, then measurements for each scan)
    idx_uncertainTracksMeas = np.sum(Amatrix[:, idx_selectedHyps], axis=1).astype(np.int32) == 0

    # If a track or measurement used by the partial solution, remove it from
    # the problem to be solved by integer linear programming
    for i, val in enumerate(idx_uncertainTracksMeas):
        if not val:
            idx_unselectedHyps[Amatrix[i, :] == 1] = False

    # Solve remaining problem using OR tools solver to find a feasible solution
    A_uncertain = Amatrix[:, idx_unselectedHyps][idx_uncertainTracksMeas, :]
    c_uncertain = hypothesisCosts[idx_unselectedHyps] * 1000000

    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver("SCIP")

    # Add constraints
    vars = [solver.BoolVar(str(i)) for i in range(c_uncertain.size)]
    for A_uncertain_row in A_uncertain:
        selected_vars = [var for var, A_val in zip(vars, A_uncertain_row) if A_val]
        solver.Add(solver.Sum(selected_vars) == 1)

    # Run the solver
    solver.Minimize(solver.Sum([c * var for var, c in zip(vars, c_uncertain)]))
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):  # pragma: no cover
        raise RuntimeError("Infeasible primal problem")

    uprimal_uncertain = [bool(v.solution_value()) for v in vars]

    # Get solution to full problem by combining the partial and linear programming solutions
    u_primal_hat = u_hat_mean == 1
    u_primal_hat[idx_unselectedHyps] = uprimal_uncertain

    # Obtain primal cost
    primal_cost_hat = hypothesisCosts @ u_primal_hat

    return u_primal_hat, primal_cost_hat


@dataclass
class AlgorithmState:
    delta: np.array
    u_hat: np.array
    bestPrimalCost: float
    uprimal: np.array
    should_break: bool

    @staticmethod
    def initialise(slide_window: int, hyp_count: int):
        """Initialises the algorithm state, for use before the algorithm runs"""
        return AlgorithmState(
            # Lagrange multiplier \delta is initialised with 0
            delta=np.zeros((slide_window, hyp_count), dtype=np.float64),
            # subproblem solutions
            u_hat=np.zeros((slide_window, hyp_count), dtype=np.bool_),
            # store the best feasible primal cost obtained so far (upper bound)
            bestPrimalCost=np.inf,
            # the best primal solution (with cost=bestPrimalCost)
            uprimal=np.zeros((hyp_count,), dtype=np.bool_),
            # whether the main algorithm loop should now stop
            should_break=False
        )

    def get_best_hypothesis_indices(self):
        """Extracts the results of the algorithm: the indices of the best hypotheses."""
        return np.nonzero(self.uprimal == 1)[0]


def algorithm_step(
    state: AlgorithmState,
    hyp_info: HypInfo,
):
    slide_window = state.delta.shape[0]
    # get suboptimal solution for each subproblem
    sub_dual_cost = np.zeros((slide_window,), dtype=np.float64)
    for k in range(slide_window):
        c_hat, assignedHypotheses = _getSuboptimalSolutionForSubproblem(
            state.delta[k], hyp_info.all_costs, hyp_info.time_step_indices[k], slide_window
        )
        state.u_hat[k] = assignedHypotheses
        sub_dual_cost[k] = c_hat.T @ assignedHypotheses

    # Get proportion of assignments over the measurement scans for each hypothesis
    u_hat_mean = state.u_hat.mean(axis=0)

    # If mean is zero or one then all scans agree
    # All the subproblem solutions are equal means we have found the optimal solution
    if np.all(np.logical_or(u_hat_mean == 0, u_hat_mean == 1)):
        state.uprimal = state.u_hat[0]
        state.should_break = True
        return

    # Calculate dual cost
    dual_cost_hat = np.sum(sub_dual_cost)

    # Get primal solution
    u_primal_hat, primal_cost_hat = _getPrimalSolution(
        u_hat_mean, hyp_info.constraint_matrix, hyp_info.all_costs
    )

    # Replace best primal cost
    if primal_cost_hat < state.bestPrimalCost:
        state.bestPrimalCost = primal_cost_hat
        state.uprimal = u_primal_hat
    else:
        # Jump out of the loop if the best primal cost obtained does not increase
        state.should_break = True
        return

    # Jump out of the loop if the gap is small enough
    gap = (state.bestPrimalCost - dual_cost_hat) / state.bestPrimalCost
    if gap < GAP_THRESHOLD:
        state.should_break = True
        return

    # Calculate step size used in subgradient methods
    # Calculate subgradient
    g = state.u_hat - u_hat_mean
    # calculate step size used in subgradient method
    stepSize = (state.bestPrimalCost - dual_cost_hat)/(np.linalg.norm(g)**2)
    # update Lagrange multiplier
    state.delta = state.delta + stepSize * g


def prune_hypotheses(best_hypotheses: List[Hyp], hyps: List[Hyp]) -> Dict[int, List[Hyp]]:
    """n-scan pruning.

    This routine discards any hypotheses that disagree with the best hypothesis before the
    sliding window as it will be in the next iteration. For example, if the sliding window is
    of length 3, then for each track this discards hypotheses that disagree with the best
    hypothesis for that track except for the last 2 measurements.

    Assuming that n-scan pruning is done every time step, it suffices to compare only the first
    measurement in the sliding window.

    The result is returned as a map from track ID to the hypotheses that should be kept for that
    track.
    """

    result = {}
    for h in hyps:
        if h.isDummy:
            continue
        hyp_matches_best_at_start = (
            h.measHistorySlideWindow[0] == best_hypotheses[h.trackID].measHistorySlideWindow[0]
        )
        if hyp_matches_best_at_start:
            if h.trackID not in result:
                result[h.trackID] = []
            result[h.trackID].append(h)
    return result
