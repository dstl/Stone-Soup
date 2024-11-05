import numpy as np
from stonesoup.types.array import StateVector, CovarianceMatrix

def tile_with_circles(minpos, maxpos, numx, numy):
    """
    Return centres and radius of a grid of circles with numx columns and numy rows which tile the region defined
    by minpos and maxpos
    """
    field_size = maxpos - minpos
    grid_size = max(field_size[0] / numx, field_size[1] / numy)
    radius = grid_size / np.sqrt(2);
    grid_start = (field_size - np.array([numx - 1, numy - 1]) * grid_size) / 2.0 + minpos
    centres = []
    for i in range(numx):
        for j in range(numy):
            centres.append(grid_start + np.array([i, j]) * grid_size)
    return centres, radius


def merge_position_and_velocity(position, velocity, statedim, position_mapping, velocity_mapping):
    """
    Create a state by merging a position and a velocity
    """
    state = StateVector(np.zeros((1, statedim)))
    state[position_mapping, :] = StateVector(position)
    state[velocity_mapping, :] = StateVector(velocity)
    return state


def merge_position_and_velocity_covariance(poscov, velcov, statedim, position_mapping, velocity_mapping):
    """
    Create a state covariance by merging a position covariance and a velocity covariance
    """
    covariance = CovarianceMatrix(np.zeros((statedim, statedim)))
    covariance[np.ix_(position_mapping, position_mapping)] = poscov
    covariance[np.ix_(velocity_mapping, velocity_mapping)] = velcov
    return covariance


def fit_normal_to_uniform(minval, maxval):
    """
    """
    mean = StateVector((minval + maxval)/2.0)
    cov = CovarianceMatrix(np.diag(np.power(maxval - minval, 2.0)/12.0))
    return mean, cov


def to_single_state(tracks, statedim):
    """
    Convert a set of tracks with two-state vectors to a set of tracks with one-state vectors
    """
    new_tracks = set()
    for track in tracks:
        states = []
        for state in track.states:
            if isinstance(state, Update):
                new_state = GaussianStateUpdate(state.state_vector[-statedim:], state.covar[-statedim:, -statedim:],
                                                hypothesis=state.hypothesis,
                                                timestamp=state.timestamp)
            else:
                new_state = GaussianState(state.state_vector[-statedim:], state.covar[-statedim:, -statedim:],
                                          timestamp=state.timestamp)
            states.append(new_state)
        new_tracks.add(Track(id=track.id, states=states))
    return new_tracks
