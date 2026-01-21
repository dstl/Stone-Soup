import numpy as np

from ..base import Base, Property


class TransitionMatrix(Base):
    transition_matrix: np.ndarray = Property(
        doc="Transition Probability matrix.")
    num_states: int = Property(default=0, doc="Number of states.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure transition_matrix is in the right format (rows sum to 1)
        self.transition_matrix = np.atleast_2d(self.transition_matrix)
        transition_matrix = (self.transition_matrix /
                             np.tile(np.sum(self.transition_matrix, axis=1),
                                     (self.transition_matrix.shape[1], 1)).T)
        dims = transition_matrix.shape
        num_histories = dims[0]
        num_states = dims[1]
        hs = dict()
        if num_histories > 1:
            h_index = int(np.log(num_histories) / np.log(num_states))
        else:
            h_index = 0
        hs[h_index] = transition_matrix
        while num_histories > 1:
            h = int(num_histories / num_states)
            h_index -= 1
            temp_matrix = np.empty((0, dims[1]), float)
            for i in range(h):
                temp = transition_matrix[[x * h + i for x in range(num_states)]].sum(axis=0)
                temp = temp / sum(temp)
                temp_matrix = np.append(temp_matrix, [temp], axis=0)
            hs[h_index] = temp_matrix
            transition_matrix = temp_matrix
            num_histories = h
        self.transition_matrices = hs

    def __getitem__(self, state):
        history_length = len(state.model_histories)
        if history_length <= 1 and self.num_states > 1:
            history_length += 1
        return self.transition_matrices[np.max([0, history_length - 1])]

    @property
    def get_all_transition_matrices(self):
        return self.transition_matrices

    def _property_transition_matrix(self):
        pass

    def _property_matrix_(self):
        pass
