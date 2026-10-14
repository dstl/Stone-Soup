import numpy as np

from ..base import Property
from .base import Type


class TransitionMatrix(Type):
    """
    Transition Probability Matrix (TPM) for use in model reduction and model augmentation.

    The TPM is used to determine the probability of transitioning from one model to another
    based on the model history of a state. The TPM can be defined for different lengths of model
    history, allowing for more complex transition dynamics to be captured. The class provides a
    method to retrieve the appropriate transition matrix based on the model history of a given
    state.
    """
    transition_matrix: np.ndarray = Property(
        doc="Transition Probability matrix.")
    num_components: int = Property(default=0, doc="Number of states.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure transition_matrix is in the right format (rows sum to 1)
        self.transition_matrix = np.atleast_2d(self.transition_matrix)

        if self.num_components > 1:
            if self.num_components != self.transition_matrix.shape[1]:
                raise ValueError("num_components (%d) is not compatible with transition_matrix "
                                 "number of columns (%d)." % (self.num_components,
                                                              self.transition_matrix.shape[1]))

            num_rows = self.transition_matrix.shape[0]
            if num_rows > 1:
                history_length = int(round(np.log(num_rows) / np.log(self.num_components)))
                if self.num_components ** history_length != num_rows:
                    raise ValueError("transition_matrix number of rows (%d) is not compatible "
                                     "with num_components (%d). Rows must equal "
                                     "num_components**history_length." % (num_rows,
                                                                          self.num_components,))

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
        if history_length <= 1 and self.num_components > 1:
            history_length += 1
        # Clamp the index to the valid range of available transition matrices
        max_key = max(self.transition_matrices.keys())
        index = max(0, min(history_length - 1, max_key))
        return self.transition_matrices[index]

    @property
    def get_all_transition_matrices(self):
        return self.transition_matrices

    def _property_transition_matrix(self):
        pass

    def _property_matrix_(self):
        pass
