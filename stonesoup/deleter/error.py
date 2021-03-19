# -*- coding: utf-8 -*-
"""Contains collection of error based deleters"""
from typing import Sequence

import numpy as np

from ..base import Property
from .base import Deleter


class CovarianceBasedDeleter(Deleter):
    """ Track deleter based on covariance matrix size.

    Deletes tracks whose state covariance matrix (more specifically its trace)
    exceeds a given threshold.
    """

    covar_trace_thresh: float = Property(doc="Covariance matrix trace threshold")
    mapping: Sequence = Property(default=None,
                                 doc="Track state vector indices whose corresponding covariances' "
                                     "sum is to be considered. Defaults to None, whereby the "
                                     "entire track covariance trace is considered.")

    def check_for_deletion(self, track, **kwargs):
        """Check if a given track should be deleted

        A track is flagged for deletion if the trace of its state covariance
        matrix is higher than :py:attr:`~covar_trace_thresh`.

        Parameters
        ----------
        track : Track
            A track object to be checked for deletion.

        Returns
        -------
        bool
            `True` if track should be deleted, `False` otherwise.
        """

        diagonals = np.diag(track.state.covar)
        if self.mapping:
            track_covar_trace = np.sum(diagonals[self.mapping])
        else:
            track_covar_trace = np.sum(diagonals)

        if track_covar_trace > self.covar_trace_thresh:
            return True
        return False
