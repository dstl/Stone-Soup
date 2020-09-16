# -*- coding: utf-8 -*-
"""Contains collection of error based deleters"""
import numpy as np

from ..base import Property
from .base import Deleter


class CovarianceBasedDeleter(Deleter):
    """ Track deleter based on covariance matrix size.

    Deletes tracks whose state covariance matrix (more specifically its trace)
    exceeds a given threshold.
    """

    covar_trace_thresh: float = Property(doc="Covariance matrix trace threshold")

    def check_for_deletion(self, track, **kwargs):
        """Check if a given track should be deleted

        A track is flagged for deletion if the trace of its state covariance
        matrix is higher than :py:attr:`~covar_trace_thresh`.

        Parameters
        ----------
        track : :class:`stonesoup.types.Track`
            A track object to be checked for deletion.

        Returns
        -------
        : :class:`bool`
            ``True`` if track should be deleted, ``False`` otherwise.
        """

        track_covar_trace = np.trace(track.state.covar)

        if(track_covar_trace > self.covar_trace_thresh):
            return True
        return False
