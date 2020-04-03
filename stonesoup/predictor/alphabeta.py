# -*- coding: utf-8 -*-
"""alpha-beta for Stone Soup Predictor interface"""
from .base import Predictor
from ..types.prediction import StatePrediction


class AlphaBetaPredictor(Predictor):
    r"""Alpha-beta predictor which inherits from base

    A predictor is used to predict a new :class:`~.State` given a prior
    :class:`~.State`.

    .. math::

        \hat{\mathbf{x}}_{k} = \hat{\mathbf{x}}_{k-1} + \Delta T \hat{\mathbf{v}}_{k-1}

    where :math:`\mathbf{x}_{k-1}` is the prior state,
    :math:\Delta T is the time interval.

    """

    transition_model = None
    control_model = None

    def predict(self, prior, timestamp=None, **kwargs):
        """The prediction function itself

        Parameters
        ----------
        prior : :class:`~.State`
            The prior state
        timestamp : :class:`datetime.datetime`, optional
            Time at which the prediction is made (used by the transition
            model)

        Returns
        -------
        : :class:`~.StatePrediction`
            State prediction
        """
        try:
            time_interval = timestamp - prior.timestamp
        except TypeError as error:
            # TypeError: (timestamp or prior.timestamp) is None
            raise ValueError('Time stamps are required') from error

        new_state_vector = prior.state_vector.copy()
        for n in range(0, prior.ndim, 2):
            new_state_vector[n, 0] += new_state_vector[n+1, 0] * (
                time_interval.total_seconds())

        return StatePrediction(new_state_vector, timestamp=timestamp)
