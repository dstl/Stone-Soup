# -*- coding: utf-8 -*-
"""alpha-beta for Stone Soup Predictor interface"""
from .base import Predictor
from ..types.prediction import StatePrediction


class AlphaBetaPredictor(Predictor):
    r"""Alpha-beta predictor which inherits from base

    A predictor is used to predict a new :class:`~.State` given a prior
    :class:`~.State`. and a transition model

    .. math::

        f_k( \mathbf{x}_{k-1}) = \mathbf{x}_{k-1} + \Delta T \mathbf{v}_{k-1} \ \mathrm{and} \
        \mathbf{\nu}_k \sim \mathcal{N}(0,Q_k)

    where :math:`\mathbf{x}_{k-1}` is the prior state, :math:`\mathbf{v}_{k-1}` is the prior
    first derivative of :math:`\mathbf{x}_{k-1}` vector and :math:`\Delta T` is the time interval
    between measurements.

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
