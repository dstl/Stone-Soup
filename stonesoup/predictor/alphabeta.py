# -*- coding: utf-8 -*-
"""alpha-beta for Stone Soup Predictor interface"""
import numpy as np

from .base import Predictor
from ..base import Property
from ..types.prediction import Prediction, StatePrediction
from ..models.transition.linear import ConstantVelocity


class AlphaBetaPredictor(Predictor):
    r"""Alpha-beta predictor

    The \alpha-\beta predictor assumes that the state vector is composed of 'position' and
    'velocity' components.

    .. math::

        \mathbf{x}_{k} = \mathbf{x}_{k-1} + \Delta T \mathbf{v}_{k-1}, \

    where :math:`\mathbf{x}_{k-1}` is the prior state, :math:`\mathbf{v}_{k-1}` is the prior
    first derivative of :math:`\mathbf{x}_{k-1}` with respect to time and :math:`\Delta T` is the
    time interval. The placement of the position and velocity within the state vector is made
    explicit by the `position_map` and `velocity_map` binary vectors which must be of the same
    size and non-overlapping.

    No control model is assumed.

    This class assumes the velocity is in units of the length per unit second. If different units
    are required, scale the prior appropriately,

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if type(self.transition_model) is not ConstantVelocity:
            raise TypeError("Transition model must be constant velocity")

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
        : :class:`~.Prediction`
            State prediction
        """
        try:
            time_interval = timestamp - prior.timestamp
        except TypeError as error:
            # TypeError: (timestamp or prior.timestamp) is None
            raise ValueError('A time stamp is required for the alpha-beta predictor') from error

        new_state_vector = self.transition_model.function(prior, noise=False,
                                                          time_interval=time_interval)

        return Prediction.from_state(prior, new_state_vector, timestamp=timestamp,
                                     transition_model=self.transition_model)
