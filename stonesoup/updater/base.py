# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Base, Property
from ..models.measurement import MeasurementModel
from types import FunctionType
import functools


def null_convert(state):
    """
    Routine to do a null conversion on the Gaussian state
    Parameters
    ----------
    state: :class:'~GaussianState'
        The input state.

    Returns
    -------
    :class:'~GaussianState'
    """
    return state


# The decorator to call conversion routines before and after a function
def prepost(fn):
    # The new function the decorator returns
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        self = args[0]
        state = args[1]
        state = self.convert2local_state(state)
        out = fn(self, state, **kwargs)
        out = self.convert2common_state(out)
        return out
    return wrapper


class Updater(Base):
    r"""Updater base class

    An updater is used to update the predicted state, utilising a measurement
    and a :class:`~.MeasurementModel`.  The general observation model is

    .. math::

        \mathbf{z} = h(\mathbf{x}, \mathbf{\sigma})

    where :math:`\mathbf{x}` is the state, :math:`\mathbf{\sigma}`, the
    measurement noise and :math:`\mathbf{z}` the resulting measurement.

    """

    measurement_model = Property(MeasurementModel, doc="measurement model")
    convert2common_state = Property(
            FunctionType,
            default=null_convert,
            doc="Routine to convert from an internal Gaussian state"
            "to common Gaussian state")
    convert2local_state = Property(
            FunctionType,
            default=null_convert,
            doc="Routine to convert from a common Gaussian state"
            "to the internal Gaussian state required for this predictor")

    @abstractmethod
    def predict_measurement(
            self, state_prediction, measurement_model=None, **kwargs):
        """Get measurement prediction from state prediction

        Parameters
        ----------
        state_prediction : :class:`~.StatePrediction`
            The state prediction
        measurement_model: :class:`~.MeasurementModel`, optional
            The measurement model used to generate the measurement prediction.
            Should be used in cases where the measurement model is dependent
            on the received measurement. The default is `None`, in which case
            the updater will use the measurement model specified on
            initialisation

        Returns
        -------
        : :class:`~.MeasurementPrediction`
            The predicted measurement
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, hypothesis, **kwargs):
        """Update state using prediction and measurement.

        Parameters
        ----------
        hypothesis : :class:`~.Hypothesis`
            Hypothesis with predicted state and associated detection used for
            updating.

        Returns
        -------
        : :class:`~.State`
            The state posterior
        """
        raise NotImplementedError
