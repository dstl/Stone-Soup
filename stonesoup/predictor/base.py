# -*- coding: utf-8 -*-
"""Base classes for Stone Soup Predictor interface"""
from abc import abstractmethod

from ..base import Base, Property
from ..models.transition import TransitionModel
from ..models.control import ControlModel
from types import FunctionType
from ..updater.base import Updater
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


class Predictor(Base):
    r"""Predictor base class

    A predictor is used to predict a new :class:`~.State` given a prior
    :class:`~.State` and a :class:`~.TransitionModel`. In addition, a
    :class:`~.ControlModel` may be used to model an external influence on the
    state.

    .. math::

        \mathbf{x}_{k|k-1} = f_k(\mathbf{x}_{k-1}, \mathbf{\nu}_k) +
        b_k(\mathbf{u}_k, \mathbf{\eta}_k)

    where :math:`\mathbf{x}_{k-1}` is the prior state,
    :math:`f_k(\mathbf{x}_{k-1})` is the transition function,
    :math:`\mathbf{u}_k` the control vector, :math:`b_k(\mathbf{u}_k)` the
    control input and :math:`\mathbf{\nu}_k` and :math:`\mathbf{\eta}_k` the
    transition and control model noise respectively.
    """

    transition_model: TransitionModel = Property(doc="transition model")
    control_model: ControlModel = Property(default=None, doc="control model")
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
        raise NotImplementedError
