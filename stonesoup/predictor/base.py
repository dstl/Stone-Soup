# -*- coding: utf-8 -*-
"""Base classes for Stone Soup Predictor interface"""
from abc import abstractmethod

from ..base import Base, Property
from ..models.transition import TransitionModel
from ..models.control import ControlModel


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
