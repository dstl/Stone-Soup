# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Base, Property
from ..models.transition import TransitionModel


class Smoother(Base):
    """Smoother Base Class

    (Fixed interval) Smoothers in general are used to infer a state, or series of states,
    :math:`\mathbf{x}_k` from measurements :math:`\mathbf{z}_{1:K}` where :math:`k < K`.

    The calculation is forward-backward in nature. The forward algorithm is "standard" filtering,
    provided by other Stone Soup components. The Smoother's input is therefore a :class:`~.Track`
    (created by whatever means) The :meth:`smooth` function undertakes the backward algorithm.

    """

    transition_model: TransitionModel = Property(default=None, doc="Transition Model.")

    @abstractmethod
    def smooth(self, *args, **kwargs):
        raise NotImplementedError

    #@abstractmethod
    #def track_smooth(self, *args, **kwargs):
    #    raise NotImplementedError
