# -*- coding: utf-8 -*-
from typing import Sequence

from stonesoup.base import Property
from stonesoup.predictor import Predictor
from stonesoup.predictor._utils import predict_lru_cache
from stonesoup.types.state import CompositeState


class CompositePredictor(Predictor):
    """A composition of multiple sub-predictors"""
    sub_predictors: Sequence[Predictor] = Property(doc="A sequence of sub-predictors")

    @property
    def transition_model(self):
        raise NotImplementedError("A composition of predictors have no defined transition model")

    @predict_lru_cache()
    def predict(self, prior, timestamp=None, **kwargs):
        r"""The predict function

        Parameters
        ----------
        prior : :class:`~.State`
            :math:`\mathbf{x}_{k-1}` assumed to be :class:`~.CompositeState` type, representing an
            object existing in a composite state space
        timestamp : :class:`datetime.datetime`, optional
            :math:`k`
        **kwargs :
            These are passed, via :meth:`~.KalmanFilter.transition_function` to
            :meth:`~.LinearGaussianTransitionModel.matrix`

        Returns
        -------
        : :class:`~.CompositeState`
            :math:`\mathbf{x}_{k|k-1}`, the predicted composite state
        """

        if not isinstance(prior, CompositeState):
            raise ValueError("CombinedPredictor can only be used with CompositeState types")
        if len(prior.sub_states) != len(self.sub_predictors):
            raise ValueError(
                "CompositeState must be composed of same number of sub-states as sub-predictors")

        new_states = []

        for predictor, state in zip(self.sub_predictors, prior.sub_states):
            predicted = predictor.predict(state, timestamp, **kwargs)
            new_states.append(predicted)

        return CompositeState(inner_states=new_states)

    def __getitem__(self, index):
        return self.sub_predictors.__getitem__(index)

    def __setitem__(self, index, value):
        return self.sub_predictors.__setitem__(index, value)

    def __delitem__(self, index):
        return self.sub_predictors.__delitem__(index)

    def __iter__(self):
        return iter(self.sub_predictors)

    def __len__(self):
        return self.sub_predictors.__len__()

    def insert(self, index, value):
        inserted_state = self.sub_predictors.insert(index, value)
        return inserted_state

    def append(self, value):
        """Add value at end of :attr:`sub_predictors`.

        Parameters
        ----------
        value: :class:`~.Predictor`
            A Predictor to be added at the end of :attr:`sub_predictors`.
        """
        self.sub_predictors.append(value)
