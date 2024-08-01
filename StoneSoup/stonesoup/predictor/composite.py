from typing import Sequence

from ..base import Property
from ..predictor import Predictor
from ..predictor._utils import predict_lru_cache
from ..types.prediction import CompositePrediction
from ..types.state import CompositeState


class CompositePredictor(Predictor):
    """Composite predictor type

    A composition of ordered sub-predictors (:class:`~.Predictor`). Independently predicts each
    sub-state of a :class:`CompositeState` forward using a corresponding sub-predictor.
    """

    sub_predictors: Sequence[Predictor] = Property(
        doc="Sequence of sub-predictors comprising the composite predictor. Must not be empty.")

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if not isinstance(self.sub_predictors, Sequence):
            raise ValueError(f"Sub-predictors must be defined as an ordered list, not "
                             f"{type(self.sub_predictors)}")

        if len(self.sub_predictors) == 0:
            raise ValueError("Cannot create an empty composite predictor")

        if any(not isinstance(sub_predictor, Predictor) for sub_predictor in self.sub_predictors):
            raise ValueError("All sub-predictors must be a Predictor type")

    @property
    def transition_model(self):
        raise NotImplementedError("A composition of predictors has no defined transition model")

    @predict_lru_cache()
    def predict(self, prior, timestamp=None, **kwargs):
        r"""The predict function

        Parameters
        ----------
        prior : :class:`~.CompositeState`
            The composite state of an object to be predicted forwards
        timestamp : :class:`datetime.datetime`, optional
            :math:`k`
        **kwargs :
            These are passed to each sub-predictor's prediction method

        Returns
        -------
        : :class:`~.CompositeState`
            The predicted composite state
        """

        if not isinstance(prior, CompositeState):
            raise ValueError("CompositePredictor can only predict forward CompositeState types")

        if len(prior) != len(self):
            raise ValueError(f"Mismatch in number of prior sub-states {len(prior)} and number "
                             f"of sub-predictors {len(self)}")

        prediction_sub_states = []

        for sub_predictor, sub_state in zip(self.sub_predictors, prior.sub_states):
            sub_prediction = sub_predictor.predict(prior=sub_state, timestamp=timestamp, **kwargs)
            prediction_sub_states.append(sub_prediction)

        return CompositePrediction(sub_states=prediction_sub_states)

    def __contains__(self, item):
        return self.sub_predictors.__contains__(item)

    def __getitem__(self, index):
        """Can be indexed as a list, or sliced, in which case a new composite predictor will be
        created from the sub-list of sub-predictors."""
        if isinstance(index, slice):
            return self.__class__(self.sub_predictors.__getitem__(index))
        return self.sub_predictors.__getitem__(index)

    def __iter__(self):
        return self.sub_predictors.__iter__()

    def __len__(self):
        return self.sub_predictors.__len__()
