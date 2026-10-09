from collections.abc import Callable
from functools import lru_cache

from ..base import Property
from ..smoother import Smoother
from ..types.hypothesis import SingleHypothesis
from ..types.multihypothesis import MultipleHypothesis
from ..types.prediction import Prediction
from ..types.state import StateMutableSequence
from ..types.update import Update
from . import Predictor
from ._utils import predict_lru_cache


def get_hypothesis(
        hypothesis: SingleHypothesis | MultipleHypothesis,
        f: Callable[[SingleHypothesis], bool] = lambda x: True) -> SingleHypothesis | None:
    """Get single hypothesis

    Parameters
    ----------
    hypothesis: :class:`~.SingleHypothesis` or :class:`~.MultipleHypothesis`
        Input hypothesis which will be returned in case of being single hypothesis, or
        first single hypothesis from multiple hypothesis filtered by :attr:`f`.
    f: callable
        Function which takes a hypothesis and returns boolean. Default always
        returns true.

    Returns
    -------
    : :class:`~.SingleHypothesis`
        First single hypothesis from multiple hypothesis filtered by :attr:`f`.
    """
    if isinstance(hypothesis, MultipleHypothesis):
        return next((hyp for hyp in hypothesis if f(hyp)), None)
    else:
        return hypothesis


def get_prior_state(state):
    """Get the state preceding the given state

    If the state is an instance of `Prediction`, this is the `prior` attribute.
    If the state is an instance of `Update`, this is the associated hypothesis's
    prediction's `prior`. Otherwise, there is no prior state.

    Parameters
    ----------
    state : :class:`~.State`
        The state to get the prior state of.

    Returns
    -------
    : :class:`~.State` or None
        The prior state, or ``None`` if there is no prior state.
    """
    if isinstance(state, Prediction):
        return state.prior
    elif isinstance(state, Update):
        return getattr(get_hypothesis(state.hypothesis).prediction, 'prior', None)
    else:
        return None


def get_past_states(state):
    """
    Yields the sequence of past states leading up to the given state.

    Traverses backwards through a chain of states, yielding each state in the chain,
    following :func:`get_prior_state`. Stops when there are no more prior states.

    Parameters
    ----------
    state : object
        The initial state to start traversing from.

    Yields
    ------
    : :class:`~stonesoup.types.state.State`
    """
    while state is not None:
        yield state
        state = get_prior_state(state)


class OOSPredictorWrapper(Predictor):
    """
    OOSPredictorWrapper is a wrapper class for handling out-of-sequence (OOS) prediction scenarios.

    This class enables prediction in cases where the sequence of states is not strictly
    chronological, by supporting both forward and backward prediction, as well as optional
    smoothing using future information.
    """
    transition_model = None
    predictor: Predictor = Property(
        doc="Primary predictor used for prediction forward in time")
    backward_predictor: Predictor = Property(
        default=None,
        doc="Predictor used when predicting backward from the earliest possible state. "
            "Default ``None``, where a ValueError will be raised if attempting to predict "
            "backwards")
    smoother: Smoother = Property(
        default=None,
        doc="Smoother used to improve predictions in out of sequence states using \"future\" "
            "information. Default ``None``, where smoother is not used")

    @staticmethod
    @lru_cache
    def _predict_states(prior, timestamp):
        states = []
        for state in get_past_states(prior):
            states.append(state)
            if timestamp >= state.timestamp:
                break
        return states

    @predict_lru_cache()
    def predict(self, prior, timestamp, **kwargs):
        states = self._predict_states(prior, timestamp)
        new_prior = states[-1]

        if len(states) > 1 and self.smoother:
            state_seq = StateMutableSequence(
                [s for s in reversed(states[:-1]) if isinstance(s, Update)])
            for state in get_past_states(states[-1]):
                if isinstance(state, Update):
                    state_seq.insert(0, state)
                    if len(state_seq) > 1:
                        smooth_states = self.smoother.smooth(state_seq)
                        new_prior = smooth_states[0]
                    break

        if new_prior.timestamp <= timestamp:
            prediction = self.predictor.predict(new_prior, timestamp, **kwargs)
        elif self.backward_predictor:  # states[-1].timestamp > timestamp
            prediction = self.backward_predictor.predict(new_prior, timestamp, **kwargs)
        else:  # states[-1].timestamp > timestamp
            raise ValueError(f"Cannot predict: {timestamp} < {states[-1].timestamp}")

        prediction.prior = states[-1]
        prediction.prior.orig_prior = prior
        return prediction
