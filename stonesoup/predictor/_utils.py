import functools

from ..functions.caching import instance_lru_cache
from ..types.state import State, StateMutableSequence


def predict_lru_cache(maxsize=128, typed=False):
    """LRU Cache decorator for :meth:`~.Predictor.predict` methods

    This ensures the current state is extracted for the LRU cache to function
    correctly, as caching should be on current state, not on mutable sequence.

    The cache is stored per predictor instance so that predictor objects can be
    garbage collected (see issue #801). Behaviour otherwise matches
    :func:`functools.lru_cache`.
    """

    def decorator(func):
        cached = instance_lru_cache(maxsize=maxsize, typed=typed)(func)

        @functools.wraps(func)
        def predict(self, prior, *args, **kwargs):
            if isinstance(prior, StateMutableSequence) and not isinstance(prior, State):
                prior = prior.state
            return cached(self, prior, *args, **kwargs)

        predict.cache_info = cached.cache_info
        predict.cache_clear = cached.cache_clear
        return predict

    return decorator
