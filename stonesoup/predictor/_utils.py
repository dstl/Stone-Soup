import functools

from ..types.state import StateMutableSequence


def predict_lru_cache(*args, **kwargs):
    """LRU Cache decorator for :meth:`~.Predictor.predict` methods

    This ensures the current state is extracted for the LRU cache to function
    correctly, as caching should be on current state, not on mutable sequence.

    This should function same as :class:`functools.lru_cache` otherwise.
    """

    lru_cache_instance = functools.lru_cache(*args, **kwargs)

    def decorator(func):
        func = lru_cache_instance(func)

        @functools.wraps(func)
        def predict(self, prior, *args, **kwargs):
            if isinstance(prior, StateMutableSequence):
                prior = prior.state
            return func(self, prior, *args, **kwargs)
        return predict
    return decorator
