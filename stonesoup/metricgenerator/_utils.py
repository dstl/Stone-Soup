# -*- coding: utf-8 -*-
from functools import lru_cache, wraps


def clearable_lru_cache():
    """Cache decorator that allows keeping track of which methods are decorated.
    Requires a class utilising this to have a `cache` boolean property, determining whether the
    decorated method is cached or not.
    """

    def cache_decorator(func):
        @wraps(func)
        def cache_factory(self, *args, **kwargs):
            if self.cache:
                instance_cache = lru_cache(*self.cache_args, **self.cache_kwargs)(func)
                instance_cache = instance_cache.__get__(self, self.__class__)
                setattr(self, func.__name__, instance_cache)

                self.cached_functions.append(instance_cache)

                return instance_cache(*args, **kwargs)
            else:
                return func(self, *args, **kwargs)

        return cache_factory

    return cache_decorator
