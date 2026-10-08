"""Caching helpers for Stone Soup components."""
from __future__ import annotations

import functools
import weakref


def instance_lru_cache(maxsize: int | None = 128, typed: bool = False):
    """Per-instance :func:`functools.lru_cache` for methods.

    Unlike decorating an instance method with :func:`functools.lru_cache`
    directly, the cache is stored on the instance. That avoids the well-known
    reference cycle where a class-level cache keeps instances alive after they
    would otherwise be garbage collected (see issue #801).

    The wrapped method must still be called with hashable arguments (aside from
    ``self``), as with :func:`functools.lru_cache`.
    """

    def decorator(method):
        cache_attr = f"_lru_cache_{method.__name__}"

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            cache = self.__dict__.get(cache_attr)
            if cache is None:
                self_ref = weakref.ref(self)

                @functools.lru_cache(maxsize=maxsize, typed=typed)
                def cache(*cache_args, **cache_kwargs):
                    obj = self_ref()
                    if obj is None:  # pragma: no cover - instance already gone
                        raise ReferenceError(
                            f"Cached method {method.__name__!r} called after "
                            f"its instance was garbage collected")
                    return method(obj, *cache_args, **cache_kwargs)

                # Store directly in __dict__ so Property-based __setattr__ is
                # not required, and so the cache is collected with the instance.
                self.__dict__[cache_attr] = cache
            return cache(*args, **kwargs)

        def cache_info(self):
            cache = self.__dict__.get(cache_attr)
            if cache is None:
                # Empty stats consistent with functools.lru_cache before any calls.
                empty = functools.lru_cache(maxsize=maxsize, typed=typed)(lambda: None)
                return empty.cache_info()
            return cache.cache_info()

        def cache_clear(self):
            cache = self.__dict__.get(cache_attr)
            if cache is not None:
                cache.cache_clear()

        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        return wrapper

    return decorator
