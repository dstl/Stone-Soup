"""Caching helpers for Stone Soup components."""
from __future__ import annotations

import functools
import weakref


def instance_lru_cache(maxsize: int | None = 128, typed: bool = False):
    """Per-instance :func:`functools.lru_cache` for methods.

    Decorating an instance method with :func:`functools.lru_cache` directly
    creates a single class-level cache whose keys include ``self``, which keeps
    instances alive after they would otherwise be garbage collected (see issue
    #801).

    Instead, this decorator keeps a :class:`weakref.WeakKeyDictionary` on the
    decorator itself, mapping each instance to its own LRU cache. Nothing is
    stored on the instance, so instances remain picklable and copyable, and an
    instance's cache is discarded automatically when the instance is garbage
    collected.

    The wrapped method must still be called with hashable arguments (aside from
    ``self``), as with :func:`functools.lru_cache`. The instance itself must be
    hashable and support weak references, which is the case for all
    :class:`~.Base` subclasses.

    The decorated method exposes ``cache_info(instance)`` and
    ``cache_clear(instance=None)``; calling ``cache_clear()`` with no instance
    clears the caches of all instances.
    """

    def decorator(method):
        caches = weakref.WeakKeyDictionary()

        def _make_cache(instance):
            # Only hold a weak reference to the instance, so the cache (the
            # dictionary value) doesn't keep the dictionary key alive.
            instance_ref = weakref.ref(instance)

            @functools.lru_cache(maxsize=maxsize, typed=typed)
            def cache(*args, **kwargs):
                obj = instance_ref()
                if obj is None:  # pragma: no cover - entry removed with instance
                    raise ReferenceError(
                        f"Cached method {method.__name__!r} called after its "
                        f"instance was garbage collected")
                return method(obj, *args, **kwargs)

            return cache

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            try:
                cache = caches[self]
            except KeyError:
                cache = caches.setdefault(self, _make_cache(self))
            return cache(*args, **kwargs)

        def cache_info(instance):
            """Return :func:`functools.lru_cache` statistics for `instance`."""
            cache = caches.get(instance)
            if cache is None:
                # Statistics of an unused cache, consistent with lru_cache
                cache = functools.lru_cache(maxsize=maxsize, typed=typed)(method)
            return cache.cache_info()

        def cache_clear(instance=None):
            """Clear the cache for `instance`, or for all instances if None."""
            if instance is None:
                caches.clear()
            else:
                caches.pop(instance, None)

        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        wrapper._instance_caches = caches
        return wrapper

    return decorator
