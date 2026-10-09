"""Provides an ability to serialise Stone Soup objects.

Two formats are supported:

- YAML (:class:`~.serialise.yaml.YAML`), a human readable format, well suited to configuration
  files.
- CBOR (:class:`~.serialise.cbor.CBOR`), a binary format, which is faster to read and write,
  and more compact, well suited to recording data. This requires the optional cbor2_ dependency.

.. _cbor2: https://cbor2.readthedocs.io/
"""
from .yaml import YAML

__all__ = ['YAML', 'CBOR']


def __getattr__(name):
    # CBOR imported lazily, as cbor2 is an optional dependency
    if name == 'CBOR':
        from .cbor import CBOR
        return CBOR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
