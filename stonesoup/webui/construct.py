# -*- coding: utf-8 -*-
from collections.abc import Mapping
import datetime

import dateutil.parser
import numpy as np
import pytimeparse

from ..base import Base, ListProperty
from ..types import Type
from ..types.array import Array

from .. import (  # noqa: F401
    dataassociator, deleter, feeder, hypothesiser, initiator, metricgenerator,
    mixturereducer, models, predictor, reader, resampler, simulator, tracker,
    types, updater, writer)

classes = {"{0.__module__}.{0.__name__}".format(cls): cls
           for cls in Base.subclasses}


class ConstructError(Exception):
    """Construction error for generating tracker from UI.

    Parameters
    ----------
    error : Exception
        Original error raised.
    key : str
        Key which raised original error, to form last entry in :attr:`keys`

    Attributes
    ----------
    keys : list of str
        List of keys mapping back to object which raised the error.
    """
    def __init__(self, error, key):
        super().__init__(error)
        self.error = error
        self.keys = [key]

    def __str__(self):
        return "{}: [{}]".format(super().__str__(), "][".join(self.keys))


def construct(component_map):
    cls = classes[component_map['__class__']]
    properties = {}
    for key, value in component_map.items():
        if key.startswith('__'):
            continue
        try:
            property_ = getattr(cls, key)
        except AttributeError as err:
            raise ConstructError(err, key)

        # Need to construct object
        if isinstance(value, Mapping):
            if '__class__' in value:
                try:
                    properties[key] = construct(value)
                except ConstructError as err:
                    err.keys.insert(0, key)
                    raise err
            else:
                # Not defined when UI set to None for optional properties
                properties[key] = None
        elif isinstance(property_, ListProperty):
            properties[key] = []
            for item_num, item in enumerate(value):
                if isinstance(item, Mapping) and '__class__' in item:
                    try:
                        properties[key].append(construct(item))
                    except ConstructError as err:
                        err.keys.insert(0, key)
                        err.keys.insert(0, item_num)
                        raise err
                elif not isinstance(item, Mapping):
                    try:
                        properties[key].append(
                            construct_type(property_.cls, item))
                    except (TypeError, ValueError) as err:
                        new_err = ConstructError(err, item_num)
                        new_err.keys.insert(0, key)
                        raise new_err
        else:
            try:
                properties[key] = construct_type(property_.cls, value)
            except (TypeError, ValueError) as err:
                raise ConstructError(err, key)

    return cls(**properties)


def construct_type(cls, value):
    if issubclass(cls, Type):
        return cls(value)
    elif issubclass(cls, np.ndarray) and not issubclass(cls, Array):
        return np.array(value)
    elif issubclass(cls, datetime.datetime):
        return dateutil.parser.parse(value)
    elif issubclass(cls, datetime.timedelta):
        return datetime.timedelta(seconds=pytimeparse.parse(value))
    else:
        return cls(value)
