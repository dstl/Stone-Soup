# -*- coding: utf-8 -*-
from abc import ABCMeta
from collections import OrderedDict
from importlib import import_module


class Property:
    """Wrapper class for Property definition on components.

    Parameters
    ----------
    cls : class
        A Python class.
    default : any, optional
        A default value, which should be same type as class or None.
    """

    def __init__(self, cls, **kwargs):
        if 'default' in kwargs:
            self.default = kwargs.pop('default')


class BaseMeta(ABCMeta):
    def __new__(mcls, name, bases, namespace, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        cls._subclasses = set()
        cls._properties = OrderedDict()
        for bcls in reversed(cls.mro()[1:]):
            if type(bcls) is mcls:
                bcls._subclasses.add(cls)
                cls._properties.update(bcls._properties)
                if bcls.__module__.endswith(".base"):
                    # Add to base's parent module
                    module = import_module("..", bcls.__module__)
                    setattr(module, cls.__name__, cls)
        cls._properties.update(
            {key: value for key, value in namespace.items()
             if isinstance(value, Property)})
        for name in list(cls._properties):
            if hasattr(cls._properties[name], 'default'):
                cls._properties.move_to_end(name)
        return cls

    @property
    def subclasses(cls):
        """Set of subclasses for the class"""
        return cls._subclasses.copy()

    @property
    def properties(cls):
        """Set of properties required to initialise the class"""
        return cls._properties.copy()


class Base(metaclass=BaseMeta):
    """Base class for framework components."""

    def __init__(self, *args, **kwargs):
        properties = self.__class__.properties
        if len(args) > len(properties):
            raise TypeError("got too many positional arguments")

        for arg, name in zip(args, list(properties)):
            if name in kwargs:
                raise TypeError("got multiple values for argument {!r}".format(
                    name))
            setattr(self, name, arg)
            del properties[name]

        for name, property_ in properties.items():
            try:
                value = kwargs.pop(name)
            except KeyError:
                try:
                    value = property_.default
                except AttributeError:
                    raise TypeError("missing required argument: {!r}".format(
                        name))
            setattr(self, name, value)

        for name in kwargs:
            raise TypeError("got an unexpected keyword argument {!r}".format(
                name))
