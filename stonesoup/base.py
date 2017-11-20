# -*- coding: utf-8 -*-
from abc import ABCMeta
from importlib import import_module


class BaseMeta(ABCMeta):
    def __new__(mcls, name, bases, namespace, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        cls._subclasses = set()
        for bcls in cls.mro()[1:]:
            if type(bcls) is mcls:
                bcls._subclasses.add(cls)
                if bcls.__module__.endswith(".base"):
                    # Add to base's parent module
                    module = import_module("..", bcls.__module__)
                    setattr(module, cls.__name__, cls)
        return cls

    @property
    def subclasses(cls):
        """Set of subclasses for the class"""
        return cls._subclasses.copy()
