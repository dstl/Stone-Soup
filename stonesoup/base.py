# -*- coding: utf-8 -*-
import inspect
from abc import ABCMeta
from collections import OrderedDict


class Property:
    """Property(cls, default=inspect.Parameter.empty)
    Property class for definition of attributes on component classes.

    A class must be provided such that the framework is aware of how components
    are put together to create a valid run within the framework. Additionally,
    the class is used by the user interface to generate configuration options
    to the users. The class is not used for any type checking, in the spirit of
    Python's duck typing.

    A default value can be specified to signify the property on the class is
    optional. As ``None`` and ``False`` are reasonable default values,
    :class:`inspect.Parameter.empty` is used to signify the argument is
    mandatory. (Also aliased to :attr:`Property.empty` for ease)

    Parameters
    ----------
    cls : class
        A Python class.
    default : any, optional
        A default value, which should be same type as class or None. Defaults
        to :class:`inspect.Parameter.empty` (alias :attr:`Property.empty`)
    """
    empty = inspect.Parameter.empty
    """Alias to :class:`inspect.Parameter.empty`"""

    def __init__(self, cls, **kwargs):
        self.cls = cls
        try:
            self.default = kwargs.pop('default')
        except KeyError:
            self.default = self.empty


class BaseMeta(ABCMeta):
    """Base metaclass for Stone Soup components.

    This metaclass enables the use of the :class:`Property` class to define
    attributes of a class. This includes generation of the init method
    signature.

    The init method signature if defined on a class, the arguments must match
    as declared. However, keyword only arguments can be added to the init
    method if required, as these won't effect the use of the class in the
    framework.
    """
    def __new__(mcls, name, bases, namespace, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)

        cls._subclasses = set()
        cls._properties = OrderedDict()
        # Update subclass lists, and update properties (in reverse order)
        for bcls in reversed(cls.mro()[1:]):
            if type(bcls) is mcls:
                bcls._subclasses.add(cls)
                cls._properties.update(bcls._properties)
        cls._properties.update(
            {key: value for key, value in namespace.items()
             if isinstance(value, Property)})
        # Optional arguments must follow mandatory
        for name in list(cls._properties):
            if cls._properties[name].default is not Property.empty:
                cls._properties.move_to_end(name)

        if '__init__' in namespace:
            cls._validate_init()
        cls._generate_signature()

        return cls

    def _validate_init(cls):
        """Validates custom init's arguments."""
        init_signature = inspect.signature(cls.__init__)

        declared_names = list(cls._properties)
        positional_names = [
            parameter.name
            for parameter in init_signature.parameters.values()
            if parameter.kind in (
                parameter.POSITIONAL_ONLY,
                parameter.POSITIONAL_OR_KEYWORD)][1:]  # Ignore 'self' (item 0)
        if positional_names != declared_names[:len(positional_names)]:
            raise TypeError("init arguments don't match declared properties: "
                            "arguments do not match or wrong order")

        has_var_positional = any(
            parameter.kind == parameter.VAR_POSITIONAL
            for parameter in init_signature.parameters.values())
        has_var_keyword = any(
            parameter.kind == parameter.VAR_KEYWORD
            for parameter in init_signature.parameters.values())
        if positional_names != declared_names and not (
                    has_var_positional and has_var_keyword):
            raise TypeError("init arguments don't match declared properties: "
                            "missing argument (or *args and **kwargs missing)")

        keyword_parameters = [
            parameter
            for parameter in init_signature.parameters.values()
            if parameter.kind == parameter.KEYWORD_ONLY]
        if any(parameter.default is parameter.empty
               for parameter in keyword_parameters):
            raise TypeError("new keyword arguments must have default value")

    def _generate_signature(cls):
        """Generates __init__ signature with declared properties."""
        init_signature = inspect.signature(cls.__init__)
        parameters = [next(iter(init_signature.parameters.values()))]  # 'self'
        parameters.extend(
            inspect.Parameter(
                name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=property_.default)
            for name, property_ in cls._properties.items())
        parameters.extend(
            parameter
            for parameter in init_signature.parameters.values()
            if parameter.kind == parameter.KEYWORD_ONLY)
        cls.__init__.__signature__ = init_signature.replace(
            parameters=parameters)

    @property
    def subclasses(cls):
        """Set of subclasses for the class"""
        return cls._subclasses.copy()

    @property
    def properties(cls):
        """Set of properties required to initialise the class"""
        return cls._properties.copy()


class Base(metaclass=BaseMeta):
    """Base class for framework components.

    This is the base class which should be used for any Stone Soup components.
    Building on the :class:`BaseMeta` this provides a init method which
    populates the declared properties with their values.

    Subclasses can override this method, but they should either call this via
    :func:`super()` or ensure they manually populated the properties as
    declared."""

    def __init__(self, *args, **kwargs):
        init_signature = inspect.signature(self.__init__)
        bound_arguments = init_signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        for name, value in bound_arguments.arguments.items():
            setattr(self, name, value)
