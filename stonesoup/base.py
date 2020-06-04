# -*- coding: utf-8 -*-
"""Provides base for Stone Soup components.

To aid creation of components in Stone Soup, a declarative approach is used to
declare properties of components. These declared properties are then used to
generate the signature for the class, populate documentation, and generate
forms for the user interface.

An example would be:

.. code-block:: python

    class Foo(Base):
        '''Example Foo class'''
        foo = Property(str, doc="foo string parameter")
        bar = Property(int, default=10, doc="bar int parameter, default is 10")


This is equivalent to the following:

.. code-block:: python

    class Foo:
        '''Example Foo class

        Parameters
        ----------
        foo : str
            foo string parameter
        bar : int, optional
            bar int parameter, default is 10
        '''

        def __init__(self, foo, bar=10):
            self.foo = foo
            self.bar = 10

.. note::

    The init method is actually part of :class:`Base` class so in the case of
    having to customise initialisation, :func:`super` should be used e.g.:

    .. code-block:: python

        class Foo(Base):
        '''Example Foo class'''
        foo = Property(str, doc="foo string parameter")
        bar = Property(int, default=10, doc="bar int parameter, default is 10")

        def __init__(self, foo, bar=bar.default, *args, **kwargs):
            if bar < 0:
                raise ValueError("...")
            super().__init__(foo, bar, *args, **kwargs)


"""
import inspect
import sys
from abc import ABCMeta
from collections import OrderedDict
from types import MappingProxyType


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

    A description string can also be provided which will be rendered in the
    documentation.

    A property can be specified as read only using the (optional) ``readonly``
    flag. Such properties can be written only once (when the parent object is
    instantiated). Any subsequent write raises an ``AttributeError``

    Parameters
    ----------
    cls : class
        A Python class.
    default : any, optional
        A default value, which should be same type as class or None. Defaults
        to :class:`inspect.Parameter.empty` (alias :attr:`Property.empty`)
    doc : str, optional
        Doc string for property
    readonly : bool, optional

    Attributes
    ----------
    cls
    default
    doc
    empty : :class:`inspect.Parameter.empty`
        Alias to :class:`inspect.Parameter.empty`
    """
    empty = inspect.Parameter.empty
    _property_name = None

    def __init__(self, cls, *, default=inspect.Parameter.empty, doc=None,
                 readonly=False):
        self.cls = cls
        self.default = default
        self.doc = self.__doc__ = doc
        # Fix for when ":" in doc string being interpreted as type in NumpyDoc
        if doc is not None and ':' in doc:
            self.__doc__ = ": " + doc
        self._setter = None
        self._getter = None
        self._deleter = None

        if readonly:
            def _setter(instance, value):
                # if the value hasn't been set before then we can set it once
                if not hasattr(instance, self._property_name):
                    setattr(instance, self._property_name, value)
                else:
                    # if the value has been set, raise an AttributeError
                    raise AttributeError(
                        '{} is readonly'.format(self._property_name))

            self._setter = _setter

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if self._getter is None:
            return getattr(instance, self._property_name)
        else:
            return self._getter(instance)

    def __set__(self, instance, value):
        if self._setter is None:
            setattr(instance, self._property_name, value)
        else:
            self._setter(instance, value)

    def __delete__(self, instance):
        if self._deleter is None:
            delattr(instance, self._property_name)
        else:
            self._deleter(instance, self._property_name)

    def __set_name__(self, owner, name):
        if not isinstance(owner, BaseMeta):
            raise AttributeError("Cannot use Property on this class type")
        self._property_name = "_property_{}".format(name)

    def deleter(self, method):  # real signature unknown
        """ Descriptor to change the deleter on a property. """
        self._deleter = method

    def getter(self, method):  # real signature unknown
        """ Descriptor to change the getter on a property. """
        self._getter = method

    def setter(self, method):  # real signature unknown
        """ Descriptor to change the setter on a property. """
        self._setter = method


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

    @classmethod
    def __prepare__(mcls, name, bases, **kwargs):
        return OrderedDict()

    def __new__(mcls, name, bases, namespace):
        if '__init__' not in namespace:
            # Must replace init so we don't overwrite parent class's
            # and blank line below so this doesn't become its docstring!

            def __init__(self, *args, **kwargs):
                super(cls, self).__init__(*args, **kwargs)
            namespace['__init__'] = __init__
        cls = super().__new__(mcls, name, bases, namespace)

        cls._subclasses = set()
        cls._properties = OrderedDict()
        # Update subclass lists, and update properties (in reverse order)
        for bcls in reversed(cls.mro()[1:]):
            if type(bcls) is mcls:
                bcls._subclasses.add(cls)
                cls._properties.update(bcls._properties)
        cls._properties.update(
            (key, value) for key, value in namespace.items()
            if isinstance(value, Property))
        for name in list(cls._properties):
            # Remove items which are no longer properties
            if name in namespace and not isinstance(namespace[name], Property):
                del cls._properties[name]
                continue
            # Optional arguments must follow mandatory
            if cls._properties[name].default is not Property.empty:
                cls._properties.move_to_end(name)

        if sys.version_info <= (3, 6):  # pragma: no cover
            for name, property_ in cls._properties.items():
                property_.__set_name__(cls, name)

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
        return frozenset(cls._subclasses)

    @property
    def properties(cls):
        """Set of properties required to initialise the class"""
        return MappingProxyType(cls._properties)


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

    def __repr__(self):
        params = ("{}={!r}".format(name, getattr(self, name))
                  for name in type(self).properties)
        return "{}({})".format(type(self).__name__, ", ".join(params))
