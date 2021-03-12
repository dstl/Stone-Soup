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
        foo: str = Property(doc="foo string parameter")
        bar: int = Property(default=10, doc="bar int parameter, default is 10")


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
        foo: str = Property(doc="foo string parameter")
        bar: int = Property(default=10, doc="bar int parameter, default is 10")

        def __init__(self, foo, bar=bar.default, *args, **kwargs):
            if bar < 0:
                raise ValueError("...")
            super().__init__(foo, bar, *args, **kwargs)


"""
import inspect
import weakref
from abc import ABCMeta
from collections import OrderedDict
from copy import deepcopy, copy
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

    def __init__(self, cls=None, *, default=inspect.Parameter.empty, doc=None,
                 readonly=False):
        self.cls = cls
        self.default = default
        self.doc = self.__doc__ = doc
        # Fix for when ":" in doc string being interpreted as type in NumpyDoc
        if doc is not None and ':' in doc:
            self.__doc__ = ": " + doc
        self._property_name = None
        self._setter = None
        self._getter = None
        self._deleter = None
        self.readonly = readonly

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if self._getter is None:
            return getattr(instance, self._property_name)
        else:
            return self._getter(instance)

    def __set__(self, instance, value):
        if self.readonly:
            if not hasattr(instance, self._property_name):
                setattr(instance, self._property_name, value)
            else:
                # if the value has been set, raise an AttributeError
                raise AttributeError(
                    '{} is readonly'.format(self._property_name))

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
        new_property = copy(self)
        new_property._deleter = method
        return new_property

    def getter(self, method):  # real signature unknown
        """ Descriptor to change the getter on a property. """
        new_property = copy(self)
        new_property._getter = method
        return new_property

    def setter(self, method):  # real signature unknown
        """ Descriptor to change the setter on a property. """
        new_property = copy(self)
        new_property._setter = method
        return new_property


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
        # Update subclass lists, and update properties from direct bases (in reverse order as
        # first defined class must take precedence, and dictionary update overwrites)
        for base_class in reversed(cls.mro()[1:]):
            if type(base_class) is mcls:
                base_class._subclasses.add(cls)
                if base_class in bases:
                    cls._properties.update(base_class._properties)

        for key, value in namespace.items():
            if isinstance(value, Property):
                annotation_cls = namespace.get('__annotations__', {}).get(key, None)
                if value.cls is not None and annotation_cls is not None:
                    raise ValueError(f'Type was specified both by type hint '
                                     f'({str(annotation_cls)}) and argument ({str(value.cls)}) '
                                     f'for property {key} of class {name}')
                elif value.cls is None and annotation_cls is not None:
                    value.cls = annotation_cls
                elif value.cls is not None and annotation_cls is None:
                    # Just use value.cls in this case
                    pass
                elif value.cls is None and annotation_cls is None:
                    raise ValueError(f'Type was not specified '
                                     f'for property {key} of class {name}')

                if isinstance(value.cls, str) and value.cls == name:
                    value.cls = cls

                if not (isinstance(value.cls, type)
                        or getattr(value.cls, '__module__', "") == 'typing'):
                    raise ValueError(f'Invalid type specification ({str(value.cls)}) '
                                     f'for property {key} of class {cls.__name__}')

                # Finally set property.
                cls._properties[key] = value

            elif key in cls._properties:
                # New definition of "key" which isn't a Property any more.
                del cls._properties[key]

        for prop_name in list(cls._properties):
            # Optional arguments must follow mandatory
            if cls._properties[prop_name].default is not Property.empty:
                cls._properties.move_to_end(prop_name)

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

    def register(cls, subclass):
        cls._subclasses.add(subclass)
        return super().register(subclass)

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

    def __deepcopy__(self, memodict={}):
        # Create a new class
        new = object.__new__(type(self))
        memodict[id(self)] = new   # add the new class to the memo
        # Insert a deepcopy of all instance attributes
        new.__dict__.update(deepcopy(self.__dict__, memodict))
        # Manually update any weakrefs to point to copies, if they exist.
        for name, prop in new.__dict__.items():
            if isinstance(prop, weakref.ref):
                original_target = prop()  # call the weakref to get a reference
                try:
                    # if we are copying the parent as well, the copy should be in memodict
                    copy_of_target = memodict[id(original_target)]
                except KeyError:
                    # if we can't find the parent, then leave the original ref in place
                    pass
                else:
                    new.__setattr__(name, weakref.ref(copy_of_target))
        return new
