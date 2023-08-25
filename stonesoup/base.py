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
import textwrap
from reprlib import Repr
from abc import ABCMeta
from collections import OrderedDict
from copy import copy
from functools import cached_property
from types import MappingProxyType
# from typing import TYPE_CHECKING


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

    Property also can be used in similar way to Python standard `property`
    using `getter`, `setter` and `deleter` decorators.

    Parameters
    ----------
    cls : class, optional
        A Python class. Where not specified, a type annotation is required,
        and providing both will raise an error.
    default : any, optional
        A default value, which should be same type as class or None. Defaults
        to :class:`inspect.Parameter.empty` (alias :attr:`Property.empty`)
    doc : str, optional
        Doc string for property
    readonly : bool, optional
        If `True`, then property can only be set during initialisation.

    Attributes
    ----------
    cls
    default
    doc
    readonly
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
        self._clear_cached = set()

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if self._getter is None:
            return getattr(instance, self._property_name)
        else:
            return self._getter(instance)

    def __set__(self, instance, value):
        if self.readonly and hasattr(instance, self._property_name):
            # if the value has been set, raise an AttributeError
            raise AttributeError('{} is readonly'.format(self._property_name))

        for cached_value in self._clear_cached:
            if cached_value in instance.__dict__:
                del instance.__dict__[cached_value]

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


def _format_note(property_names):
    multiple = len(property_names) > 1
    prop_str = [f":attr:`{prop_name}`" for prop_name in property_names]
    return textwrap.dedent(f"""\


        Note
        ----
        This will be cached until {", ".join(prop_str[:-1])}
        {"or " if multiple else ""}{prop_str[-1]} {"are" if multiple else "is" } replaced.
        """)


def clearable_cached_property(*property_names: str):
    """cached property which is cleared on provided properties being modified

    This decorator will use the standard library functools.cached_property
    but will automatically clear this cache if the provided Stone Soup properties are
    set to a different value.

    Care should be made where a Stone Soup Property is a mutable type, that the cache
    will not be clear as there is no way to track changes of mutable types.
    """

    def decorator(func):
        if func.__doc__ is None:
            func.__doc__ = ""
        func.__doc__ = func.__doc__ + _format_note(property_names)
        cached_method = cached_property(func)
        cached_method._property_names = property_names
        return cached_method
    return decorator


class BaseRepr(Repr):
    def __init__(self):
        self.maxlevel = 10
        self.maxtuple = 10
        self.maxlist = 10
        self.maxarray = 10
        self.maxdict = 20
        self.maxset = 10
        self.maxfrozenset = 10
        self.maxdeque = 10
        self.maxstring = 500
        self.maxlong = 40
        self.maxother = 50000

    def repr_list(self, obj, level):
        if len(obj) > self.maxlist:
            max_len = round(self.maxlist/2)
            first = ',\n '.join(self.repr1(x, level - 1) for x in obj[:max_len])
            last = ',\n '.join(self.repr1(x, level - 1) for x in obj[-max_len:])
            return f'[{first},\n ...\n ...\n ...\n {last}]'
        else:
            return '[{}]'.format(',\n '.join(self.repr1(x, level - 1) for x in obj))

    @classmethod
    def whitespace_remove(cls, maxlen_whitespace, val):
        """Remove excess whitespace, replacing with ellipses"""
        large_whitespace = ' ' * (maxlen_whitespace+1)
        fixed_whitespace = ' ' * maxlen_whitespace
        if large_whitespace in val:
            excess = val.find(large_whitespace)  # Find the excess whitespace
            line_end = ''.join(val[excess:].partition('\n')[1:])
            val = ''.join([val[0:excess], fixed_whitespace, '...', line_end])
            return cls.whitespace_remove(maxlen_whitespace, val)
        else:
            return val


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

    _repr = BaseRepr()

    def __new__(mcls, name, bases, namespace):

        properties = OrderedDict()
        # Update properties from direct bases (in reverse order as
        # first defined class must take precedence, and dictionary update overwrites)
        for base_class in reversed(bases):
            if type(base_class) is mcls:
                if base_class in bases:
                    properties.update(base_class._properties)

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

                if not (isinstance(value.cls, type)
                        or getattr(value.cls, '__module__', "") == 'typing'
                        or value.cls == name
                        or isinstance(value.cls, str)):
                    print('name = ', name)
                    print('value.cls = ', value.cls)

                    raise ValueError(f'Invalid type specification ({str(value.cls)}) '
                                     f'for property {key} of class {name}')

                # Finally set property.
                properties[key] = value

            elif key in properties:
                # New definition of "key" which isn't a Property any more.
                del properties[key]

        for key, value in list(namespace.items()):
            if isinstance(value, cached_property):
                for prop_name in getattr(value, '_property_names', []):
                    new_prop = copy(properties[prop_name])
                    new_prop._clear_cached = new_prop._clear_cached | {key}
                    properties[prop_name] = namespace[prop_name] = new_prop

        for prop_name in list(properties):
            # Optional arguments must follow mandatory
            if properties[prop_name].default is not Property.empty:
                properties.move_to_end(prop_name)

        if '__init__' not in namespace:
            # Must replace init so we don't overwrite parent class's
            # and blank line below so this doesn't become its docstring!

            def __init__(self, *args, **kwargs):
                super(cls, self).__init__(*args, **kwargs)
            namespace['__init__'] = __init__

        namespace['_properties'] = properties
        namespace['_subclasses'] = set()

        cls = super().__new__(mcls, name, bases, namespace)

        for property_ in cls.properties.values():
            if isinstance(property_.cls, str) and property_.cls == name:
                property_.cls = cls

        for base_class in cls.mro()[1:]:
            if type(base_class) is mcls:
                base_class._subclasses.add(cls)

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
                default=property_.default, annotation=property_.cls)
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
        cls = type(self)
        prop_iter = iter(cls.properties.items())

        for arg in args:
            try:
                name, _ = next(prop_iter)
            except StopIteration:
                raise TypeError('too many positional arguments') from None
            if name in kwargs:
                raise TypeError(f'multiple values for argument {name!r}')
            setattr(self, name, arg)

        for name, prop in prop_iter:
            value = kwargs.pop(name, prop.default)
            if value is Property.empty:
                raise TypeError(f'missing a required argument: {name!r}')
            setattr(self, name, value)

        if kwargs:
            raise TypeError(f'got an unexpected keyword argument {next(iter(kwargs))!r}')

    def __repr__(self):
        whitespace = ' ' * 4  # Indents every line
        max_len_whitespace = 80  # Ensures whitespace doesn't get rid of space on RHS too much
        max_out = 50000  # Keeps total length from being too excessive
        params = []
        for name in type(self).properties:
            value = getattr(self, name)
            extra_whitespace = ' ' * (len(name) + 1) + whitespace  # Lines up rows of arrays
            repr_value = Base._repr.repr(value)
            if '\n' in repr_value:
                value = repr_value.replace('\n', '\n' + extra_whitespace)
            params.append(f'{whitespace}{name}={value}')
        value = "{}(\n{})".format(type(self).__name__, ",\n".join(params))
        rep = Base._repr.whitespace_remove(max_len_whitespace, value)
        truncate = '\n...\n...  (truncated due to length)\n...'
        return ''.join([rep[:max_out], truncate]) if len(rep) > max_out else rep
